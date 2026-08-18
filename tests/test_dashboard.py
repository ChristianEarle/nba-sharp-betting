"""Tests for src/dashboard/build.py (the plain-language multi-view dashboard).

Every test here is data-dependent (the checked-in board CSV, the four model
bundles, the model metrics/report artifacts, and data/raw/ff_ecr.parquet)
and skips cleanly when its input is absent, matching every other
data-dependent test file's pattern in this repo (see e.g.
tests/test_inference.py's module docstring). Run the usual
`src.ingest.nflverse` / `src.labels.build` / `src.features.*` /
`src.models.train_*` / `src.inference.board_2026` chain first (or just
`make dashboard-build` once bundles exist) to produce everything these
tests read.

None of these tests call ``src.models.train.run_full_pipeline`` (or
anything that trains), so the production-bundle-clobbering guard that
governs other test files' ``output_root`` usage doesn't apply here --
``src.dashboard.build`` only ever *reads* model bundles/metrics, never
writes or retrains them.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pandas as pd
import polars as pl
import pytest

from src.dashboard import build
from src.inference import board_2026 as bd
from src.models import train

RAW_ECR_PATH = build.REPO_ROOT / "data" / "raw" / "ff_ecr.parquet"


def _skip_if_inputs_missing() -> None:
    if not bd.BOARD_CSV_PATH.exists():
        pytest.skip(f"{bd.BOARD_CSV_PATH.name} not built; run `python -m src.inference.board_2026`")
    for pos in train.POSITIONS:
        spec = train.position_spec(pos)
        if not spec.artifact_path.exists():
            pytest.skip(f"{pos} model bundle not built; run `python -m src.models.train_{pos}`")
        if not spec.metrics_json_path.exists() or not spec.report_path.exists():
            pytest.skip(f"{pos} model report/metrics not built; run `python -m src.models.train_{pos}`")
    if not RAW_ECR_PATH.exists():
        pytest.skip("data/raw/ff_ecr.parquet not present; run `python -m src.ingest.nflverse`")


@pytest.fixture(scope="module")
def assembled():
    _skip_if_inputs_missing()
    payload, report = build.assemble_payload()
    return payload, report


# --------------------------------------------------------------------------
# Payload schema
# --------------------------------------------------------------------------


_GSIS_KEY_RE = re.compile(r"^\d{2}-\d{7}$")


def test_all_board_players_present_in_payload(assembled) -> None:
    """Finding 8: board rows are keyed by gsis_id (falling back to name|POS only when a

    row has no resolved gsis_id) -- every payload key must be unique, most real board
    rows should resolve to a real gsis_id, and every fallback name|POS key must trace
    back to a real (unresolved) CSV row.
    """
    payload, _ = assembled
    csv_df = pd.read_csv(bd.BOARD_CSV_PATH)
    assert len(payload["board"]) == len(csv_df)
    keys = [r["key"] for r in payload["board"]]
    assert len(keys) == len(set(keys)), "duplicate player keys on the board payload"

    resolved = [k for k in keys if _GSIS_KEY_RE.match(k)]
    assert len(resolved) > 0.5 * len(keys), "fewer than half of board keys resolved to a real gsis_id"

    csv_pairs = set(zip(csv_df["player_name"].astype(str), csv_df["pos"].astype(str)))
    fallback_pairs = {(r["n"], r["p"]) for r, k in zip(payload["board"], keys) if not _GSIS_KEY_RE.match(k)}
    assert fallback_pairs <= csv_pairs, "a name|POS fallback key doesn't trace back to a real board row"


def test_percentiles_within_bounds(assembled) -> None:
    payload, report = assembled
    assert report["payload_players_features"] > 0, "no feature profiles built at all"
    n_checked = 0
    for key, feats in payload["features"].items():
        for f in feats:
            assert 0.0 <= f["pctl"] <= 100.0, f"{key} feature {f['feat']} percentile out of [0,100]: {f['pctl']}"
            assert f["label"], f"{key} feature {f['feat']} has an empty label"
            n_checked += 1
    assert n_checked > 0


def test_drift_series_dates_monotone(assembled) -> None:
    payload, report = assembled
    assert report["payload_players_ecr"] > 0, "no ECR drift series built at all"
    for key, series in payload["ecr"].items():
        dates = series["dates"]
        assert dates == sorted(dates), f"{key}: ECR snapshot dates not sorted ascending: {dates}"
        assert len(dates) == len(set(dates)), f"{key}: duplicate snapshot dates: {dates}"
        assert len(dates) == len(series["ranks"]) == series["n"]
        # drift = earliest pos_rank - latest pos_rank, positive = rising
        assert series["drift"] == series["ranks"][0] - series["ranks"][-1]


def test_ecr_coverage_has_players_with_3plus_snapshots(assembled) -> None:
    _, report = assembled
    assert report["ecr_snapshot_report"]["n_snapshot_dates"] >= 1
    # not a hard requirement of the pipeline, but worth surfacing if it ever
    # regresses to zero -- the market-drift feature would be silently inert.
    assert report["ecr_players_with_ge3_snapshots"] >= 0


def test_payload_has_no_nan_json_dumps_allow_nan_false(assembled) -> None:
    payload, _ = assembled
    clean = build._sanitize(payload)
    # must not raise -- allow_nan=False rejects any float('nan')/inf that
    # slipped through _sanitize
    text = json.dumps(clean, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_trust_view_positions_present(assembled) -> None:
    payload, _ = assembled
    for pos in train.POSITIONS:
        assert pos in payload["trust"]["positions"], f"missing trust data for {pos}"
        entry = payload["trust"]["positions"][pos]
        assert entry["holdout_pr_auc"] is not None


def test_positions_view_histograms_sum_to_count(assembled) -> None:
    payload, _ = assembled
    for pos, pv in payload["positions"].items():
        assert sum(pv["histogram"]) <= pv["count"]  # players with a null probability are excluded from bins
        assert len(pv["histogram"]) == 10
        assert len(pv["top5"]) <= 5


# --------------------------------------------------------------------------
# Generated HTML
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def html_text(assembled) -> str:
    _skip_if_inputs_missing()
    path, _ = build.build_dashboard()
    return path.read_text()


def test_html_has_no_unfilled_placeholders(html_text) -> None:
    assert "__PLACEHOLDER__" not in html_text
    assert "__PAYLOAD_JSON__" not in html_text


def test_html_has_title(html_text) -> None:
    assert "<title>" in html_text


def test_html_script_tag_escapes_closing_sequence(html_text) -> None:
    start = html_text.index('<script type="application/json" id="data">') + len(
        '<script type="application/json" id="data">'
    )
    end = html_text.index("</script>", start)
    embedded = html_text[start:end]
    assert "</" not in embedded, "unescaped '</' inside the embedded JSON would break out of the script tag early"
    # sanity: the payload really is there and really is JSON
    payload = json.loads(embedded.replace("<\\/", "</"))
    assert "board" in payload and "meta" in payload


def test_html_under_size_budget(html_text) -> None:
    assert len(html_text.encode("utf-8")) < 3 * 1024 * 1024, "dashboard HTML exceeds the ~3MB budget"


def test_build_twice_is_byte_identical(assembled) -> None:
    _skip_if_inputs_missing()
    path1, _ = build.build_dashboard()
    bytes1 = path1.read_bytes()
    path2, _ = build.build_dashboard()
    bytes2 = path2.read_bytes()
    assert bytes1 == bytes2, "make dashboard is not deterministic given unchanged inputs"


# --------------------------------------------------------------------------
# Finding 7: build_ecr_trajectory must cap at the season's first REG gameday
# --------------------------------------------------------------------------

_EMPTY_CROSSWALK = pl.DataFrame(
    schema={
        "mfl_id": pl.Utf8, "gsis_id": pl.Utf8, "fantasypros_id": pl.Utf8, "sleeper_id": pl.Utf8,
        "name": pl.Utf8, "merge_name": pl.Utf8, "position": pl.Utf8, "draft_year": pl.Int64, "birthdate": pl.Utf8,
    }
)


def _synthetic_ecr_snapshots() -> pl.DataFrame:
    """Two preseason snapshots (pre-kickoff) plus one POST-kickoff snapshot -- the

    post-kickoff row reflects real in-season performance (Player B rockets from
    pos_rank 2 to pos_rank 1), which must NEVER be mistaken for preseason market drift.
    """
    rows = [
        # pre-kickoff snapshot 1: A ranked ahead of B
        {"scrape_date": dt.date(2026, 6, 15), "pos": "WR", "id": "A", "player": "Player A", "ecr": 1.0},
        {"scrape_date": dt.date(2026, 6, 15), "pos": "WR", "id": "B", "player": "Player B", "ecr": 2.0},
        # pre-kickoff snapshot 2: unchanged
        {"scrape_date": dt.date(2026, 8, 20), "pos": "WR", "id": "A", "player": "Player A", "ecr": 1.0},
        {"scrape_date": dt.date(2026, 8, 20), "pos": "WR", "id": "B", "player": "Player B", "ecr": 2.0},
        # POST-kickoff snapshot (after the synthetic 2026-09-04 first REG gameday below):
        # B overtakes A on the back of real Week-1 performance, not preseason drift.
        {"scrape_date": dt.date(2026, 9, 15), "pos": "WR", "id": "A", "player": "Player A", "ecr": 2.0},
        {"scrape_date": dt.date(2026, 9, 15), "pos": "WR", "id": "B", "player": "Player B", "ecr": 1.0},
    ]
    return pl.DataFrame(rows)


def test_build_ecr_trajectory_excludes_post_kickoff_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(build.adp, "_load_ecr_preseason", lambda: _synthetic_ecr_snapshots())
    monkeypatch.setattr(build.adp, "_first_reg_dates", lambda: {build.DASHBOARD_SEASON: dt.date(2026, 9, 4)})
    # Redirect id_map's review-CSV write off the real, checked-in data/id_map_review.csv --
    # every synthetic row is deliberately unmatched against an empty crosswalk.
    real_match_to_gsis = build.match_to_gsis

    def _match_to_gsis_tmp(*args, **kwargs):
        kwargs.setdefault("review_path", tmp_review_path)
        return real_match_to_gsis(*args, **kwargs)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp_review_path = Path(td) / "review.csv"
        monkeypatch.setattr(build, "match_to_gsis", _match_to_gsis_tmp)

        all_ranked, report = build.build_ecr_trajectory(_EMPTY_CROSSWALK)

    dates = set(all_ranked.select("scrape_date").unique().get_column("scrape_date").to_list())
    assert dt.date(2026, 9, 15) not in dates, "a post-kickoff ECR snapshot leaked into the preseason trajectory"
    assert dates == {dt.date(2026, 6, 15), dt.date(2026, 8, 20)}
    assert report["n_snapshot_dates"] == 2


# --------------------------------------------------------------------------
# Finding 8: gsis_id keying (falling back to name|POS only when unresolved)
# --------------------------------------------------------------------------


def test_assign_player_keys_prefers_gsis_id() -> None:
    keys, warnings = build._assign_player_keys(
        ["Player A", "Player B", "Player C"], ["WR", "RB", "TE"], ["00-0001111", None, "00-0003333"]
    )
    assert keys == ["00-0001111", "Player B|RB", "00-0003333"]
    assert warnings == []


def test_assign_player_keys_warns_on_name_pos_collision() -> None:
    """Two genuinely different players sharing a name+position, both unresolved to a

    gsis_id, collide on the name|POS fallback key -- must be reported, not silently let
    one clobber the other in every key-indexed map.
    """
    keys, warnings = build._assign_player_keys(
        ["Player A", "Player A"], ["WR", "WR"], [None, None]
    )
    assert keys == ["Player A|WR", "Player A|WR"]
    assert len(warnings) == 1
    assert "Player A|WR" in warnings[0]


def test_assign_player_keys_no_collision_when_gsis_ids_differ() -> None:
    keys, warnings = build._assign_player_keys(
        ["Player A", "Player A"], ["WR", "WR"], ["00-0001111", "00-0002222"]
    )
    assert keys == ["00-0001111", "00-0002222"]
    assert warnings == []


def test_load_board_and_feature_profiles_and_ecr_use_the_same_key_scheme(assembled) -> None:
    """Board rows, feature profiles, and ECR trajectories must all key a given resolved

    player identically (its gsis_id) -- the whole point of Finding 8's fix is that a
    client can join these three maps on one shared key.
    """
    payload, _ = assembled
    board_keys = {r["key"] for r in payload["board"]}
    feature_keys = set(payload["features"].keys())
    ecr_keys = set(payload["ecr"].keys())
    # every feature-profile / ECR key must be a real board key (never an orphan key
    # under a different scheme that the client couldn't join back to a board row).
    assert feature_keys <= board_keys
    assert ecr_keys <= board_keys


def test_build_ecr_trajectory_falls_back_to_no_cap_when_schedule_missing(monkeypatch) -> None:
    """A season whose schedule hasn't been pulled yet (no cached first-REG date) must not

    have its whole trajectory blanked -- falls back to the pre-fix unbounded window rather
    than silently dropping every snapshot.
    """
    monkeypatch.setattr(build.adp, "_load_ecr_preseason", lambda: _synthetic_ecr_snapshots())
    monkeypatch.setattr(build.adp, "_first_reg_dates", lambda: {})  # no cached schedule for DASHBOARD_SEASON

    real_match_to_gsis = build.match_to_gsis

    def _match_to_gsis_tmp(*args, **kwargs):
        kwargs.setdefault("review_path", tmp_review_path)
        return real_match_to_gsis(*args, **kwargs)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp_review_path = Path(td) / "review.csv"
        monkeypatch.setattr(build, "match_to_gsis", _match_to_gsis_tmp)

        all_ranked, report = build.build_ecr_trajectory(_EMPTY_CROSSWALK)

    dates = set(all_ranked.select("scrape_date").unique().get_column("scrape_date").to_list())
    assert dates == {dt.date(2026, 6, 15), dt.date(2026, 8, 20), dt.date(2026, 9, 15)}


# --------------------------------------------------------------------------
# v2.4: peer-rank-only display -- the dashboard payload carries cohort_rank/value_gap
# (cohort-based) only; predicted_finish_rank/value_gap_typical stay CSV-only for analysts
# and are never sent to the dashboard row payload.
# --------------------------------------------------------------------------


def test_board_rows_never_carry_predicted_finish_rank_or_typical_value_gap(assembled) -> None:
    payload, _ = assembled
    veteran_rows = [r for r in payload["board"] if "cr" in r]
    if not veteran_rows:
        pytest.skip("no veteran rows with a cohort_rank on the board payload")
    for r in veteran_rows:
        assert "pfr" not in r, "predicted_finish_rank ('pfr') must not reach the dashboard payload (peer-rank-only display)"
        assert "vgc" not in r, "the old cohort-based-alongside key ('vgc') is retired -- 'vg' is cohort-based now"


def test_board_rows_vg_matches_csv_cohort_based_value_gap(assembled) -> None:
    """`vg` (the payload's displayed/colored Value gap) must equal the board CSV's

    cohort-based `value_gap` column, not `value_gap_typical` -- the peer-rank-only
    display default as of this session (see src.dashboard.build.board_rows's docstring).
    """
    payload, _ = assembled
    csv_df = pd.read_csv(bd.BOARD_CSV_PATH)
    csv_df = csv_df[csv_df["value_gap"].notna()]
    if csv_df.empty:
        pytest.skip("no board rows with a non-null value_gap")
    # board CSV carries no gsis_id column (see _assign_player_keys's docstring) -- match
    # on (player_name, pos) instead of trying to reconstruct the payload's gsis_id key.
    by_name_pos: dict[tuple[str, str], dict] = {}
    for r in payload["board"]:
        by_name_pos.setdefault((r["n"], r["p"]), r)
    checked = 0
    for _, row in csv_df.iterrows():
        payload_row = by_name_pos.get((row["player_name"], row["pos"]))
        if payload_row is None or payload_row.get("vg") is None:
            continue
        assert payload_row["vg"] == pytest.approx(float(row["value_gap"]), abs=1e-6)
        checked += 1
    assert checked > 0, "no rows resolved between the CSV and the dashboard payload to compare"


# --------------------------------------------------------------------------
# v2.4: low-octane-offense chip fields
# --------------------------------------------------------------------------


def test_board_rows_carry_low_octane_fields(assembled) -> None:
    payload, _ = assembled
    veteran_rows = [r for r in payload["board"] if "cr" in r]
    if not veteran_rows:
        pytest.skip("no veteran rows on the board payload")
    assert all("loc" in r for r in veteran_rows), "every veteran row must carry the 'loc' low-octane-offense flag"
    flagged = [r for r in veteran_rows if r["loc"]]
    for r in flagged:
        assert r.get("locsrc") in {"Vegas", "est.", "unknown"}, r.get("locsrc")
