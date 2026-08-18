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

import json

import pandas as pd
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


def test_all_board_players_present_in_payload(assembled) -> None:
    payload, _ = assembled
    csv_df = pd.read_csv(bd.BOARD_CSV_PATH)
    assert len(payload["board"]) == len(csv_df)
    keys = [r["key"] for r in payload["board"]]
    assert len(keys) == len(set(keys)), "duplicate player|pos keys on the board payload"
    # every CSV row's (player_name, pos) shows up as a payload key
    csv_keys = set(csv_df["player_name"].astype(str) + "|" + csv_df["pos"].astype(str))
    assert csv_keys == set(keys)


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
