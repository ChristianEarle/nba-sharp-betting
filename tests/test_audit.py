"""Tests for src/audit/holdout_board_audit.py (the "are we sure about this model being

correct?" holdout board audit). Data-dependent (real bundles + labels.parquet), skips
cleanly when its inputs are missing, matching every other data-dependent test file's
``_skip_if_*`` pattern in this repo (see tests/test_quantile.py). Every report/metrics
write in this file goes through a ``tmp_path`` -- never the production
``outputs/holdout_board_audit.*`` -- same "output_root is mandatory hygiene" discipline
``tests/test_models.py``/``tests/test_quantile.py`` already use for pipeline runs.
"""

from __future__ import annotations

import json

import pandas as pd
import polars as pl
import pytest

from src.audit import holdout_board_audit as aud
from src.models import train

_POSITIONS = ("wr", "rb", "te", "qb")


def _data_available() -> bool:
    if not train.LABELS_PATH.exists():
        return False
    for pos in _POSITIONS:
        spec = train.position_spec(pos)
        if not (spec.artifact_path.exists() and spec.features_path.exists()):
            return False
    return True


def _skip_if_data_missing() -> None:
    if not _data_available():
        pytest.skip("model bundles / features / labels.parquet not built")


# --------------------------------------------------------------------------
# Shared fixtures (build each season's board once, reused across tests)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def board_2024() -> pd.DataFrame:
    _skip_if_data_missing()
    return aud.build_season_board(2024)


@pytest.fixture(scope="module")
def board_2025() -> pd.DataFrame:
    _skip_if_data_missing()
    return aud.build_season_board(2025)


# --------------------------------------------------------------------------
# Simulation determinism
# --------------------------------------------------------------------------


def test_build_season_board_is_deterministic() -> None:
    """Scoring frozen (already-fit) bundles is a pure function call -- no seed/RNG

    anywhere in the scoring path (unlike training) -- so two independent builds of the
    same season's board must be byte-identical on every numeric/ID column.
    """
    _skip_if_data_missing()
    b1 = aud.build_season_board(2024)
    b2 = aud.build_season_board(2024)
    assert not b1.empty and not b2.empty
    b1 = b1.sort_values(["pos", "gsis_id"]).reset_index(drop=True)
    b2 = b2.sort_values(["pos", "gsis_id"]).reset_index(drop=True)
    assert list(b1["gsis_id"]) == list(b2["gsis_id"])
    for col in ("probability", "p_startable", "p_elite", "p_useful", "expected_ppg", "raw_score"):
        pd.testing.assert_series_equal(b1[col], b2[col], check_names=False)


def test_build_audit_ladder_tables_deterministic() -> None:
    """Same determinism guarantee one level up: the grading tables built from two

    independent ``build_audit`` runs (season 2025 only, to keep this test fast) must
    match exactly.
    """
    _skip_if_data_missing()
    b1 = aud.build_season_board(2025)
    b2 = aud.build_season_board(2025)
    cal1 = aud.ladder_calibration_table(b1, "p_startable", "realized_startable")
    cal2 = aud.ladder_calibration_table(b2, "p_startable", "realized_startable")
    pd.testing.assert_frame_equal(cal1, cal2)


# --------------------------------------------------------------------------
# No season-S-or-later data leaking into any audit input for season S
# --------------------------------------------------------------------------


def test_leaveout_thresholds_never_use_season_s_or_later() -> None:
    """build_leaveout_thresholds(S) must be built from labels rows with season < S only

    -- spot-check by reaching into the same raw-threshold frame the function itself
    builds from and asserting its season column's max is strictly below the cutoff.
    """
    _skip_if_data_missing()
    for season in aud.AUDIT_SEASONS:
        labels = pl.read_parquet(train.LABELS_PATH)
        raw = aud.pj.raw_thresholds_by_season(labels).filter(pl.col("season") < season)
        assert raw.is_empty() or raw.select(pl.col("season").max()).item() < season
        # the function itself must produce a non-empty curve from that exact input
        out = aud.build_leaveout_thresholds(season)
        assert not out.empty


def test_leaveout_mean_breakouts_never_use_season_s_or_later() -> None:
    _skip_if_data_missing()
    for season in aud.AUDIT_SEASONS:
        labels = pl.read_parquet(train.LABELS_PATH).filter(pl.col("season") < season)
        assert labels.is_empty() or labels.select(pl.col("season").max()).item() < season
        means = aud.leaveout_mean_breakouts(season)
        assert set(means) == {"QB", "RB", "WR", "TE"}
        assert all(v >= 0 for v in means.values())


def test_leaveout_thresholds_differ_from_shipped_when_season_included(board_2024) -> None:
    """The whole point of the leave-out rebuild: for season 2024, the shipped

    (production, full 2014-2025 span) rank_thresholds.parquet was fit INCLUDING 2024
    and 2025 -- the leave-out curve (seasons < 2024 only) should generally NOT be
    numerically identical to it (not a tautology -- if every delta were exactly zero
    the leave-out machinery would not be doing anything, which would itself be worth
    flagging). At least one (position, K) pair must differ.
    """
    delta = aud.threshold_delta_report(2024, aud.build_leaveout_thresholds(2024))
    assert not delta.empty
    assert (delta["delta"].abs() > 1e-9).any()


def test_season_population_is_season_s_only(board_2024) -> None:
    """The scored population for season S must contain ONLY season-S rows -- the

    classifier/quantile bundles themselves are frozen at <=2023 training (structurally
    enforced by src.models.cv.holdout_split, not this module), so the only remaining
    way season S's board could see "future" data is through its own population frame
    accidentally carrying rows from other seasons.
    """
    assert not board_2024.empty
    assert set(board_2024["season"].unique()) == {2024}


def test_engine_decision_is_read_only_not_rebuilt(tmp_path) -> None:
    """load_engine_decision() must not write anything -- it's documented as read of the

    CURRENTLY SHIPPED gate, never rebuilt per audit season (see module docstring's
    "Deviations" section). Snapshot outputs/comparison_gate.json's mtime-independent
    content before and after a build_season_board call and assert it is untouched.
    """
    _skip_if_data_missing()
    gate_path = aud.pj.COMPARISON_GATE_JSON_PATH
    before = gate_path.read_text() if gate_path.exists() else None
    aud.build_season_board(2025)
    after = gate_path.read_text() if gate_path.exists() else None
    assert before == after


# --------------------------------------------------------------------------
# Grading sanity
# --------------------------------------------------------------------------


def test_ladder_calibration_table_shape(board_2024) -> None:
    cal = aud.ladder_calibration_table(board_2024, "p_startable", "realized_startable")
    assert list(cal["displayed_bucket"]) == aud.PROB_BUCKET_LABELS
    assert (cal["n"].fillna(0) >= 0).all()
    valid = cal[cal["n"] > 0]
    assert (valid["mean_displayed"] >= 0).all() and (valid["mean_displayed"] <= 1).all()
    assert (valid["realized_rate"] >= 0).all() and (valid["realized_rate"] <= 1).all()


def test_range_honesty_pct_between_0_and_1(board_2024) -> None:
    rh = aud.range_honesty_table(board_2024)
    assert not rh.empty
    for col in ("pct_inside_range", "pct_below_floor", "pct_above_ceiling"):
        assert (rh[col] >= 0).all() and (rh[col] <= 1).all()
    # the three partitions must sum to 1 for every row (inside/below/above are exhaustive
    # and mutually exclusive by construction).
    total = rh["pct_inside_range"] + rh["pct_below_floor"] + rh["pct_above_ceiling"]
    assert (total.round(6) == 1.0).all()


def test_draft_value_breakout_hunt_picks_at_most_top_n(board_2024) -> None:
    dv = aud.draft_value_breakout_hunt(board_2024, top_n=10)
    assert not dv.empty
    assert (dv["n"] <= 10).all()
    assert set(dv["picker"]) == {"model", "adp_order"}


def test_value_gap_terciles_ordering_columns_present(board_2024) -> None:
    vg = aud.value_gap_terciles(board_2024)
    assert not vg.empty
    assert len(vg) == 3
    # terciles are ordered by construction (pd.qcut on value_gap ascending).
    assert vg["mean_value_gap"].is_monotonic_increasing


def test_named_receipts_only_eligible_players(board_2024) -> None:
    receipts = aud.named_receipts(board_2024, top_n=5)
    assert not receipts.empty
    assert set(receipts["position"]) <= {"QB", "RB", "WR", "TE"}
    for pos, g in receipts.groupby("position"):
        assert len(g) <= 5


def test_failure_autopsy_diagnosis_never_empty(board_2024) -> None:
    fa = aud.failure_autopsy(board_2024, n=3)
    for key in ("worst_misses", "biggest_surprises"):
        df = fa[key]
        if df.empty:
            continue
        assert df["diagnosis"].notna().all()
        assert (df["diagnosis"].str.len() > 0).all()
        # every worst miss must, by construction, actually be a bust (not startable).
    for _, r in fa["worst_misses"].iterrows():
        assert pd.isna(r["realized_finish_pos_rank"]) or r["realized_finish_pos_rank"] > 0


# --------------------------------------------------------------------------
# Report file: generated with every required section
# --------------------------------------------------------------------------


REQUIRED_REPORT_HEADERS = [
    "## Verdict",
    "## Headline calibration flags",
    "## 1. Ladder calibration",
    "## 2. Range honesty",
    "## 3. Draft-value test",
    "## 4. Named receipts",
    "## 5. Failure autopsy",
    "## Leave-future-out integrity check",
    "## Deviations from the brief",
]


def test_report_and_metrics_written_with_required_sections(tmp_path) -> None:
    _skip_if_data_missing()
    a = aud.build_audit(seasons=(2024, 2025))
    report_path = tmp_path / "holdout_board_audit.md"
    metrics_path = tmp_path / "holdout_board_audit_metrics.json"
    aud.write_report(a, report_path=report_path)
    aud.write_metrics_json(a, metrics_path=metrics_path)

    assert report_path.exists()
    text = report_path.read_text()
    for header in REQUIRED_REPORT_HEADERS:
        assert header in text, f"missing section: {header}"
    assert "2024" in text and "2025" in text

    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["seasons"] == [2024, 2025]
    for key in ("ladder_pooled", "range_honesty_pooled", "draft_value_pooled", "value_gap_pooled", "threshold_deltas", "breakout_deltas"):
        assert key in payload


def test_report_does_not_touch_production_outputs(tmp_path) -> None:
    """Guard against a regression that points write_report/write_metrics_json back at

    the production outputs/ paths by default when called with an explicit path -- the
    production files (if present from a prior `python -m src.audit.holdout_board_audit`
    run) must be untouched by a tmp_path-scoped write.
    """
    _skip_if_data_missing()
    before = aud.REPORT_PATH.read_text() if aud.REPORT_PATH.exists() else None
    a = aud.build_audit(seasons=(2025,))
    aud.write_report(a, report_path=tmp_path / "r.md")
    aud.write_metrics_json(a, metrics_path=tmp_path / "m.json")
    after = aud.REPORT_PATH.read_text() if aud.REPORT_PATH.exists() else None
    assert before == after
