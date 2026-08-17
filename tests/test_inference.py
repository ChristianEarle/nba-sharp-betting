"""Tests for Phase 6 (2026 inference) and Phase 6/7's board deliverable.

Every test here is data-dependent (a saved model bundle, a built 2026
feature matrix, or the board CSV) and skips cleanly when its input is
absent, matching every other data-dependent test file's pattern in this
repo (``_skip_if_*`` helpers). Run ``python -m src.inference.board_2026``
(after the usual ``src.ingest.nflverse`` / ``src.labels.build`` /
``src.features.*`` / ``src.models.train_*`` chain) to produce everything
these tests read.
"""

from __future__ import annotations

import joblib
import pandas as pd
import polars as pl
import pytest

from src.inference import board_2026
from src.models import train

# --------------------------------------------------------------------------
# Golden-player regression test
# --------------------------------------------------------------------------

# Verified directly against the currently-saved bundles before encoding
# (see the session's final report): each of these is a real breakout==1
# holdout row (2024 or 2025) that scores strictly above its position+season
# median calibrated probability when scored from the stored
# features_{pos}.parquet row through the saved bundle -- not cherry-picked
# for a huge score, just a clear, reproducible margin over the median.
GOLDEN_PLAYERS = [
    ("wr", 2025, "Chris Olave"),
    ("te", 2025, "Kyle Pitts"),
    ("rb", 2024, "Chuba Hubbard"),
    ("qb", 2024, "Baker Mayfield"),
]


def _score_holdout(pos: str) -> pd.DataFrame | None:
    spec = train.position_spec(pos)
    if not (spec.artifact_path.exists() and spec.features_path.exists() and spec.labels_path.exists()):
        return None
    bundle = joblib.load(spec.artifact_path)
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
    holdout = df[df["season"].isin([2024, 2025])].copy()
    if holdout.empty:
        return None

    tree_cols = bundle["tree_feature_cols"]
    weights = bundle["blend_weights"]
    preds = {}
    if weights.get("lgbm", 0) > 0:
        preds["lgbm"] = bundle["lgbm"].predict_proba(holdout[tree_cols])[:, 1]
    if weights.get("xgb", 0) > 0:
        preds["xgb"] = bundle["xgb"].predict_proba(holdout[tree_cols])[:, 1]
    if weights.get("logistic", 0) > 0:
        model, imputer, scaler = bundle["logistic"]
        X = scaler.transform(imputer.transform(holdout[bundle["logistic_feature_cols"]]))
        preds["logistic"] = model.predict_proba(X)[:, 1]
    nonzero_weights = {k: w for k, w in weights.items() if k in preds}
    blended = train.apply_blend(nonzero_weights, **preds)
    holdout["prob"] = train.apply_calibration(bundle["calibration_method"], bundle["calibrator"], blended)
    return holdout


@pytest.mark.parametrize("pos,season,player_name", GOLDEN_PLAYERS)
def test_golden_breakout_scores_above_position_season_median(pos: str, season: int, player_name: str) -> None:
    holdout = _score_holdout(pos)
    if holdout is None:
        pytest.skip(f"{pos}: bundle or feature/label data not built; run `python -m src.models.train {pos}`")

    row = holdout[(holdout["season"] == season) & (holdout["player_name"] == player_name)]
    if row.empty:
        pytest.skip(f"{player_name} {season} not in {pos} holdout rows (bundle/feature data has drifted)")

    median = holdout.loc[holdout["season"] == season, "prob"].median()
    assert row["prob"].iloc[0] > median, (
        f"{player_name} {season} ({pos}): prob={row['prob'].iloc[0]:.4f} did not beat "
        f"the {season} position median ({median:.4f})"
    )


# --------------------------------------------------------------------------
# Board schema test
# --------------------------------------------------------------------------


def _board_df() -> pd.DataFrame:
    if not board_2026.BOARD_CSV_PATH.exists():
        pytest.skip(f"{board_2026.BOARD_CSV_PATH.name} not built; run `python -m src.inference.board_2026`")
    return pd.read_csv(board_2026.BOARD_CSV_PATH)


REQUIRED_BOARD_COLUMNS = [
    "player_name",
    "pos",
    "team",
    "probability",
    "raw_score",
    "probability_saturated",
    "expected_rank_delta",
    "consensus_ecr_pos_rank",
    "adp",
    "sleeper_adp_pos_rank",
    "adp_gap",
    "implied_pts",
    "edge",
    "availability",
    "shap_top3",
    "rationale",
    "section",
    "probability_heuristic",
]


def test_board_required_columns_present() -> None:
    df = _board_df()
    missing = [c for c in REQUIRED_BOARD_COLUMNS if c not in df.columns]
    assert not missing, f"breakout_board_2026.csv missing columns: {missing}"


def test_board_probabilities_in_unit_interval() -> None:
    """`probability_heuristic` (rookies) is a plain, uncapped [0, 1] probability.

    `probability` (veterans) is clamped to [board_2026.PROB_DISPLAY_LO,
    board_2026.PROB_DISPLAY_HI] for display -- see board_2026's module
    docstring ("Displayed-probability clamp") for why: isotonic
    calibration saturates to an exact 0/1 in terminal OOF buckets at this
    sample size, which reads as false certainty and collapses ordering.
    """
    df = _board_df()

    heuristic_vals = df["probability_heuristic"].dropna()
    assert (heuristic_vals >= 0).all() and (heuristic_vals <= 1).all(), "probability_heuristic: values outside [0, 1]"

    prob_vals = df["probability"].dropna()
    assert not prob_vals.empty
    assert (prob_vals >= board_2026.PROB_DISPLAY_LO - 1e-9).all() and (prob_vals <= board_2026.PROB_DISPLAY_HI + 1e-9).all(), (
        f"probability: values outside the display clamp [{board_2026.PROB_DISPLAY_LO}, {board_2026.PROB_DISPLAY_HI}]"
    )


def test_board_saturated_rows_flagged_and_ordered_by_raw_score() -> None:
    """Every probability_saturated==1 row hit the display clamp boundary (0.01 or 0.95),

    and within a tied probability bucket, raw_score gives a strict-enough ordering to
    break the tie (verified on this build's actual saturated QB/TE clusters).
    """
    df = _board_df()
    veterans = df[df["section"] == "veteran"]
    saturated = veterans[veterans["probability_saturated"] == 1]
    if saturated.empty:
        pytest.skip("no saturated rows on this build's board (isotonic didn't hit a terminal bucket)")

    assert set(saturated["probability"].round(2).unique()) <= {board_2026.PROB_DISPLAY_LO, board_2026.PROB_DISPLAY_HI}

    for (pos, prob), group in veterans.groupby(["pos", "probability"]):
        if len(group) > 1:
            assert group["raw_score"].nunique() > 1, (
                f"{pos} probability={prob}: {len(group)} tied rows share an identical raw_score too -- "
                "ordering among them is arbitrary"
            )


def test_board_has_zero_saturated_rows_v1_5() -> None:
    """v1.5: src.inference.board_2026 scores off train.SmoothedIsotonic (never exactly 0/1

    on the pooled OOF it was fit on -- see tests/test_models_positions.py's
    test_smoothed_calibrator_output_strictly_inside_unit_interval), so the board's
    probability_saturated count should be exactly 0 -- a real assertion, not the general
    robustness check above (which intentionally still tolerates future saturation and
    just checks it's handled correctly if it ever reappears).
    """
    df = _board_df()
    veterans = df[df["section"] == "veteran"]
    if veterans.empty:
        pytest.skip("no veteran rows on the board")
    n_saturated = int(veterans["probability_saturated"].sum())
    assert n_saturated == 0, f"expected 0 saturated veteran rows with SmoothedIsotonic, got {n_saturated}"


def test_board_rookies_are_flagged_and_separate_from_veterans() -> None:
    df = _board_df()
    sections = set(df["section"].unique())
    assert "veteran" in sections
    rookie_sections = [s for s in sections if "rookie" in s]
    assert rookie_sections, "no rookie section found on the board"

    veterans = df[df["section"] == "veteran"]
    rookies = df[df["section"].isin(rookie_sections)]
    # Veterans are scored on `probability` (calibrated), never `probability_heuristic`;
    # rookies are the reverse -- the two scales must never be mixed on one column.
    assert veterans["probability"].notna().any()
    assert rookies["probability_heuristic"].notna().any()
    assert veterans["probability_heuristic"].isna().all(), "veterans must not carry a heuristic probability"
    assert rookies["probability"].isna().all(), "rookies must not carry a calibrated veteran probability"


def test_board_availability_column_exists() -> None:
    df = _board_df()
    assert "availability" in df.columns


# --------------------------------------------------------------------------
# 2026 feature-matrix sanity
# --------------------------------------------------------------------------

# Approximate expected row counts per position's 2026 veteran (non-rookie)
# population -- rosters.parquet season==2026, years_exp>=1. QB's real
# active-roster population is smaller than the other three (roughly one
# start/backup/third string per team, ~32-100), so it gets its own,
# narrower bound rather than sharing WR/RB/TE's range.
ROW_COUNT_BOUNDS = {
    "qb": (50, 150),
    "rb": (100, 250),
    "wr": (150, 350),
    "te": (80, 200),
}


def _feature_matrix(pos: str) -> pl.DataFrame:
    path = board_2026.veteran_features_path(pos)
    if not path.exists():
        pytest.skip(f"{path.name} not built; run `python -m src.inference.board_2026`")
    return pl.read_parquet(path)


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_2026_feature_matrix_row_count_sane(pos: str) -> None:
    df = _feature_matrix(pos)
    lo, hi = ROW_COUNT_BOUNDS[pos]
    assert lo <= df.height <= hi, f"{pos}: {df.height} 2026 veteran rows outside sanity bounds [{lo}, {hi}]"


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_2026_feature_matrix_no_duplicate_keys(pos: str) -> None:
    df = _feature_matrix(pos)
    dupes = df.group_by(["season", "gsis_id"]).len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"{pos}: duplicate (season, gsis_id) rows in the 2026 feature matrix: {dupes}"


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_2026_new_oc_is_all_null(pos: str) -> None:
    """configs/coaching_changes.csv has no 2026 rows yet (documented limitation) --

    new_oc must read null for every 2026 row, not a guessed 0.
    """
    df = _feature_matrix(pos)
    assert df.get_column("new_oc").null_count() == df.height, f"{pos}: new_oc should be all-null for 2026"


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_2026_new_hc_is_not_all_null(pos: str) -> None:
    """schedules.parquet's 2026 Week-1 home/away_coach columns are fully populated,

    so new_hc must be derivable for at least some 2026 rows.
    """
    df = _feature_matrix(pos)
    assert df.get_column("new_hc").null_count() < df.height, f"{pos}: new_hc is entirely null for 2026"
