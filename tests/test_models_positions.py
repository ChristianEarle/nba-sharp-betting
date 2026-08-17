"""Tests for Phase 4 RB/TE/QB modeling (src/models/train.py via its per-position

thin wrappers). Parameterized over all four positions (wr included) for the
tiny-config determinism + fold-boundary checks, per the brief ("reuse tiny-
config determinism + fold-boundary per position (parameterize, don't
copy-paste three times)") — tests/test_models.py keeps its own WR-only
suite untouched; this file is the shared, parameterized version plus
RB/TE/QB-only baseline-sanity checks (identical mechanism to
tests/test_models.py's, just looped).

Data-dependent tests read the gitignored data/processed/{features_<pos>,
labels}.parquet and skip when absent. Every pipeline run in this file uses
each position's config's ``optuna.tiny_trials`` (currently 5) instead of
the real 60/30-trial config. The full run is __main__-only:
``python -m src.models.train <rb|te|qb>``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import baselines as bl
from src.models import cv
from src.models import metrics as mt
from src.models import train
from src.models.train_qb import load_config as qb_load_config, run_full_pipeline as qb_run_full_pipeline
from src.models.train_rb import load_config as rb_load_config, run_full_pipeline as rb_run_full_pipeline
from src.models.train_te import load_config as te_load_config, run_full_pipeline as te_run_full_pipeline

_POSITIONS = ("wr", "rb", "te", "qb")

_LOAD_CONFIG = {
    "wr": lambda: train.load_config(train.position_spec("wr").config_path),
    "rb": rb_load_config,
    "te": te_load_config,
    "qb": qb_load_config,
}
_RUN_PIPELINE = {
    "wr": lambda **kw: train.run_full_pipeline(train.position_spec("wr"), **kw),
    "rb": rb_run_full_pipeline,
    "te": te_run_full_pipeline,
    "qb": qb_run_full_pipeline,
}


def _spec(pos: str) -> train.PositionSpec:
    return train.position_spec(pos)


def _data_available(pos: str) -> bool:
    spec = _spec(pos)
    return spec.features_path.exists() and spec.labels_path.exists()


def _skip_if_data_missing(pos: str) -> None:
    if not _data_available(pos):
        pytest.skip(f"features_{pos}.parquet / labels.parquet not built; run `python -m src.features.{pos}`")


# --------------------------------------------------------------------------
# CV fold boundaries (no data dependency -- pure src.models.cv, position-
# agnostic already; parameterizing here just documents that every position
# shares the identical fold construction, not four copies of it).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", _POSITIONS)
def test_validation_folds_respect_boundary(pos: str) -> None:
    for fold in cv.validation_folds():
        assert max(fold.train_seasons) < fold.val_season


@pytest.mark.parametrize("pos", _POSITIONS)
def test_holdout_split_boundary_and_years(pos: str) -> None:
    split = cv.holdout_split()
    assert max(split.train_seasons) < min(split.holdout_seasons)
    assert set(split.holdout_seasons) == {2024, 2025}


# --------------------------------------------------------------------------
# Baseline sanity (data-dependent), RB/TE/QB — same checks as
# tests/test_models.py runs for WR.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", ("rb", "te", "qb"))
def test_baselines_return_series_aligned_to_input(pos: str) -> None:
    _skip_if_data_missing(pos)
    spec = _spec(pos)
    cfg = _LOAD_CONFIG[pos]()
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=cfg)
    scores = bl.compute_all_baselines(df, cfg)
    assert set(scores) == set(bl.BASELINE_NAMES)
    for name, s in scores.items():
        assert len(s) == len(df), f"{pos}/{name}"
        assert not s.isna().any(), f"{pos}: {name} produced NaN scores"


@pytest.mark.parametrize("pos", ("rb", "te", "qb"))
def test_adp_baseline_top10_is_earliest_expectation_ranks(pos: str) -> None:
    _skip_if_data_missing(pos)
    spec = _spec(pos)
    cfg = _LOAD_CONFIG[pos]()
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=cfg)
    season = int(df["season"].max())
    sub = df[df["season"] == season].reset_index(drop=True)
    if len(sub) < 10:
        pytest.skip(f"{pos}: fewer than 10 rows in {season}, top-10 comparison degenerate")

    score = bl.adp_knows_best(sub)
    idx = mt.top_k_indices(score.to_numpy(), 10)
    top_ranks = sorted(sub.iloc[idx]["expectation_pos_rank"].tolist())
    ten_smallest = sorted(sub["expectation_pos_rank"].nsmallest(10).tolist())
    assert top_ranks == ten_smallest


# --------------------------------------------------------------------------
# Tiny-config full pipeline: determinism + fold-boundary + output sanity,
# parameterized across every position (module-scoped fixture per position
# so the tiny pipeline runs once per position, shared across the cheap
# per-position assertions).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module", params=_POSITIONS)
def tiny_result(request):
    pos = request.param
    _skip_if_data_missing(pos)
    cfg = _LOAD_CONFIG[pos]()
    trials = cfg["optuna"]["tiny_trials"]
    result = _RUN_PIPELINE[pos](cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42)
    return pos, result


def test_calibration_output_in_unit_interval_no_nan(tiny_result) -> None:
    pos, result = tiny_result
    holdout_pred = result["holdout_preds"]["pred_calibrated"].to_numpy()
    assert not np.isnan(holdout_pred).any(), pos
    assert (holdout_pred >= 0).all() and (holdout_pred <= 1).all(), pos

    val_pred = result["pooled_val"]["pred_calibrated"].to_numpy()
    assert not np.isnan(val_pred).any(), pos
    assert (val_pred >= 0).all() and (val_pred <= 1).all(), pos


def test_holdout_predictions_cover_only_holdout_seasons(tiny_result) -> None:
    pos, result = tiny_result
    seasons = set(result["holdout_preds"]["season"].unique().tolist())
    assert seasons == {2024, 2025}, pos


def test_calibration_method_is_isotonic_or_platt(tiny_result) -> None:
    pos, result = tiny_result
    assert result["calib_method"] in ("isotonic", "platt"), pos


def test_fold_boundary_respected_in_tiny_run(tiny_result) -> None:
    """Every fold this run actually used respects max(train) < val -- structural, but

    re-checked here against the folds a real run object carries.
    """
    pos, result = tiny_result
    for fold in result["folds"]:
        assert max(fold.train_seasons) < fold.val_season, pos


@pytest.mark.parametrize("pos", _POSITIONS)
def test_determinism_two_runs_identical_holdout_predictions(pos: str) -> None:
    _skip_if_data_missing(pos)
    cfg = _LOAD_CONFIG[pos]()
    trials = cfg["optuna"]["tiny_trials"]
    r1 = _RUN_PIPELINE[pos](cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42)
    r2 = _RUN_PIPELINE[pos](cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42)

    p1 = r1["holdout_preds"].sort_values(["season", "gsis_id"])["pred_calibrated"].to_numpy()
    p2 = r2["holdout_preds"].sort_values(["season", "gsis_id"])["pred_calibrated"].to_numpy()
    np.testing.assert_array_equal(p1, p2, err_msg=pos)

    assert r1["blend_weights"] == r2["blend_weights"], pos
    assert r1["calib_method"] == r2["calib_method"], pos
    assert r1["frozen"]["lgbm_params"] == r2["frozen"]["lgbm_params"], pos
    assert r1["frozen"]["xgb_params"] == r2["frozen"]["xgb_params"], pos
