"""Tests for Phase 4 WR modeling (src/models/cv.py, train_wr.py, baselines.py, metrics.py).

Data-dependent tests read the gitignored data/processed/{features_wr,labels}.parquet
and skip when absent, matching test_features.py / test_labels.py's pattern. Every
pipeline run in this file uses ``configs/model_wr.yaml``'s ``optuna.tiny_trials``
(currently 5) instead of the real 60/30-trial config, per the brief ("keep runtime
sane: use the tiny config for tests"). The full run is __main__-only:
``python -m src.models.train_wr``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import baselines as bl
from src.models import cv
from src.models import metrics as mt
from src.models.train_wr import FEATURES_WR_PATH, LABELS_PATH, load_config, load_modeling_frame, run_full_pipeline


def _data_available() -> bool:
    return FEATURES_WR_PATH.exists() and LABELS_PATH.exists()


def _skip_if_data_missing() -> None:
    if not _data_available():
        pytest.skip("features_wr.parquet / labels.parquet not built; run `python -m src.features.wr`")


# --------------------------------------------------------------------------
# CV fold boundaries (no data dependency -- pure src.models.cv)
# --------------------------------------------------------------------------


def test_validation_folds_respect_boundary() -> None:
    for fold in cv.validation_folds():
        assert max(fold.train_seasons) < fold.val_season


def test_validation_folds_match_brief() -> None:
    """(train <=2019 -> val 2020), (<=2020 -> 2021), (<=2021 -> 2022), (<=2022 -> 2023)."""
    got = [(fold.train_seasons[-1], fold.val_season) for fold in cv.validation_folds()]
    assert got == [(2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023)]


def test_holdout_years_never_in_any_validation_fold() -> None:
    holdout_years = set(cv.HOLDOUT_SEASONS)
    for fold in cv.validation_folds():
        assert fold.val_season not in holdout_years
        assert holdout_years.isdisjoint(fold.train_seasons)


def test_holdout_split_boundary_and_years() -> None:
    split = cv.holdout_split()
    assert max(split.train_seasons) < min(split.holdout_seasons)
    assert set(split.holdout_seasons) == {2024, 2025}


def test_fold_construction_rejects_boundary_violation() -> None:
    """A Fold that would violate max(train) < val cannot be constructed at all --

    the invariant holds structurally, not just by a downstream check.
    """
    with pytest.raises(AssertionError):
        cv.Fold(train_seasons=(2020, 2021), val_season=2020)
    with pytest.raises(AssertionError):
        cv.Fold(train_seasons=(2021,), val_season=2021)


def test_validation_folds_signature_has_no_holdout_parameter() -> None:
    """Structural guard: validation_folds() cannot be pointed at a holdout year --

    its only parameter controls the training start, not which seasons validate.
    """
    import inspect

    params = set(inspect.signature(cv.validation_folds).parameters)
    assert params == {"train_start_season"}


# --------------------------------------------------------------------------
# Baseline sanity (data-dependent: needs the joined modeling frame)
# --------------------------------------------------------------------------


def test_adp_baseline_top10_is_earliest_expectation_ranks() -> None:
    """adp_knows_best's top-10 by score must, by construction, be the 10 rows with the

    smallest expectation_pos_rank that season -- it is nothing but -expectation_pos_rank.
    """
    _skip_if_data_missing()
    cfg = load_config()
    df = load_modeling_frame(cfg=cfg)
    season = int(df["season"].max())
    sub = df[df["season"] == season].reset_index(drop=True)

    score = bl.adp_knows_best(sub)
    idx = mt.top_k_indices(score.to_numpy(), 10)
    top_ranks = sorted(sub.iloc[idx]["expectation_pos_rank"].tolist())
    ten_smallest = sorted(sub["expectation_pos_rank"].nsmallest(10).tolist())
    assert top_ranks == ten_smallest


def test_baselines_return_series_aligned_to_input() -> None:
    _skip_if_data_missing()
    cfg = load_config()
    df = load_modeling_frame(cfg=cfg)
    scores = bl.compute_all_baselines(df, cfg)
    assert set(scores) == set(bl.BASELINE_NAMES)
    for name, s in scores.items():
        assert len(s) == len(df), name
        assert not s.isna().any(), f"{name} produced NaN scores"


# --------------------------------------------------------------------------
# Full tiny-pipeline run (shared fixture for the cheap checks; determinism
# gets its own two fresh runs so it isn't testing a cached object).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_result(tmp_path_factory):
    _skip_if_data_missing()
    cfg = load_config()
    trials = cfg["optuna"]["tiny_trials"]
    # output_root is mandatory hygiene here: without it this tiny run would
    # overwrite the PRODUCTION bundle/report/metrics under data/models/ and
    # outputs/ with a 5-trial version — which happened, silently, before the
    # parameter existed. See run_full_pipeline's docstring.
    out = tmp_path_factory.mktemp("tiny_wr_artifacts")
    return run_full_pipeline(
        cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42, output_root=out
    )


def test_calibration_output_in_unit_interval_no_nan(tiny_result) -> None:
    holdout_pred = tiny_result["holdout_preds"]["pred_calibrated"].to_numpy()
    assert not np.isnan(holdout_pred).any()
    assert (holdout_pred >= 0).all() and (holdout_pred <= 1).all()

    val_pred = tiny_result["pooled_val"]["pred_calibrated"].to_numpy()
    assert not np.isnan(val_pred).any()
    assert (val_pred >= 0).all() and (val_pred <= 1).all()


def test_holdout_predictions_cover_only_holdout_seasons(tiny_result) -> None:
    seasons = set(tiny_result["holdout_preds"]["season"].unique().tolist())
    assert seasons == {2024, 2025}


def test_calibration_method_is_isotonic_or_platt(tiny_result) -> None:
    assert tiny_result["calib_method"] in ("isotonic", "platt")


def test_determinism_two_runs_identical_holdout_predictions(tmp_path) -> None:
    _skip_if_data_missing()
    cfg = load_config()
    trials = cfg["optuna"]["tiny_trials"]
    r1 = run_full_pipeline(
        cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42, output_root=tmp_path / "a"
    )
    r2 = run_full_pipeline(
        cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42, output_root=tmp_path / "b"
    )

    p1 = r1["holdout_preds"].sort_values(["season", "gsis_id"])["pred_calibrated"].to_numpy()
    p2 = r2["holdout_preds"].sort_values(["season", "gsis_id"])["pred_calibrated"].to_numpy()
    np.testing.assert_array_equal(p1, p2)

    assert r1["blend_weights"] == r2["blend_weights"]
    assert r1["calib_method"] == r2["calib_method"]
    assert r1["frozen"]["lgbm_params"] == r2["frozen"]["lgbm_params"]
    assert r1["frozen"]["xgb_params"] == r2["frozen"]["xgb_params"]


def test_tiny_run_never_touches_production_artifacts(tmp_path) -> None:
    """The clobbering guard.

    Before ``output_root`` existed, this exact tiny run overwrote the shipped
    bundle under data/models/ (and its report/metrics) with a 5-trial version
    every time the suite ran -- the board silently degraded and metrics files
    described a model that no longer existed on disk. Lock the fix: a redirected
    run must leave every production artifact byte-identical.
    """
    _skip_if_data_missing()
    from src.models import train as _train

    spec = _train.position_spec("wr")
    prod_paths = [spec.artifact_path, spec.report_path, spec.metrics_json_path]
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in prod_paths if p.exists()}
    if not before:
        pytest.skip("no production artifacts present to protect")

    cfg = load_config()
    trials = cfg["optuna"]["tiny_trials"]
    run_full_pipeline(
        cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42, output_root=tmp_path
    )

    after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in prod_paths if p.exists()}
    assert before == after, "tiny run modified production artifacts"
    assert (tmp_path / spec.artifact_path.name).exists(), "redirected bundle missing"
