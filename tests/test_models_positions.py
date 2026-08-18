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
def tiny_result(request, tmp_path_factory):
    pos = request.param
    _skip_if_data_missing(pos)
    cfg = _LOAD_CONFIG[pos]()
    trials = cfg["optuna"]["tiny_trials"]
    # output_root keeps this tiny run away from the PRODUCTION bundles under
    # data/models/ — omitting it silently overwrites the shipped model with a
    # 5-trial version (which happened; see run_full_pipeline's docstring).
    out = tmp_path_factory.mktemp(f"tiny_{pos}_artifacts")
    result = _RUN_PIPELINE[pos](
        cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42, output_root=out
    )
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


# --------------------------------------------------------------------------
# v1.5 -- Laplace-smoothed isotonic calibration (train.SmoothedIsotonic).
# --------------------------------------------------------------------------


def test_smoothed_calibrator_persisted_alongside_frozen_calibrator(tiny_result) -> None:
    pos, result = tiny_result
    frozen = result["frozen"]
    assert frozen["smoothed_calibration_method"] == "smoothed_isotonic", pos
    assert isinstance(frozen["smoothed_calibrator"], train.SmoothedIsotonic), pos
    # The old calibrator is never deleted/overwritten by adding the smoothed one.
    assert frozen["calibration_method"] in ("isotonic", "platt"), pos
    assert frozen["calibrator"] is not None, pos


def test_pooled_oof_val_is_persisted_and_matches_pooled_validation(tiny_result) -> None:
    pos, result = tiny_result
    oof = result["frozen"]["pooled_oof_val"]
    assert set(["season", "gsis_id", "pred_blend", "breakout"]).issubset(oof.columns), pos
    assert len(oof) == len(result["pooled_val"]), pos
    assert not oof["pred_blend"].isna().any(), pos


def test_smoothed_calibrator_output_strictly_inside_unit_interval(tiny_result) -> None:
    """The whole point of SmoothedIsotonic: never an exact 0.0 or 1.0 on the pooled OOF

    it was fit on -- unlike plain isotonic, which can saturate at terminal buckets (see
    the module docstring in src/models/train.py and README's v1 "Isotonic calibration
    saturates at the extremes" limitation).
    """
    pos, result = tiny_result
    frozen = result["frozen"]
    oof = frozen["pooled_oof_val"]
    smoothed = frozen["smoothed_calibrator"].predict(oof["pred_blend"].to_numpy())
    assert not np.isnan(smoothed).any(), pos
    assert (smoothed > 0.0).all(), f"{pos}: smoothed calibrator hit 0.0"
    assert (smoothed < 1.0).all(), f"{pos}: smoothed calibrator hit 1.0"


def test_smoothed_calibrator_is_monotonic_in_raw_score(tiny_result) -> None:
    pos, result = tiny_result
    frozen = result["frozen"]
    oof = frozen["pooled_oof_val"].sort_values("pred_blend")
    smoothed = frozen["smoothed_calibrator"].predict(oof["pred_blend"].to_numpy())
    diffs = np.diff(smoothed)
    assert (diffs >= -1e-12).all(), f"{pos}: smoothed calibrator output is not monotonic non-decreasing"


def test_smoothed_isotonic_matches_rule_of_succession_on_a_toy_case() -> None:
    """Direct unit test of SmoothedIsotonic, independent of any position's data: a raw

    score cleanly separable into two PAV blocks (all-negative low block, all-positive high
    block) should smooth to (0+1)/(n_lo+2) and (n_hi+1)/(n_hi+2) respectively, never 0 or 1.
    """
    raw = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    smoothed = train.SmoothedIsotonic().fit(raw, y)
    preds = smoothed.predict(raw)
    assert (preds > 0.0).all() and (preds < 1.0).all()
    # Monotonic and strictly separates the two blocks (low block < high block).
    assert preds[:3].max() < preds[3:].min()
    # Rule of succession on the low (all-negative, n=3) and high (all-positive, n=3) blocks.
    assert preds[0] == pytest.approx(1.0 / 5.0)
    assert preds[-1] == pytest.approx(4.0 / 5.0)


def test_fold_boundary_respected_in_tiny_run(tiny_result) -> None:
    """Every fold this run actually used respects max(train) < val -- structural, but

    re-checked here against the folds a real run object carries.
    """
    pos, result = tiny_result
    for fold in result["folds"]:
        assert max(fold.train_seasons) < fold.val_season, pos


@pytest.mark.parametrize("pos", _POSITIONS)
def test_determinism_two_runs_identical_holdout_predictions(pos: str, tmp_path) -> None:
    _skip_if_data_missing(pos)
    cfg = _LOAD_CONFIG[pos]()
    trials = cfg["optuna"]["tiny_trials"]
    r1 = _RUN_PIPELINE[pos](
        cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42, output_root=tmp_path / "a"
    )
    r2 = _RUN_PIPELINE[pos](
        cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42, output_root=tmp_path / "b"
    )

    p1 = r1["holdout_preds"].sort_values(["season", "gsis_id"])["pred_calibrated"].to_numpy()
    p2 = r2["holdout_preds"].sort_values(["season", "gsis_id"])["pred_calibrated"].to_numpy()
    np.testing.assert_array_equal(p1, p2, err_msg=pos)

    assert r1["blend_weights"] == r2["blend_weights"], pos
    assert r1["calib_method"] == r2["calib_method"], pos
    assert r1["frozen"]["lgbm_params"] == r2["frozen"]["lgbm_params"], pos
    assert r1["frozen"]["xgb_params"] == r2["frozen"]["xgb_params"], pos


# --------------------------------------------------------------------------
# v1.7 -- seed-ensemble determinism (tiny config, 2 seeds, output_root). A
# fixed ``optuna.seeds`` list must reproduce byte-identical results across
# independent process invocations, exactly like the single-seed case above,
# but additionally checking the per-seed structure (``frozen["per_seed"]``,
# ``ensemble_seeds``) survives identically too -- not just the final
# ensemble score, which could in principle coincide even if the per-seed
# pieces feeding it differed.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", _POSITIONS)
def test_determinism_seed_ensemble_two_runs_identical(pos: str, tmp_path) -> None:
    import copy

    _skip_if_data_missing(pos)
    cfg = copy.deepcopy(_LOAD_CONFIG[pos]())
    cfg["optuna"]["seeds"] = [42, 1337]  # 2 seeds, distinct from the real 3-seed config -- keeps this fast
    trials = cfg["optuna"]["tiny_trials"]

    r1 = _RUN_PIPELINE[pos](cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42, output_root=tmp_path / "a")
    r2 = _RUN_PIPELINE[pos](cfg=cfg, classifier_trials=trials, regression_trials=trials, seed=42, output_root=tmp_path / "b")

    assert r1["ensemble_seeds"] == [42, 1337] == r2["ensemble_seeds"], pos

    p1 = r1["holdout_preds"].sort_values(["season", "gsis_id"])["pred_calibrated"].to_numpy()
    p2 = r2["holdout_preds"].sort_values(["season", "gsis_id"])["pred_calibrated"].to_numpy()
    np.testing.assert_array_equal(p1, p2, err_msg=f"{pos}: ensemble pred_calibrated diverged")

    raw1 = r1["holdout_preds"].sort_values(["season", "gsis_id"])["pred_blend_raw"].to_numpy()
    raw2 = r2["holdout_preds"].sort_values(["season", "gsis_id"])["pred_blend_raw"].to_numpy()
    np.testing.assert_array_equal(raw1, raw2, err_msg=f"{pos}: ensemble raw score diverged")

    for seed in (42, 1337):
        info1, info2 = r1["per_seed"][seed], r2["per_seed"][seed]
        assert info1["lgbm_params"] == info2["lgbm_params"], f"{pos}/seed={seed}"
        assert info1["xgb_params"] == info2["xgb_params"], f"{pos}/seed={seed}"
        assert info1["blend_weights"] == info2["blend_weights"], f"{pos}/seed={seed}"
        assert info1["lgbm_final_n_estimators"] == info2["lgbm_final_n_estimators"], f"{pos}/seed={seed}"
        assert info1["xgb_final_n_estimators"] == info2["xgb_final_n_estimators"], f"{pos}/seed={seed}"


def test_seed_ensemble_raw_score_is_mean_of_per_seed_blends(tiny_result) -> None:
    """The ensemble's holdout raw score must equal the unweighted mean of every seed's own

    blended holdout score -- the exact "raw score = mean of per-seed blended scores"
    contract the v1.7 brief specifies (src.models.train.holdout_predictions).
    """
    pos, result = tiny_result
    frozen = result["frozen"]
    seeds = frozen["ensemble_seeds"]
    if len(seeds) < 2:
        pytest.skip(f"{pos}: this position's config carries fewer than 2 optuna.seeds")

    holdout_preds = result["holdout_preds"]
    tree_cols = frozen["tree_feature_cols"]
    df = result["df"]
    from src.models.cv import holdout_split as _holdout_split
    from src.models.train import TRAIN_START_SEASON, _fit_holdout_lgbm, _fit_holdout_xgb, _logistic_predict, _scale_pos_weight, apply_blend

    split = _holdout_split(train_start_season=frozen.get("train_season_start", TRAIN_START_SEASON))
    train_df = df[df["season"].isin(split.train_seasons)]
    holdout_df = df[df["season"].isin(split.holdout_seasons)].sort_values(["season", "gsis_id"])
    spw = _scale_pos_weight(train_df["breakout"])

    logit_model, logit_imputer, logit_scaler = frozen["_holdout_models"]["logistic"] if "_holdout_models" in frozen else (None, None, None)
    # frozen["_holdout_models"] isn't persisted on `result["frozen"]` after save_artifacts strips
    # private keys, but tiny_result's `result` dict is the raw run_full_pipeline return, which
    # still carries it (save_artifacts only strips a *copy* it writes to disk).
    assert logit_model is not None, f"{pos}: expected _holdout_models on the in-memory result"
    logit_cols = frozen["logistic_feature_cols"]
    logit_pred = _logistic_predict(logit_model, logit_imputer, logit_scaler, holdout_df, logit_cols)

    per_seed_blends = []
    for seed, info in frozen["per_seed"].items():
        lgbm_model = _fit_holdout_lgbm(train_df, tree_cols, info["lgbm_params"], info["lgbm_final_n_estimators"], seed, spw)
        xgb_model = _fit_holdout_xgb(train_df, tree_cols, info["xgb_params"], info["xgb_final_n_estimators"], seed, spw)
        lgbm_pred = lgbm_model.predict_proba(holdout_df[tree_cols])[:, 1]
        xgb_pred = xgb_model.predict_proba(holdout_df[tree_cols])[:, 1]
        per_seed_blends.append(apply_blend(info["blend_weights"], lgbm=lgbm_pred, xgb=xgb_pred, logistic=logit_pred))

    recomputed_mean = np.mean(np.vstack(per_seed_blends), axis=0)
    actual = holdout_preds.sort_values(["season", "gsis_id"])["pred_blend_raw"].to_numpy()
    np.testing.assert_allclose(recomputed_mean, actual, rtol=1e-10, atol=1e-12, err_msg=pos)
