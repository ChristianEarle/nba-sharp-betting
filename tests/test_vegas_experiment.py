"""Tests for src/models/vegas_experiment.py (v1.5 Phase C, Deliverable 3 -- the binding
v1.5 promotion gate).

Pure decision/metric logic is exercised against synthetic frames.
score_baseline / fit_and_score_vegas_variant / run_position / promote_position
additionally touch a real saved bundle + real features, so they skip (the
repo-wide "_skip_if_*" convention) when a position's bundle/features aren't
cached. promote_position is tested by pointing a fake PositionSpec's
config_path/artifact_path at tmp_path while reusing the REAL bundle+features
as inputs -- promote_position never reads spec.features_path/labels_path
(only spec.config_path and spec.artifact_path), so this exercises the real
promotion write-path without ever touching the real config/bundle files.
"""

from __future__ import annotations

from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import train
from src.models import vegas_experiment as ve


def _skip_if_missing(pos: str) -> None:
    spec = train.position_spec(pos)
    if not spec.artifact_path.exists():
        pytest.skip(f"{spec.artifact_path} not built; run `python -m src.models.train {pos}`")
    if not (spec.features_path.exists() and spec.labels_path.exists()):
        pytest.skip(f"features_{pos}.parquet / labels.parquet not built")


# --------------------------------------------------------------------------
# promotion_decision -- pure, synthetic
# --------------------------------------------------------------------------


def _pooled_table(base_top10: float, vegas_top10: float, base_prauc: float = 0.2, vegas_prauc: float = 0.2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": "pooled", "variant": "v1.5_baseline", "n": 100, "n_pos": 10, "top10_precision": base_top10, "pr_auc": base_prauc},
            {"season": "pooled", "variant": "plus_vegas", "n": 100, "n_pos": 10, "top10_precision": vegas_top10, "pr_auc": vegas_prauc},
        ]
    )


def test_promotion_decision_promotes_on_strict_improvement() -> None:
    promote, reason = ve.promotion_decision(_pooled_table(0.2, 0.3))
    assert bool(promote) is True
    assert "PROMOTE" in reason


def test_promotion_decision_rejects_tie() -> None:
    promote, reason = ve.promotion_decision(_pooled_table(0.2, 0.2))
    assert bool(promote) is False
    assert "tie" in reason.lower()


def test_promotion_decision_rejects_regression() -> None:
    promote, reason = ve.promotion_decision(_pooled_table(0.3, 0.2))
    assert bool(promote) is False
    assert "regression" in reason.lower()


# --------------------------------------------------------------------------
# metrics_table -- pure, synthetic
# --------------------------------------------------------------------------


def test_metrics_table_shape_and_values() -> None:
    # 14 rows (> k=10) so top-10 precision actually filters instead of degenerating to the base rate.
    n = 14
    breakout = [1, 1] + [0] * (n - 2)
    holdout_df = pd.DataFrame({"season": [2024] * 7 + [2025] * 7, "breakout": breakout})
    # baseline ranks the two true positives (index 0, 1) on top; vegas ranks them dead last.
    baseline_score = np.array([1.0, 0.9] + [0.1 * (1 - i / n) for i in range(n - 2)])
    vegas_score = np.array([0.0, 0.0] + [0.9 - 0.01 * i for i in range(n - 2)])
    table = ve.metrics_table(holdout_df, baseline_score, vegas_score)

    assert set(table["season"].unique()) == {2024, 2025, "pooled"}
    assert set(table["variant"].unique()) == {"v1.5_baseline", "plus_vegas"}

    base_pooled = table[(table["season"] == "pooled") & (table["variant"] == "v1.5_baseline")].iloc[0]
    vegas_pooled = table[(table["season"] == "pooled") & (table["variant"] == "plus_vegas")].iloc[0]
    # baseline score ranks the two true positives in its top-10 -> both breakouts caught.
    assert base_pooled["top10_precision"] == pytest.approx(2 / 10)
    # vegas score ranks them dead last -> neither breakout makes the top-10.
    assert vegas_pooled["top10_precision"] == pytest.approx(0.0)
    assert base_pooled["n"] == n and base_pooled["n_pos"] == 2


# --------------------------------------------------------------------------
# score_baseline / fit_and_score_vegas_variant -- real bundle, fold boundary
# --------------------------------------------------------------------------


def test_fit_and_score_vegas_variant_uses_only_pre_holdout_training_rows() -> None:
    _skip_if_missing("qb")
    spec = train.position_spec("qb")
    bundle = joblib.load(spec.artifact_path)
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
    split = train.holdout_split()
    train_df = df[df["season"].isin(split.train_seasons)]
    holdout_df = df[df["season"].isin(split.holdout_seasons)]
    if train_df.empty or holdout_df.empty:
        pytest.skip("qb modeling frame too small for this check")

    assert train_df["season"].max() < holdout_df["season"].min()  # the fold-boundary invariant itself

    score, fitted = ve.fit_and_score_vegas_variant(bundle, train_df, holdout_df)
    assert len(score) == len(holdout_df)
    assert np.isfinite(score).all()
    assert set(ve.VEGAS_COLS) <= set(fitted["tree_cols"])


def test_score_baseline_matches_bundle_tree_cols_length() -> None:
    _skip_if_missing("qb")
    spec = train.position_spec("qb")
    bundle = joblib.load(spec.artifact_path)
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
    holdout_df = df[df["season"].isin(train.HOLDOUT_SEASONS)]
    if holdout_df.empty:
        pytest.skip("qb holdout frame is empty")
    score = ve.score_baseline(bundle, holdout_df)
    assert len(score) == len(holdout_df)
    assert np.isfinite(score).all()


# --------------------------------------------------------------------------
# run_position -- real bundle smoke test
# --------------------------------------------------------------------------


def test_run_position_qb_smoke() -> None:
    _skip_if_missing("qb")
    result = ve.run_position("qb")
    assert result is not None
    assert result["position"] == "qb"
    assert isinstance(result["promote"], (bool, np.bool_))
    assert set(result["table"]["variant"].unique()) == {"v1.5_baseline", "plus_vegas"}


# --------------------------------------------------------------------------
# promote_position -- real bundle+df, fake spec pointed at tmp_path so the
# real config/bundle files are never touched.
# --------------------------------------------------------------------------


def test_promote_position_writes_bundle_and_config(tmp_path) -> None:
    _skip_if_missing("qb")
    real_spec = train.position_spec("qb")
    bundle = joblib.load(real_spec.artifact_path)
    df = train.load_modeling_frame(real_spec.features_path, real_spec.labels_path, cfg=bundle["cfg"])

    fake_config_path = tmp_path / "model_qb.yaml"
    fake_config_path.write_text("some: config\nexcluded_features: [implied_ppg, implied_win_prob, has_vegas]\nmore: stuff\n")
    fake_artifact_path = tmp_path / "qb_model_bundle.joblib"
    fake_spec = replace(real_spec, config_path=fake_config_path, artifact_path=fake_artifact_path)

    info = ve.promote_position(fake_spec, bundle, df)

    assert fake_artifact_path.exists()
    new_bundle = joblib.load(fake_artifact_path)
    assert set(ve.VEGAS_COLS) <= set(new_bundle["tree_feature_cols"])
    assert new_bundle["cfg"]["excluded_features"] == []

    config_text = fake_config_path.read_text()
    assert "excluded_features: []" in config_text
    assert "implied_ppg, implied_win_prob, has_vegas" not in config_text.split("excluded_features")[0]

    assert "holdout_preds" in info and "holdout_reg" in info
    assert info["holdout_preds"]["pred_calibrated"].between(0.0, 1.0).all()

    # The REAL on-disk bundle/config for qb must be untouched by this test.
    assert real_spec.artifact_path.exists()
    real_bundle_still = joblib.load(real_spec.artifact_path)
    assert set(ve.VEGAS_COLS).isdisjoint(set(real_bundle_still["tree_feature_cols"]))
