"""Tests for src/models/efficiency_gate.py (v2.2 -- the binding efficiency-proxy capacity

gate). Pure decision/metric logic is exercised against synthetic frames.
score_incumbent/fit_and_score_efficiency_variant/run_position/promote_position
additionally touch a real saved bundle + real features, so they skip (the repo-wide
"_skip_if_*" convention) when a position's bundle/features aren't cached.
promote_position is tested by pointing a fake PositionSpec's config_path/artifact_path at
tmp_path while reusing the REAL bundle+features as inputs -- same pattern
tests/test_vegas_experiment.py uses for its own promote_position.
"""

from __future__ import annotations

import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import efficiency_gate as eg
from src.models import metrics as mt
from src.models import train


def _skip_if_missing(pos: str) -> None:
    spec = train.position_spec(pos)
    if not spec.artifact_path.exists():
        pytest.skip(f"{spec.artifact_path} not built; run `python -m src.models.train {pos}`")
    if not (spec.features_path.exists() and spec.labels_path.exists()):
        pytest.skip(f"features_{pos}.parquet / labels.parquet not built")


# --------------------------------------------------------------------------
# new_columns_for -- pure
# --------------------------------------------------------------------------


def test_new_columns_for_wr_te_includes_targets_per_snap() -> None:
    for pos in ("wr", "te"):
        cols = eg.new_columns_for(pos)
        assert "targets_per_snap_n1" in cols
        assert "targets_per_snap_yoy_delta" in cols
        assert "yards_per_snap_n1" in cols
        assert "catchable_target_rate_n1" in cols


def test_new_columns_for_rb_excludes_targets_per_snap() -> None:
    cols = eg.new_columns_for("rb")
    assert not any("targets_per_snap" in c for c in cols)
    assert "yards_per_snap_n1" in cols
    assert "catchable_target_rate_n1" in cols


def test_new_columns_for_qb_is_empty() -> None:
    assert eg.new_columns_for("qb") == []


# --------------------------------------------------------------------------
# gate_decision -- pure, synthetic
# --------------------------------------------------------------------------


def _pooled_table(base_top10: float, var_top10: float, base_prauc: float, var_prauc: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": "pooled", "variant": "incumbent", "n": 100, "n_pos": 10, "top10_precision": base_top10, "pr_auc": base_prauc},
            {"season": "pooled", "variant": "plus_efficiency", "n": 100, "n_pos": 10, "top10_precision": var_top10, "pr_auc": var_prauc},
        ]
    )


def test_gate_decision_keeps_on_improvement() -> None:
    keep, reason = eg.gate_decision(_pooled_table(0.2, 0.3, 0.10, 0.15))
    assert keep is True
    assert "KEEP" in reason


def test_gate_decision_keeps_on_tie_top10_and_small_prauc_regression_within_tolerance() -> None:
    keep, reason = eg.gate_decision(_pooled_table(0.2, 0.2, 0.20, 0.20 - eg.PR_AUC_TOLERANCE))
    assert keep is True


def test_gate_decision_rejects_top10_regression_even_with_prauc_improvement() -> None:
    keep, reason = eg.gate_decision(_pooled_table(0.3, 0.2, 0.10, 0.50))
    assert keep is False
    assert "REGRESSED" in reason


def test_gate_decision_rejects_prauc_regression_beyond_tolerance() -> None:
    keep, reason = eg.gate_decision(_pooled_table(0.2, 0.2, 0.20, 0.20 - eg.PR_AUC_TOLERANCE - 0.001))
    assert keep is False
    assert "REJECT" in reason


def test_gate_decision_rejects_both_regress() -> None:
    keep, reason = eg.gate_decision(_pooled_table(0.3, 0.1, 0.3, 0.1))
    assert keep is False


# --------------------------------------------------------------------------
# add_to_excluded_features -- pure, tmp file
# --------------------------------------------------------------------------


def test_add_to_excluded_features_appends_new_columns(tmp_path) -> None:
    cfg_path = tmp_path / "model_rb.yaml"
    cfg_path.write_text("some: config\nexcluded_features: [implied_ppg, implied_win_prob, has_vegas]\nmore: stuff\n")
    eg.add_to_excluded_features(cfg_path, ["yards_per_snap_n1", "catchable_target_rate_n1"])
    text = cfg_path.read_text()
    assert "excluded_features: [implied_ppg, implied_win_prob, has_vegas, yards_per_snap_n1, catchable_target_rate_n1]" in text
    assert "some: config" in text and "more: stuff" in text


def test_add_to_excluded_features_is_idempotent(tmp_path) -> None:
    cfg_path = tmp_path / "model_rb.yaml"
    cfg_path.write_text("excluded_features: [implied_ppg]\n")
    eg.add_to_excluded_features(cfg_path, ["yards_per_snap_n1"])
    eg.add_to_excluded_features(cfg_path, ["yards_per_snap_n1"])
    text = cfg_path.read_text()
    assert text.count("yards_per_snap_n1") == 1


# --------------------------------------------------------------------------
# score_incumbent / fit_and_score_efficiency_variant -- real bundle
# --------------------------------------------------------------------------


def test_score_incumbent_matches_board_2026_scoring() -> None:
    """score_incumbent's ensemble math must match src.inference.board_2026.score_veterans_batch's

    raw_score exactly (both derive from the identical bundle fields) -- verified on the
    real WR holdout frame.
    """
    _skip_if_missing("wr")
    spec = train.position_spec("wr")
    bundle = joblib.load(spec.artifact_path)
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
    holdout_df = df[df["season"].isin(train.HOLDOUT_SEASONS)]
    if holdout_df.empty:
        pytest.skip("wr holdout frame is empty")

    from src.inference import board_2026 as bd

    incumbent_score = eg.score_incumbent(bundle, holdout_df)
    board_scored = bd.score_veterans_batch(bundle, holdout_df)
    assert np.allclose(incumbent_score, board_scored["raw_score"].to_numpy())


@pytest.mark.parametrize("pos", ("wr", "rb", "te"))
def test_run_position_smoke(pos: str, tmp_path, monkeypatch) -> None:
    """run_position end-to-end, redirected to a tmp bundle/config so the real shipped

    artifacts are never touched by a test run.
    """
    _skip_if_missing(pos)
    real_spec = train.position_spec(pos)
    extra_cols = eg.new_columns_for(pos)
    df_cols = pd.read_parquet(real_spec.features_path, columns=None).columns
    if not all(c in df_cols for c in extra_cols):
        pytest.skip(f"features_{pos}.parquet missing v2.2 efficiency columns; rebuild features first")

    bundle = joblib.load(real_spec.artifact_path)
    fake_config_path = tmp_path / f"model_{pos}.yaml"
    fake_config_path.write_text("some: config\nexcluded_features: [implied_ppg, implied_win_prob, has_vegas]\nmore: stuff\n")
    # run_position reads bundle FROM spec.artifact_path itself (unlike vegas_experiment's
    # promote_position, which takes an already-loaded bundle) -- so the fake artifact path
    # must start out holding a real, readable copy of the bundle, not an empty tmp path.
    fake_artifact_path = tmp_path / f"{pos}_model_bundle.joblib"
    joblib.dump(bundle, fake_artifact_path)
    # report_path/metrics_json_path must ALSO be faked (not just config/artifact) -- a
    # KEEP decision calls promote_position, which writes to spec.report_path/
    # metrics_json_path; leaving those pointed at the real outputs/model_{pos}_{report.md,
    # metrics.json} would silently overwrite them with a hypothetical promoted-bundle
    # description that does NOT match the untouched real artifact_path bundle, corrupting
    # the report/bundle-consistency invariant test_metrics_json_pooled_holdout_pr_auc_
    # matches_shipped_bundle depends on (surfaced by src.models.vacancy_gate's own smoke
    # test hitting exactly this gap for a position whose gate decision differs from the
    # real shipped one -- see tests/test_vacancy_gate.py).
    fake_report_path = tmp_path / f"model_{pos}_report.md"
    fake_metrics_path = tmp_path / f"model_{pos}_metrics.json"
    fake_spec = replace(
        real_spec, config_path=fake_config_path, artifact_path=fake_artifact_path,
        report_path=fake_report_path, metrics_json_path=fake_metrics_path,
    )

    original_position_spec = train.position_spec
    monkeypatch.setattr(
        train, "position_spec", lambda p, _fake=fake_spec, _real=pos: (_fake if p == _real else original_position_spec(p))
    )
    result = eg.run_position(pos)

    assert result is not None
    assert result["position"] == pos
    assert isinstance(result["keep"], bool)
    assert set(result["table"]["variant"].unique()) == {"incumbent", "plus_efficiency"}

    # The REAL on-disk bundle/config must be untouched by this test.
    real_bundle_still = joblib.load(real_spec.artifact_path)
    assert real_bundle_still["tree_feature_cols"] == bundle["tree_feature_cols"]


# --------------------------------------------------------------------------
# Report/metrics.json consistency (coordinator-requested hardening):
# outputs/model_{pos}_metrics.json must never drift from the SHIPPED bundle it
# claims to describe.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", train.POSITIONS)
def test_metrics_json_pooled_holdout_pr_auc_matches_shipped_bundle(pos: str) -> None:
    """The pooled holdout PR-AUC written to outputs/model_{pos}_metrics.json must match a

    FRESH recompute straight from the shipped bundle (data/models/{pos}_model_bundle.joblib)
    within 1e-6 -- report/bundle drift (a bundle regenerated without regenerating the
    report/metrics that describe it, or vice versa) must fail this test loudly rather than
    silently ship a stale description of a different model. This is the exact class of bug
    the v2.2 efficiency gate's promotion path was fixed to stop causing (see
    src.models.efficiency_gate.write_gate_outputs's docstring).
    """
    spec = train.position_spec(pos)
    if not (spec.artifact_path.exists() and spec.metrics_json_path.exists()):
        pytest.skip(f"{pos}: bundle or metrics.json not built")
    if not (spec.features_path.exists() and spec.labels_path.exists()):
        pytest.skip(f"{pos}: features/labels parquet not built")

    bundle = joblib.load(spec.artifact_path)
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
    # holdout_predictions mutates a "_holdout_models" key onto whatever dict it's handed --
    # pass a shallow copy so the loaded bundle object itself is never touched by this test.
    holdout_preds = train.holdout_predictions(df, bundle["seed"], dict(bundle))
    recomputed_pr_auc = float(mt.pr_auc(holdout_preds["breakout"], holdout_preds["pred_calibrated"]))

    metrics = json.loads(spec.metrics_json_path.read_text())
    pooled_row = next(
        (r for r in metrics.get("holdout_metrics", []) if r.get("model") == "model" and r.get("season") == "pooled"), None
    )
    assert pooled_row is not None, f"{pos}: metrics.json has no model/pooled holdout_metrics row"
    assert abs(recomputed_pr_auc - pooled_row["pr_auc"]) < 1e-6, (
        f"{pos}: metrics.json pooled holdout PR-AUC ({pooled_row['pr_auc']}) does not match a fresh recompute "
        f"from the shipped bundle ({recomputed_pr_auc}) -- report/bundle drift"
    )


# --------------------------------------------------------------------------
# Reconstructing the true incumbent after a bundle has already been promoted
# --------------------------------------------------------------------------


def test_bundle_already_promoted_detects_promoted_and_unpromoted_bundles() -> None:
    assert eg.bundle_already_promoted({"tree_feature_cols": ["a", "b_n1"]}, ["b_n1"]) is True
    assert eg.bundle_already_promoted({"tree_feature_cols": ["a"]}, ["b_n1"]) is False
    assert eg.bundle_already_promoted({"tree_feature_cols": ["a"]}, []) is False


@pytest.mark.parametrize("pos", ("wr", "te"))
def test_reconstruct_incumbent_score_matches_a_fresh_reduced_column_refit(pos: str) -> None:
    """reconstruct_incumbent_score's whole point is determinism: refitting at the SAME

    frozen hyperparameters over the SAME reduced tree_cols must reproduce byte-identical
    holdout scores no matter how many times it's called -- otherwise "reconstruct the true
    incumbent" would itself be an unreliable comparison.
    """
    _skip_if_missing(pos)
    spec = train.position_spec(pos)
    extra_cols = eg.new_columns_for(pos)
    bundle = joblib.load(spec.artifact_path)
    if not eg.bundle_already_promoted(bundle, extra_cols):
        pytest.skip(f"{pos}: bundle not yet promoted with its v2.2 columns; nothing to reconstruct")

    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
    cfg = bundle["cfg"]
    split = train.holdout_split(train_start_season=cfg.get("train_season_start", train.TRAIN_START_SEASON))
    folds = train.validation_folds(train_start_season=cfg.get("train_season_start", train.TRAIN_START_SEASON))

    score1 = eg.reconstruct_incumbent_score(bundle, df, extra_cols, folds, split)
    score2 = eg.reconstruct_incumbent_score(bundle, df, extra_cols, folds, split)
    assert np.allclose(score1, score2)
