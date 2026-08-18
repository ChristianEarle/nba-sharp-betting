"""Tests for src/models/vacancy_gate.py (v2.4 -- the binding team-change/vacancy capacity

gates, classifier + quantile). Pure decision logic is exercised against synthetic frames;
run_vacancy_position/run_vacancy_position_quantile additionally touch a real saved bundle +
real features, so they skip (the repo-wide "_skip_if_*" convention) when a position's
bundle/features aren't cached -- same pattern tests/test_efficiency_gate.py uses for the
v2.2 classifier gate this module's classifier half delegates straight into.
"""

from __future__ import annotations

from dataclasses import replace

import joblib
import pandas as pd
import pytest

from src.models import quantile as qmod
from src.models import train
from src.models import vacancy_gate as vg


def _skip_if_missing(pos: str) -> None:
    spec = train.position_spec(pos)
    if not spec.artifact_path.exists():
        pytest.skip(f"{spec.artifact_path} not built; run `python -m src.models.train {pos}`")
    if not (spec.features_path.exists() and spec.labels_path.exists()):
        pytest.skip(f"features_{pos}.parquet / labels.parquet not built")


def _skip_if_quantile_missing(pos: str) -> None:
    spec = qmod.quantile_spec(pos)
    if not spec.artifact_path.exists():
        pytest.skip(f"{spec.artifact_path} not built; run `python -m src.models.quantile {pos}`")
    if not (spec.features_path.exists() and spec.labels_path.exists()):
        pytest.skip(f"features_{pos}.parquet / labels.parquet not built")


# --------------------------------------------------------------------------
# vacancy_columns_for -- pure
# --------------------------------------------------------------------------


def test_vacancy_columns_for_wr_te_rb_includes_all_five() -> None:
    for pos in ("wr", "te", "rb"):
        cols = vg.vacancy_columns_for(pos)
        assert cols == [
            "qb_continuity",
            "vacated_td_share",
            "vacated_goal_line_carry_share",
            "max_single_vacated_target_share",
            "max_single_vacated_carry_share",
        ]


def test_vacancy_columns_for_qb_is_vacated_td_share_only() -> None:
    assert vg.vacancy_columns_for("qb") == ["vacated_td_share"]


def test_vacancy_columns_for_unknown_position_is_empty() -> None:
    assert vg.vacancy_columns_for("k") == []


# --------------------------------------------------------------------------
# Part A: classifier gate reuses efficiency_gate's engine -- gate_decision/metrics_table
# themselves are already tested in tests/test_efficiency_gate.py; here we only check the
# "plus_vacancy" variant-name wiring is actually used end to end (not "plus_efficiency").
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_run_vacancy_position_smoke(pos: str, tmp_path) -> None:
    _skip_if_missing(pos)
    real_spec = train.position_spec(pos)
    extra_cols = vg.vacancy_columns_for(pos)
    df_cols = pd.read_parquet(real_spec.features_path, columns=None).columns
    if not all(c in df_cols for c in extra_cols):
        pytest.skip(f"features_{pos}.parquet missing v2.4 vacancy columns; rebuild features first")

    bundle = joblib.load(real_spec.artifact_path)
    fake_config_path = tmp_path / f"model_{pos}.yaml"
    fake_config_path.write_text("some: config\nexcluded_features: [implied_ppg, implied_win_prob, has_vegas]\nmore: stuff\n")
    fake_artifact_path = tmp_path / f"{pos}_model_bundle.joblib"
    joblib.dump(bundle, fake_artifact_path)
    # report_path/metrics_json_path must ALSO be faked (not just config/artifact) -- a
    # genuine KEEP decision calls promote_position, which writes to spec.report_path/
    # metrics_json_path; leaving those pointed at the real outputs/model_{pos}_{report.md,
    # metrics.json} would silently overwrite them with a hypothetical promoted-bundle
    # description that does NOT match the untouched real artifact_path bundle, corrupting
    # the report/bundle-consistency invariant test_metrics_json_pooled_holdout_pr_auc_
    # matches_shipped_bundle depends on.
    fake_report_path = tmp_path / f"model_{pos}_report.md"
    fake_metrics_path = tmp_path / f"model_{pos}_metrics.json"
    fake_spec = replace(
        real_spec, config_path=fake_config_path, artifact_path=fake_artifact_path,
        report_path=fake_report_path, metrics_json_path=fake_metrics_path,
    )

    original_position_spec = train.position_spec
    train.position_spec = lambda p, _fake=fake_spec, _real=pos: (_fake if p == _real else original_position_spec(p))
    try:
        result = vg.run_vacancy_position(pos)
    finally:
        train.position_spec = original_position_spec

    assert result is not None
    assert result["position"] == pos
    assert isinstance(result["keep"], bool)
    assert set(result["table"]["variant"].unique()) == {"incumbent", "plus_vacancy"}

    # The REAL on-disk bundle/config must be untouched by this test.
    real_bundle_still = joblib.load(real_spec.artifact_path)
    assert real_bundle_still["tree_feature_cols"] == bundle["tree_feature_cols"]


# --------------------------------------------------------------------------
# Part B: quantile gate -- pure decision logic
# --------------------------------------------------------------------------


def _qmetrics(spearman: float, pinball50: float, coverage: float = 0.8, n: int = 100) -> dict:
    return {
        "spearman_q50_actual": spearman,
        "pinball_by_alpha": {"0.10": 1.0, "0.25": 1.0, "0.50": pinball50, "0.75": 1.0, "0.90": 1.0},
        "coverage_q10_q90": coverage,
        "n": n,
    }


def test_quantile_gate_decision_keeps_on_improvement() -> None:
    keep, reason = vg.quantile_gate_decision(_qmetrics(0.60, 1.50), _qmetrics(0.65, 1.40))
    assert keep is True
    assert "KEEP" in reason


def test_quantile_gate_decision_keeps_on_tie_spearman_and_pinball_within_tolerance() -> None:
    base = _qmetrics(0.60, 1.50)
    variant = _qmetrics(0.60 - vg.QUANTILE_SPEARMAN_TOLERANCE, 1.50 * (1 + vg.QUANTILE_PINBALL_TOLERANCE))
    keep, reason = vg.quantile_gate_decision(base, variant)
    assert keep is True


def test_quantile_gate_decision_rejects_spearman_regression_even_with_better_pinball() -> None:
    base = _qmetrics(0.60, 1.50)
    variant = _qmetrics(0.60 - vg.QUANTILE_SPEARMAN_TOLERANCE - 0.001, 1.00)
    keep, reason = vg.quantile_gate_decision(base, variant)
    assert keep is False
    assert "REGRESSED" in reason


def test_quantile_gate_decision_rejects_pinball_regression_beyond_tolerance() -> None:
    base = _qmetrics(0.60, 1.50)
    variant = _qmetrics(0.60, 1.50 * (1 + vg.QUANTILE_PINBALL_TOLERANCE) + 0.001)
    keep, reason = vg.quantile_gate_decision(base, variant)
    assert keep is False
    assert "REJECT" in reason


def test_quantile_gate_decision_rejects_both_regress() -> None:
    keep, reason = vg.quantile_gate_decision(_qmetrics(0.60, 1.00), _qmetrics(0.30, 3.00))
    assert keep is False


def test_quantile_bundle_already_promoted_detects_promoted_and_unpromoted_bundles() -> None:
    assert vg.quantile_bundle_already_promoted({"tree_feature_cols": ["a", "b_n1"]}, ["b_n1"]) is True
    assert vg.quantile_bundle_already_promoted({"tree_feature_cols": ["a"]}, ["b_n1"]) is False
    assert vg.quantile_bundle_already_promoted({"tree_feature_cols": ["a"]}, []) is False


# --------------------------------------------------------------------------
# add_to_excluded_features_quantile -- pure, tmp file
# --------------------------------------------------------------------------


def test_add_to_excluded_features_quantile_appends_new_columns(tmp_path) -> None:
    cfg_path = tmp_path / "model_rb.yaml"
    cfg_path.write_text("some: config\nexcluded_features_quantile: []\nmore: stuff\n")
    vg.add_to_excluded_features_quantile(cfg_path, ["qb_continuity", "vacated_td_share"])
    text = cfg_path.read_text()
    assert "excluded_features_quantile: [qb_continuity, vacated_td_share]" in text
    assert "some: config" in text and "more: stuff" in text


def test_add_to_excluded_features_quantile_is_idempotent(tmp_path) -> None:
    cfg_path = tmp_path / "model_rb.yaml"
    cfg_path.write_text("excluded_features_quantile: [qb_continuity]\n")
    vg.add_to_excluded_features_quantile(cfg_path, ["qb_continuity"])
    vg.add_to_excluded_features_quantile(cfg_path, ["qb_continuity"])
    text = cfg_path.read_text()
    assert text.count("qb_continuity") == 1


def test_add_to_excluded_features_quantile_creates_missing_key(tmp_path) -> None:
    cfg_path = tmp_path / "model_rb.yaml"
    cfg_path.write_text("some: config\n")
    vg.add_to_excluded_features_quantile(cfg_path, ["vacated_td_share"])
    text = cfg_path.read_text()
    assert "excluded_features_quantile: [vacated_td_share]" in text


# --------------------------------------------------------------------------
# run_vacancy_position_quantile -- real bundle, redirected to tmp
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_run_vacancy_position_quantile_smoke(pos: str, tmp_path) -> None:
    _skip_if_quantile_missing(pos)
    real_qspec = qmod.quantile_spec(pos)
    real_pspec = train.position_spec(pos)
    extra_cols = vg.vacancy_columns_for(pos)
    df_cols = pd.read_parquet(real_qspec.features_path, columns=None).columns
    if not all(c in df_cols for c in extra_cols):
        pytest.skip(f"features_{pos}.parquet missing v2.4 vacancy columns; rebuild features first")

    bundle = joblib.load(real_qspec.artifact_path)
    fake_config_path = tmp_path / f"model_{pos}.yaml"
    fake_config_path.write_text("some: config\nexcluded_features_quantile: []\nmore: stuff\n")
    fake_artifact_path = tmp_path / f"{pos}_quantile_bundle.joblib"
    joblib.dump(bundle, fake_artifact_path)
    fake_report = tmp_path / f"model_{pos}_quantile_report.md"
    fake_metrics = tmp_path / f"model_{pos}_quantile_metrics.json"
    fake_qspec = replace(real_qspec, config_path=fake_config_path, artifact_path=fake_artifact_path, report_path=fake_report, metrics_json_path=fake_metrics)

    original_quantile_spec = qmod.quantile_spec
    original_position_spec = train.position_spec
    qmod.quantile_spec = lambda p, _fake=fake_qspec, _real=pos: (_fake if p == _real else original_quantile_spec(p))
    train.position_spec = lambda p, _f=fake_config_path, _real=pos: (
        replace(original_position_spec(p), config_path=_f) if p == _real else original_position_spec(p)
    )
    try:
        result = vg.run_vacancy_position_quantile(pos)
    finally:
        qmod.quantile_spec = original_quantile_spec
        train.position_spec = original_position_spec

    assert result is not None
    assert result["position"] == pos
    assert isinstance(result["keep"], bool)
    assert set(result["table"]["variant"].unique()) == {"incumbent", "plus_vacancy"}
    if result["keep"]:
        assert fake_artifact_path.exists()
        new_bundle = joblib.load(fake_artifact_path)
        for c in extra_cols:
            assert c in new_bundle["tree_feature_cols"]
    else:
        text = fake_config_path.read_text()
        for c in extra_cols:
            assert c in text

    # The REAL on-disk bundle/config must be untouched by this test.
    real_bundle_still = joblib.load(real_qspec.artifact_path)
    assert real_bundle_still["tree_feature_cols"] == bundle["tree_feature_cols"]
