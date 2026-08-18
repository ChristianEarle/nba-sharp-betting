"""Tests for the v2.0 quantile regression head (src/models/quantile.py) and its

derived quantities (src/inference/projections.py). Pure-logic tests (non-crossing,
threshold-curve inversion, ladder normalization) use small synthetic inputs and have
no data dependency at all; anything that needs a real trained model or
labels.parquet/features_{pos}.parquet skips cleanly when that data isn't built, per
this repo's existing ``_skip_if_*`` convention (see tests/test_models_positions.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference import projections as pj
from src.models import quantile as qm
from src.models.train import position_spec

_POSITIONS = ("wr", "rb", "te", "qb")


def _quantile_data_available(pos: str) -> bool:
    spec = qm.quantile_spec(pos)
    return spec.features_path.exists() and spec.labels_path.exists()


def _skip_if_data_missing(pos: str) -> None:
    if not _quantile_data_available(pos):
        pytest.skip(f"features_{pos}.parquet / labels.parquet not built")


# --------------------------------------------------------------------------
# Non-crossing (pure logic, synthetic)
# --------------------------------------------------------------------------


def test_enforce_noncrossing_sorts_violated_rows() -> None:
    # Row 0 is already sorted; row 1 is deliberately out of order (q50 < q25, q90 < q75).
    wide = pd.DataFrame(
        {
            "q10": [4.0, 9.0],
            "q25": [6.0, 5.0],
            "q50": [8.0, 3.0],
            "q75": [10.0, 12.0],
            "q90": [12.0, 7.0],
        }
    )
    out = qm.enforce_noncrossing(wide)
    cols = [qm.ALPHA_COL[a] for a in sorted(qm.ALPHAS)]
    arr = out[cols].to_numpy()
    assert (np.diff(arr, axis=1) >= 0).all(), "every row must be non-decreasing left to right after enforcement"
    # row 0's values are unchanged (already sorted) -- just resorted to identity.
    assert list(arr[0]) == [4.0, 6.0, 8.0, 10.0, 12.0]
    # row 1's multiset of values is preserved, just reordered ascending.
    assert sorted([9.0, 5.0, 3.0, 12.0, 7.0]) == list(arr[1])


# --------------------------------------------------------------------------
# Threshold curve (data-dependent: real labels.parquet)
# --------------------------------------------------------------------------


def test_threshold_curve_monotone_in_k() -> None:
    if not position_spec("wr").labels_path.exists():
        pytest.skip("labels.parquet not built")
    thresholds = pj.build_rank_thresholds()
    assert set(thresholds["position"].unique()) <= {"QB", "RB", "WR", "TE"}
    for pos, g in thresholds.groupby("position"):
        g = g.sort_values("K")
        vals = g["threshold_ppg"].to_numpy()
        assert (np.diff(vals) <= 1e-9).all(), f"{pos}: threshold_ppg must be non-increasing in K"
        assert list(g["K"]) == list(range(1, pj.MAX_K + 1)), f"{pos}: expected a full K=1..{pj.MAX_K} curve"


# --------------------------------------------------------------------------
# Ladder normalization + segments (pure logic, synthetic thresholds + scores)
# --------------------------------------------------------------------------


def _synthetic_thresholds() -> pd.DataFrame:
    """A small hand-built, strictly-decreasing WR threshold curve, K=1..30 -- enough

    range to exercise interpolation/extrapolation without touching real data.
    """
    rows = [{"position": "WR", "K": k, "threshold_ppg": max(2.0, 22.0 - 0.6 * k)} for k in range(1, 31)]
    return pd.DataFrame(rows)


def _synthetic_scored(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q50 = rng.uniform(3.0, 20.0, n)
    spread_lo = rng.uniform(1.0, 4.0, n)
    spread_hi = rng.uniform(1.0, 4.0, n)
    return pd.DataFrame(
        {
            "q10": q50 - 2 * spread_lo,
            "q25": q50 - spread_lo,
            "q50": q50,
            "q75": q50 + spread_hi,
            "q90": q50 + 2 * spread_hi,
            "expectation_pos_rank": rng.integers(1, 60, n).astype(float),
        }
    )


@pytest.fixture(scope="module")
def synthetic_labels_cfg() -> dict:
    return {
        "thresholds": {"WR": {"finish_top": 18, "adp_worse_than": 31}},
    }


@pytest.fixture(scope="module")
def synthetic_projected(synthetic_labels_cfg) -> pd.DataFrame:
    thresholds = _synthetic_thresholds()
    scored = _synthetic_scored(400)
    scored["pos"] = "WR"
    proj = pj.compute_projections_for_position(scored, "WR", thresholds, synthetic_labels_cfg)
    proj["pos"] = "WR"
    return pj.attach_ladder(proj)


def test_ladder_monotone_per_player(synthetic_projected) -> None:
    df = synthetic_projected
    assert (df["p_elite"] <= df["p_startable"] + 1e-9).all()
    assert (df["p_startable"] <= df["p_useful"] + 1e-9).all()
    assert (df["p_useful"] <= 1.0 + 1e-9).all()
    assert (df["p_elite"] >= 0).all()


def test_ladder_segments_sum_to_one(synthetic_projected) -> None:
    df = synthetic_projected
    total = df["seg_elite"] + df["seg_starter"] + df["seg_useful"] + df["seg_bust"]
    assert np.allclose(total.to_numpy(), 1.0, atol=1e-6)
    for col in ("seg_elite", "seg_starter", "seg_useful", "seg_bust"):
        assert (df[col] >= -1e-9).all(), f"{col} went materially negative"


def test_ladder_position_sums_match_targets(synthetic_projected) -> None:
    df = synthetic_projected
    n = len(df)
    k = 18.0
    for col, target in (("p_elite", 5.0), ("p_startable", k), ("p_useful", 2.0 * k)):
        total = df[col].sum()
        # +/-1% of the target, per the brief -- the monotonic upward-clip pass can push
        # the realized sum slightly off its independently-scaled target.
        assert abs(total - target) <= 0.01 * target + 1e-6, f"{col}: sum={total:.3f} target={target} (n={n})"


def test_p_startable_bounded() -> None:
    thresholds = _synthetic_thresholds()
    cfg = {"thresholds": {"WR": {"finish_top": 18, "adp_worse_than": 31}}}
    scored = _synthetic_scored(200)
    proj = pj.compute_projections_for_position(scored, "WR", thresholds, cfg)
    assert (proj["p_startable_raw"] > 0).all()
    assert (proj["p_startable_raw"] <= pj.CDF_HI).all()


def test_value_gap_sign_convention() -> None:
    """A player drafted much later (high expectation_pos_rank) than his predicted finish

    (low predicted_finish_rank) should have a POSITIVE value_gap (undervalued).
    """
    thresholds = _synthetic_thresholds()
    cfg = {"thresholds": {"WR": {"finish_top": 18, "adp_worse_than": 31}}}
    scored = pd.DataFrame({"q10": [14.0], "q25": [16.0], "q50": [18.0], "q75": [19.0], "q90": [20.0], "expectation_pos_rank": [45.0]})
    proj = pj.compute_projections_for_position(scored, "WR", thresholds, cfg)
    # q50=18 is near the top of this curve (K=1's threshold is ~21.4) -> predicted
    # finish rank should be small, well below the 45 he was drafted at.
    assert proj["predicted_finish_rank"].iloc[0] < proj["expectation_pos_rank"].iloc[0]
    assert proj["value_gap"].iloc[0] > 0


def test_eligible_lens_excludes_ineligible_players(synthetic_projected) -> None:
    eligible = synthetic_projected[synthetic_projected["breakout_eligible"]]
    assert (eligible["expectation_pos_rank"] >= 31).all()
    ineligible = synthetic_projected[~synthetic_projected["breakout_eligible"]]
    if len(ineligible):
        assert (ineligible["expectation_pos_rank"] < 31).all()


# --------------------------------------------------------------------------
# Full tiny-config pipeline: non-crossing, sane metrics, determinism
# --------------------------------------------------------------------------


@pytest.fixture(scope="module", params=_POSITIONS)
def tiny_quantile_result(request, tmp_path_factory):
    pos = request.param
    _skip_if_data_missing(pos)
    spec = qm.quantile_spec(pos)
    cfg = qm.train.load_config(spec.config_path)
    trials = cfg["quantile"]["optuna"]["tiny_trials"]
    out = tmp_path_factory.mktemp(f"tiny_quantile_{pos}")
    result = qm.run_full_pipeline(spec, cfg=cfg, n_trials=trials, seed=42, output_root=out)
    return pos, result


def test_holdout_predictions_noncrossing(tiny_quantile_result) -> None:
    pos, result = tiny_quantile_result
    wide = result["holdout_wide"]
    cols = [qm.ALPHA_COL[a] for a in sorted(qm.ALPHAS)]
    arr = wide[cols].to_numpy()
    assert (np.diff(arr, axis=1) >= -1e-9).all(), pos


def test_validation_predictions_noncrossing(tiny_quantile_result) -> None:
    pos, result = tiny_quantile_result
    wide = result["val_wide"]
    cols = [qm.ALPHA_COL[a] for a in sorted(qm.ALPHAS)]
    arr = wide[cols].to_numpy()
    assert (np.diff(arr, axis=1) >= -1e-9).all(), pos


def test_holdout_metrics_are_finite_and_sane(tiny_quantile_result) -> None:
    pos, result = tiny_quantile_result
    m = result["holdout_metrics"]
    for alpha, pinball in m["pinball_by_alpha"].items():
        assert pinball >= 0 and np.isfinite(pinball), f"{pos} alpha={alpha}"
    assert 0.0 <= m["coverage_q10_q90"] <= 1.0, pos
    assert np.isnan(m["spearman_q50_actual"]) or -1.0 <= m["spearman_q50_actual"] <= 1.0, pos


def test_bundle_saved_with_five_alphas(tiny_quantile_result) -> None:
    pos, result = tiny_quantile_result
    spec = result["spec"]
    import joblib

    bundle = joblib.load(spec.artifact_path)
    assert set(bundle["models"].keys()) == set(qm.ALPHAS)


@pytest.mark.parametrize("pos", _POSITIONS)
def test_determinism(pos: str, tmp_path) -> None:
    _skip_if_data_missing(pos)
    spec = qm.quantile_spec(pos)
    cfg = qm.train.load_config(spec.config_path)
    trials = cfg["quantile"]["optuna"]["tiny_trials"]
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    r1 = qm.run_full_pipeline(spec, cfg=cfg, n_trials=trials, seed=42, output_root=out1)
    r2 = qm.run_full_pipeline(spec, cfg=cfg, n_trials=trials, seed=42, output_root=out2)
    cols = [qm.ALPHA_COL[a] for a in sorted(qm.ALPHAS)]
    a1 = r1["holdout_wide"].sort_values(["season", "gsis_id"])[cols].to_numpy()
    a2 = r2["holdout_wide"].sort_values(["season", "gsis_id"])[cols].to_numpy()
    assert np.array_equal(a1, a2), f"{pos}: two identical-seed runs produced different holdout predictions"


# --------------------------------------------------------------------------
# Deliverable 3: comparison-gate table exists (data-dependent -- needs real
# production classifier + quantile bundles, not the tiny-config ones above)
# --------------------------------------------------------------------------


def test_comparison_gate_table_schema() -> None:
    have_any = any(
        position_spec(p).artifact_path.exists() and qm.quantile_spec(p).artifact_path.exists() and position_spec(p).metrics_json_path.exists()
        for p in _POSITIONS
    )
    if not have_any:
        pytest.skip("no production classifier+quantile bundle pair on disk yet")
    table = pj.comparison_gate()
    assert list(table.columns) == [
        "position",
        "quantile_top10_precision",
        "classifier_top10_precision",
        "n_eligible_holdout",
        "primary_engine",
    ]
    assert set(table["primary_engine"]) <= {"quantile", "classifier"}
    assert pj.COMPARISON_GATE_JSON_PATH.exists()
    assert pj.COMPARISON_GATE_MD_PATH.exists()
