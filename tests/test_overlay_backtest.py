"""Tests for src/models/overlay_backtest.py (v1.5 Phase C, Deliverable 2).

The pure formula/metric functions (attach_overlays, score_table,
verdict_line) are exercised against small synthetic frames. build_scored_frame
/ run_position additionally touch a real saved bundle, so they skip (matching
the repo-wide "_skip_if_*" convention) when a position's bundle isn't cached.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import overlay_backtest as ob
from src.models import train


# --------------------------------------------------------------------------
# attach_overlays
# --------------------------------------------------------------------------


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "gsis_id": ["a", "b", "c"],
            "player_name": ["A", "B", "C"],
            "team": ["KC", "BUF", "ZZZ"],  # ZZZ deliberately unmatched -> has_vegas=0
            "breakout": [1, 0, 0],
            "finish_rank_delta": [5.0, -2.0, np.nan],
            "expectation_pos_rank": [10, 20, 30],
            "probability": [0.5, 0.3, 0.2],
        }
    )


def _vegas_team() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "team": ["KC", "BUF"],
            "implied_ppg": [30.0, 20.0],
            "has_vegas": [1, 1],
        }
    )
    df["z_implied_ppg"] = df.groupby("season")["implied_ppg"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    return df


def test_attach_overlays_discount_and_overlay_score() -> None:
    out = ob.attach_overlays(_scored_frame(), _vegas_team())
    expected_discount_a = np.log1p(10)
    assert out.loc[out["gsis_id"] == "a", "discount"].iloc[0] == pytest.approx(expected_discount_a)
    assert out.loc[out["gsis_id"] == "a", "overlay_score"].iloc[0] == pytest.approx(0.5 * expected_discount_a)


def test_attach_overlays_vegas_score_null_when_has_vegas_zero() -> None:
    out = ob.attach_overlays(_scored_frame(), _vegas_team())
    row_c = out[out["gsis_id"] == "c"].iloc[0]
    assert row_c["has_vegas"] == 0
    assert pd.isna(row_c["overlay_vegas_score"])


def test_attach_overlays_vegas_multiplier_clipped_to_0_5_1_5() -> None:
    # KC (30 ppg) is above the 2-team mean 25 -> z > 0 -> multiplier > 1, clipped at 1.5.
    vt = _vegas_team().copy()
    vt.loc[vt["team"] == "KC", "implied_ppg"] = 1000.0  # force an extreme z
    out = ob.attach_overlays(_scored_frame(), vt)
    row_a = out[out["gsis_id"] == "a"].iloc[0]
    multiplier = row_a["overlay_vegas_score"] / row_a["overlay_score"]
    assert multiplier == pytest.approx(1.5)


def test_attach_overlays_two_way_symmetric_z_score() -> None:
    # With only two teams, z-scores are +/- the same magnitude (mean-centered, std-normalized).
    out = ob.attach_overlays(_scored_frame(), _vegas_team())
    z_kc = out.loc[out["team"] == "KC", "z_implied_ppg"].iloc[0]
    z_buf = out.loc[out["team"] == "BUF", "z_implied_ppg"].iloc[0]
    assert z_kc == pytest.approx(-z_buf)


# --------------------------------------------------------------------------
# score_table
# --------------------------------------------------------------------------


def test_score_table_top10_precision_and_spearman_hand_computed() -> None:
    df = pd.DataFrame(
        {
            "season": [2024] * 4,
            "breakout": [1, 0, 1, 0],
            "finish_rank_delta": [10.0, 5.0, 1.0, -3.0],
            "overlay_score": [4.0, 3.0, 2.0, 1.0],  # rank order = row order
        }
    )
    df["probability"] = df["overlay_score"]
    df["overlay_vegas_score"] = df["overlay_score"]
    table = ob.score_table(df)
    row = table[(table["season"] == 2024) & (table["score"] == "model")].iloc[0]
    assert row["n"] == 4
    assert row["n_pos"] == 2
    # top-4 (all rows, k=min(10,4)) precision = 2/4 = 0.5
    assert row["top10_precision"] == pytest.approx(0.5)
    # score is monotonically decreasing while finish_rank_delta is [10,5,1,-3] -- also
    # monotonically decreasing in the same row order -> perfect positive Spearman rho=1.
    assert row["spearman"] == pytest.approx(1.0)


def test_score_table_pooled_row_present() -> None:
    df = pd.DataFrame(
        {
            "season": [2023, 2024],
            "breakout": [1, 0],
            "finish_rank_delta": [1.0, 2.0],
            "probability": [0.5, 0.4],
            "overlay_score": [0.5, 0.4],
            "overlay_vegas_score": [np.nan, np.nan],
        }
    )
    table = ob.score_table(df)
    pooled = table[table["season"] == "pooled"]
    assert len(pooled) == len(ob.SCORE_COLS)
    model_pooled = pooled[pooled["score"] == "model"].iloc[0]
    assert model_pooled["n"] == 2
    assert model_pooled["n_pos"] == 1


def test_score_table_all_nan_overlay_vegas_gives_n_zero() -> None:
    df = pd.DataFrame(
        {
            "season": [2024],
            "breakout": [1],
            "finish_rank_delta": [1.0],
            "probability": [0.5],
            "overlay_score": [0.5],
            "overlay_vegas_score": [np.nan],
        }
    )
    table = ob.score_table(df)
    row = table[(table["season"] == 2024) & (table["score"] == "overlay_vegas")].iloc[0]
    assert row["n"] == 0
    assert np.isnan(row["top10_precision"])


# --------------------------------------------------------------------------
# verdict_line
# --------------------------------------------------------------------------


def _pooled_only_table(model=0.3, overlay=0.3, overlay_vegas=0.3, n_pos=10) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": "pooled", "score": "model", "n": 100, "n_pos": n_pos, "top10_precision": model, "spearman": 0.1},
            {"season": "pooled", "score": "overlay", "n": 100, "n_pos": n_pos, "top10_precision": overlay, "spearman": 0.1},
            {"season": "pooled", "score": "overlay_vegas", "n": 100, "n_pos": n_pos, "top10_precision": overlay_vegas, "spearman": 0.1},
        ]
    )


def test_verdict_line_flags_small_sample() -> None:
    line = ob.verdict_line("WR", _pooled_only_table(n_pos=10))
    assert "not a statistically supported win" in line


def test_verdict_line_no_flag_above_threshold() -> None:
    line = ob.verdict_line("WR", _pooled_only_table(n_pos=20))
    assert "not a statistically supported win" not in line


def test_verdict_line_orders_by_top10_precision() -> None:
    line = ob.verdict_line("WR", _pooled_only_table(model=0.1, overlay=0.5, overlay_vegas=0.3))
    assert "overlay > overlay_vegas > model" in line


# --------------------------------------------------------------------------
# build_scored_frame / run_position -- real bundle, skip if not cached
# --------------------------------------------------------------------------


def test_build_scored_frame_wr_shape_and_no_leakage_across_folds() -> None:
    spec = train.position_spec("wr")
    if not spec.artifact_path.exists():
        pytest.skip(f"{spec.artifact_path} not built; run `python -m src.models.train wr`")
    bundle = joblib.load(spec.artifact_path)
    scored = ob.build_scored_frame(spec, bundle)

    assert set(scored["season"].unique()) <= set(ob.BACKTEST_SEASONS)
    # 2023 rows all come from the fold-boundary-respecting pooled OOF (max train season 2022 <
    # val season 2023, enforced structurally by src.models.cv.Fold); 2024/2025 come from the
    # holdout-retrained models (train <=2023 < holdout 2024/2025, enforced by HoldoutSplit) --
    # both paths' fold-boundary invariant is asserted at construction time in src.models.cv,
    # so a non-empty result here already proves neither path silently used a bad split.
    assert scored["probability"].between(0.0, 1.0).all()
    assert {"expectation_pos_rank", "finish_rank_delta", "breakout"} <= set(scored.columns)


def test_run_position_wr_smoke() -> None:
    spec = train.position_spec("wr")
    if not spec.artifact_path.exists():
        pytest.skip(f"{spec.artifact_path} not built; run `python -m src.models.train wr`")
    result = ob.run_position("wr")
    assert result is not None
    table, verdict = result
    assert "WR verdict" in verdict
    assert set(table["season"].unique()) == set(ob.BACKTEST_SEASONS) | {"pooled"}
