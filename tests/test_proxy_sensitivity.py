"""Tests for src/models/proxy_sensitivity.py (v1.7 Step 1 proxy-era sensitivity experiment).

Data-dependent (reads the gitignored ``data/processed/features_{pos}/labels.parquet``
and the shipped ``data/models/{pos}_model_bundle.joblib``) and skips cleanly when absent,
matching every other pipeline test file's pattern in this repo.
"""

from __future__ import annotations

import joblib
import pytest

from src.models import proxy_sensitivity as ps
from src.models import train

_REQUIRED_PATHS = [train.position_spec(p).features_path for p in train.POSITIONS] + [
    train.position_spec(p).labels_path for p in train.POSITIONS
] + [train.position_spec(p).artifact_path for p in train.POSITIONS]


def _skip_if_raw_missing() -> None:
    if not all(p.exists() for p in _REQUIRED_PATHS):
        pytest.skip("features/labels parquet or model bundles not built")


def test_rb_bundle_cfg_restricts_train_season_start() -> None:
    """Sanity check on the bug this finding fixes: RB's shipped bundle cfg carries its own
    v1.7 `train_season_start: 2020` restriction, which is exactly why Arm A's frame load
    must override it back to the global 2014 start rather than reusing the raw bundle cfg.
    """
    _skip_if_raw_missing()
    bundle = joblib.load(train.position_spec("rb").artifact_path)
    assert bundle["cfg"].get("train_season_start", train.TRAIN_START_SEASON) >= 2020


def test_arm_a_rb_frame_contains_pre_2020_seasons() -> None:
    """Finding 1: Arm A must force train_season_start back to the global 2014 start when
    loading, even for a position (RB) whose own shipped bundle cfg restricts to 2020+.
    Regression guard for the bug where Arm A silently collapsed onto Arm B's pool.
    """
    _skip_if_raw_missing()
    spec = train.position_spec("rb")
    bundle = joblib.load(spec.artifact_path)

    pool_cfg = dict(bundle["cfg"])
    pool_cfg["train_season_start"] = ps.ARM_TRAIN_START["A"]
    df_full = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=pool_cfg)
    df_arm_a = df_full[df_full["season"] >= ps.ARM_TRAIN_START["A"]].reset_index(drop=True)

    assert (df_arm_a["season"] < 2020).any(), (
        "Arm A's loaded RB frame contains no seasons before 2020 -- the bundle cfg's own "
        "train_season_start restriction leaked through and Arm A collapsed onto Arm B's pool"
    )


def test_arm_a_and_arm_b_rb_pools_differ_in_size() -> None:
    _skip_if_raw_missing()
    spec = train.position_spec("rb")
    bundle = joblib.load(spec.artifact_path)

    sizes = {}
    for arm, train_start in ps.ARM_TRAIN_START.items():
        pool_cfg = dict(bundle["cfg"])
        pool_cfg["train_season_start"] = train_start
        df_full = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=pool_cfg)
        df_arm = df_full[df_full["season"] >= train_start].reset_index(drop=True)
        sizes[arm] = len(df_arm)

    assert sizes["A"] > sizes["B"], f"Arm A ({sizes['A']}) should be a strict superset in size of Arm B ({sizes['B']})"
