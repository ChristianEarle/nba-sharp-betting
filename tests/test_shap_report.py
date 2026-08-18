"""Tests for src/explain/shap_report.py (Finding 6: pooled_validation_frame must derive

validation seasons from the position's ACTUAL folds, not the fixed VALIDATION_SEASONS
constant). Data-dependent (reads the gitignored features/labels parquet + shipped model
bundles) and skips cleanly when absent, matching every other pipeline test file's pattern
in this repo.
"""

from __future__ import annotations

import joblib
import pytest

from src.explain import shap_report as shp
from src.models import cv
from src.models import train


def _skip_if_missing(pos: str) -> None:
    spec = train.position_spec(pos)
    if not (spec.artifact_path.exists() and spec.features_path.exists() and spec.labels_path.exists()):
        pytest.skip(f"{pos}: bundle/features/labels not built")


def test_validation_seasons_for_rb_excludes_training_only_years() -> None:
    """RB's bundle cfg restricts train_season_start to 2020+ (v1.7 real-market-only
    decision), so its validation_folds only reach val_season in {2022, 2023} -- 2020 and
    2021 are TRAINING-only years for RB, not validation years. The fixed VALIDATION_SEASONS
    constant (2020, 2021, 2022, 2023) would wrongly include them.
    """
    _skip_if_missing("rb")
    bundle = joblib.load(train.position_spec("rb").artifact_path)
    train_start = bundle["cfg"].get("train_season_start", cv.TRAIN_START_SEASON)
    if train_start <= cv.TRAIN_START_SEASON:
        pytest.skip("rb bundle cfg no longer restricts train_season_start; this regression guard no longer applies")

    seasons = shp.validation_seasons_for(bundle)
    assert 2020 not in seasons, "2020 is a training-only year for a 2020+-restricted position, not a validation year"
    assert 2021 not in seasons, "2021 is a training-only year for a 2020+-restricted position, not a validation year"
    assert set(seasons) == {2022, 2023}


def test_validation_seasons_for_matches_fixed_constant_at_default_start() -> None:
    """A position with no train_season_start restriction (the default 2014 start) must
    still see the full four VALIDATION_SEASONS -- this fix changes nothing for a normal
    (unrestricted) position.
    """
    _skip_if_missing("wr")
    bundle = joblib.load(train.position_spec("wr").artifact_path)
    if bundle["cfg"].get("train_season_start", cv.TRAIN_START_SEASON) > cv.TRAIN_START_SEASON:
        pytest.skip("wr bundle cfg restricts train_season_start; this check targets the unrestricted case")
    assert shp.validation_seasons_for(bundle) == list(cv.VALIDATION_SEASONS)


def test_pooled_validation_frame_rb_excludes_training_only_seasons() -> None:
    _skip_if_missing("rb")
    bundle = joblib.load(train.position_spec("rb").artifact_path)
    train_start = bundle["cfg"].get("train_season_start", cv.TRAIN_START_SEASON)
    if train_start <= cv.TRAIN_START_SEASON:
        pytest.skip("rb bundle cfg no longer restricts train_season_start")

    val_df = shp.pooled_validation_frame("rb", bundle)
    assert not val_df.empty
    assert set(val_df["season"].unique()) <= {2022, 2023}
    assert 2020 not in val_df["season"].unique()
    assert 2021 not in val_df["season"].unique()


def test_pooled_validation_frame_never_includes_holdout_or_2026() -> None:
    _skip_if_missing("wr")
    bundle = joblib.load(train.position_spec("wr").artifact_path)
    val_df = shp.pooled_validation_frame("wr", bundle)
    assert set(val_df["season"].unique()).isdisjoint(set(cv.HOLDOUT_SEASONS))
    assert 2026 not in val_df["season"].unique()
