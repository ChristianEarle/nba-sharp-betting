"""Tests for the Vegas team-feature wiring added to src/features/shared.py (v1.5 Phase C,
Deliverable 3): vegas_team_features / attach_vegas_team, plus one end-to-end check that
build_features_wr actually carries the three columns through to its output.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.features import shared as sh


def _vegas_team_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2024, 2024],
            "team": ["KC", "BUF"],
            "implied_ppg": [26.5, 22.0],
            "implied_win_prob": [0.62, 0.55],
            "n_events_priced": [17, 17],
            "has_vegas": [1, 1],
        }
    )


# --------------------------------------------------------------------------
# vegas_team_features
# --------------------------------------------------------------------------


def test_vegas_team_features_none_yields_empty_typed_schema() -> None:
    out = sh.vegas_team_features(None)
    assert out.height == 0
    assert set(out.columns) == {"season", "team", "implied_ppg", "implied_win_prob", "has_vegas"}


def test_vegas_team_features_empty_frame_yields_empty_typed_schema() -> None:
    out = sh.vegas_team_features(pl.DataFrame())
    assert out.height == 0
    assert set(out.columns) == {"season", "team", "implied_ppg", "implied_win_prob", "has_vegas"}


def test_vegas_team_features_drops_n_events_priced() -> None:
    out = sh.vegas_team_features(_vegas_team_frame())
    assert "n_events_priced" not in out.columns
    assert out.height == 2


# --------------------------------------------------------------------------
# attach_vegas_team
# --------------------------------------------------------------------------


def _players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2024, 2024, 2019],
            "gsis_id": ["p1", "p2", "p3"],
            "team": ["KC", "ZZZ", "KC"],  # ZZZ unmatched; p3 is pre-2020 (no Vegas data at all)
        }
    )


def test_attach_vegas_team_matches_and_fills_has_vegas_zero() -> None:
    out = sh.attach_vegas_team(_players(), _vegas_team_frame())
    p1 = out.filter(pl.col("gsis_id") == "p1").row(0, named=True)
    p2 = out.filter(pl.col("gsis_id") == "p2").row(0, named=True)
    p3 = out.filter(pl.col("gsis_id") == "p3").row(0, named=True)

    assert p1["implied_ppg"] == pytest.approx(26.5)
    assert p1["has_vegas"] == 1

    assert p2["implied_ppg"] is None
    assert p2["has_vegas"] == 0  # never null -- see docstring

    assert p3["implied_ppg"] is None
    assert p3["has_vegas"] == 0


def test_attach_vegas_team_none_source_fills_has_vegas_zero_everywhere() -> None:
    out = sh.attach_vegas_team(_players(), None)
    assert (out.get_column("has_vegas") == 0).all()
    assert out.get_column("implied_ppg").null_count() == out.height


def test_attach_vegas_team_has_vegas_never_null() -> None:
    out = sh.attach_vegas_team(_players(), _vegas_team_frame())
    assert out.get_column("has_vegas").null_count() == 0


# --------------------------------------------------------------------------
# End-to-end: build_features_wr carries the columns through (data-dependent)
# --------------------------------------------------------------------------


def test_build_features_wr_carries_vegas_columns_if_data_cached() -> None:
    features_path = sh.REPO_ROOT / "data" / "processed" / "features_wr.parquet"
    if not features_path.exists():
        pytest.skip("features_wr.parquet not built; run `python -m src.features.wr`")
    out = pl.read_parquet(features_path)
    assert {"implied_ppg", "implied_win_prob", "has_vegas"} <= set(out.columns)
    # every pre-2020 row must show has_vegas=0 (no team_lines coverage before 2020).
    pre2020 = out.filter(pl.col("season") < 2020)
    if pre2020.height:
        assert (pre2020.get_column("has_vegas") == 0).all()
