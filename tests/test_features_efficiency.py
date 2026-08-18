"""Tests for the v2.2 efficiency-proxy features (YPRR substitutes):

``src.features.shared.season_offense_snaps_table`` / ``per_snap_rate_table`` /
``ftn_catchable_target_rate_table``, wired into ``src.features.{wr,rb,te}.BASE_METRICS``
as ``yards_per_snap``/``targets_per_snap``/``catchable_target_rate``. Pure-logic tests run
against small synthetic frames (no cached data needed); build-dependent tests read the
gitignored raw/processed parquet files and skip cleanly when absent, matching every other
test file's pattern in this repo.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.features import shared as sh
from src.features.rb import build_features_rb
from src.features.te import build_features_te
from src.features.wr import build_features_wr
from src.labels.build import OUT_PATH as LABELS_PATH

PBP_PATH = sh.RAW_DIR / "pbp.parquet"
FTN_PATH = sh.RAW_DIR / "ftn_charting.parquet"

_REQUIRED_PATHS = [
    LABELS_PATH,
    sh.PATHS["player_stats"],
    sh.PATHS["rosters"],
    sh.PATHS["rosters_weekly"],
    sh.PATHS["draft_picks"],
    sh.PATHS["schedules"],
]

_BUILDERS = {
    "wr": {"fn": build_features_wr, "cols": ["yards_per_snap_n1", "targets_per_snap_n1", "catchable_target_rate_n1"]},
    "rb": {"fn": build_features_rb, "cols": ["yards_per_snap_n1", "catchable_target_rate_n1"]},
    "te": {"fn": build_features_te, "cols": ["yards_per_snap_n1", "targets_per_snap_n1", "catchable_target_rate_n1"]},
}


def _skip_if_raw_missing() -> None:
    if not all(p.exists() for p in _REQUIRED_PATHS):
        pytest.skip("raw data not cached; run `python -m src.ingest.nflverse` then `python -m src.labels.build`")


# --------------------------------------------------------------------------
# season_offense_snaps_table / per_snap_rate_table -- synthetic, no cached data
# --------------------------------------------------------------------------


@pytest.fixture()
def synthetic_snap_counts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_type": ["REG", "REG", "REG", "POST"],
            "season": [2023, 2023, 2023, 2023],
            "pfr_player_id": ["p1", "p1", "p2", "p1"],
            "offense_snaps": [50.0, 60.0, 10.0, 999.0],  # POST row must never count
            "offense_pct": [0.8, 0.9, 0.2, 1.0],
        }
    )


@pytest.fixture()
def synthetic_rosters() -> pl.DataFrame:
    return pl.DataFrame({"season": [2023, 2023], "pfr_id": ["p1", "p2"], "gsis_id": ["g1", "g2"]})


def test_season_offense_snaps_table_sums_reg_only(synthetic_snap_counts, synthetic_rosters) -> None:
    out = sh.season_offense_snaps_table(synthetic_snap_counts, synthetic_rosters)
    row = out.filter(pl.col("gsis_id") == "g1").row(0, named=True)
    assert row["offense_snaps"] == pytest.approx(110.0)  # 50 + 60, POST row excluded
    row2 = out.filter(pl.col("gsis_id") == "g2").row(0, named=True)
    assert row2["offense_snaps"] == pytest.approx(10.0)


def test_season_offense_snaps_table_drops_unresolved_pfr_id(synthetic_rosters) -> None:
    snap_counts = pl.DataFrame(
        {"game_type": ["REG"], "season": [2023], "pfr_player_id": ["unknown"], "offense_snaps": [40.0], "offense_pct": [0.5]}
    )
    out = sh.season_offense_snaps_table(snap_counts, synthetic_rosters)
    assert out.height == 0


def test_per_snap_rate_table_divides_by_season_snaps(synthetic_snap_counts, synthetic_rosters) -> None:
    season_totals = pl.DataFrame({"season": [2023, 2023], "gsis_id": ["g1", "g2"], "rec_yards": [220.0, 5.0]})
    out = sh.per_snap_rate_table(
        season_totals, synthetic_snap_counts, synthetic_rosters, numerator_col="rec_yards", out_col="yards_per_snap"
    )
    row = out.filter(pl.col("gsis_id") == "g1").row(0, named=True)
    assert row["yards_per_snap"] == pytest.approx(220.0 / 110.0)


def test_per_snap_rate_table_null_when_snaps_unresolved(synthetic_snap_counts, synthetic_rosters) -> None:
    season_totals = pl.DataFrame({"season": [2023], "gsis_id": ["g_unresolved"], "rec_yards": [100.0]})
    out = sh.per_snap_rate_table(
        season_totals, synthetic_snap_counts, synthetic_rosters, numerator_col="rec_yards", out_col="yards_per_snap"
    )
    row = out.filter(pl.col("gsis_id") == "g_unresolved").row(0, named=True)
    assert row["yards_per_snap"] is None


# --------------------------------------------------------------------------
# ftn_catchable_target_rate_table -- synthetic, no cached data
# --------------------------------------------------------------------------


@pytest.fixture()
def synthetic_ftn() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2023, 2023, 2023, 2021],
            "nflverse_game_id": ["2023_01_A_B", "2023_01_A_B", "2023_01_A_B", "2021_01_A_B"],
            "nflverse_play_id": [10, 20, 30, 10],
            "is_catchable_ball": [True, False, True, True],
        }
    )


@pytest.fixture()
def synthetic_pbp_full() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2023_01_A_B", "2023_01_A_B", "2023_01_A_B", "2021_01_A_B"],
            "play_id": [10.0, 20.0, 30.0, 10.0],
            "receiver_player_id": ["wr1", "wr1", None, "wr1"],
        }
    )


def test_ftn_catchable_target_rate_table_computes_share(synthetic_ftn, synthetic_pbp_full) -> None:
    out = sh.ftn_catchable_target_rate_table(synthetic_ftn, synthetic_pbp_full)
    # wr1's charted targets in 2023: play 10 (catchable) and play 20 (not) -- play 30 has
    # no receiver_player_id and must be excluded. 1 of 2 -> 0.5.
    row = out.filter((pl.col("season") == 2023) & (pl.col("gsis_id") == "wr1")).row(0, named=True)
    assert row["catchable_target_rate"] == pytest.approx(0.5)


def test_ftn_catchable_target_rate_table_excludes_unreceipted_plays(synthetic_ftn, synthetic_pbp_full) -> None:
    out = sh.ftn_catchable_target_rate_table(synthetic_ftn, synthetic_pbp_full)
    assert out.filter(pl.col("gsis_id").is_null()).height == 0


def test_ftn_catchable_target_rate_table_join_is_scoped_by_game_and_play(synthetic_ftn, synthetic_pbp_full) -> None:
    out = sh.ftn_catchable_target_rate_table(synthetic_ftn, synthetic_pbp_full)
    # 2021 row also resolves (join isn't season-scoped, just game+play id) -- catchable_target_rate == 1.0
    row = out.filter((pl.col("season") == 2021) & (pl.col("gsis_id") == "wr1")).row(0, named=True)
    assert row["catchable_target_rate"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Wired into WR/RB/TE feature tables -- real cached data (skips if absent)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _frames() -> dict:
    _skip_if_raw_missing()
    ff_opportunity_path = sh.RAW_DIR / "ff_opportunity.parquet"
    return {
        "labels": pl.read_parquet(LABELS_PATH),
        "player_stats": pl.read_parquet(sh.PATHS["player_stats"]),
        "ff_opportunity": pl.read_parquet(ff_opportunity_path),
        "rosters": pl.read_parquet(sh.PATHS["rosters"]),
        "rosters_weekly": pl.read_parquet(sh.PATHS["rosters_weekly"]),
        "draft_picks": pl.read_parquet(sh.PATHS["draft_picks"]),
        "schedules": pl.read_parquet(sh.PATHS["schedules"]),
        "snap_counts": pl.read_parquet(sh.PATHS["snap_counts"]) if sh.PATHS["snap_counts"].exists() else None,
        "coaching_changes": sh.load_coaching_changes(),
        "pbp": pl.read_parquet(PBP_PATH) if PBP_PATH.exists() else None,
        "ftn_charting": pl.read_parquet(FTN_PATH) if FTN_PATH.exists() else None,
    }


def _build(pos: str, frames: dict, **overrides) -> pl.DataFrame:
    kwargs = {
        "labels": frames["labels"], "player_stats": frames["player_stats"], "ff_opportunity": frames["ff_opportunity"],
        "rosters": frames["rosters"], "rosters_weekly": frames["rosters_weekly"], "draft_picks": frames["draft_picks"],
        "schedules": frames["schedules"], "snap_counts": frames["snap_counts"], "coaching_changes": frames["coaching_changes"],
        "pbp": frames["pbp"], "ftn_charting": frames["ftn_charting"],
    }
    kwargs.update(overrides)
    return _BUILDERS[pos]["fn"](**kwargs)


@pytest.mark.parametrize("pos", sorted(_BUILDERS))
def test_efficiency_columns_not_entirely_null_and_in_range(pos: str, _frames) -> None:
    if _frames["snap_counts"] is None or _frames["pbp"] is None or _frames["ftn_charting"] is None:
        pytest.skip("snap_counts/pbp/ftn_charting not cached")
    out = _build(pos, _frames)
    for col in _BUILDERS[pos]["cols"]:
        series = out.get_column(col)
        assert series.null_count() < out.height, f"{pos}: {col} is entirely null with all sources present"
        if "catchable_target_rate" in col:
            non_null = series.drop_nulls()
            assert (non_null >= 0).all() and (non_null <= 1).all(), f"{pos}: {col} outside [0, 1]"


@pytest.mark.parametrize("pos", sorted(_BUILDERS))
def test_efficiency_columns_null_when_snap_counts_absent(pos: str, _frames) -> None:
    _skip_if_raw_missing()
    out = _build(pos, _frames, snap_counts=None)
    for col in _BUILDERS[pos]["cols"]:
        if "catchable_target_rate" in col:
            continue
        assert out.get_column(col).null_count() == out.height, f"{pos}: {col} should be entirely null with snap_counts=None"


@pytest.mark.parametrize("pos", sorted(_BUILDERS))
def test_catchable_target_rate_null_when_ftn_absent(pos: str, _frames) -> None:
    _skip_if_raw_missing()
    out = _build(pos, _frames, ftn_charting=None)
    col = "catchable_target_rate_n1"
    assert out.get_column(col).null_count() == out.height, f"{pos}: {col} should be entirely null with ftn_charting=None"


@pytest.mark.parametrize("pos", sorted(_BUILDERS))
def test_catchable_target_rate_entirely_null_pre_2022_n1(pos: str, _frames) -> None:
    """catchable_target_rate_n1 is the player's season-(N-1) rate; FTN starts 2022, so no

    row with season <= 2022 (N-1 = 2021 or earlier) can have a non-null value.
    """
    if _frames["snap_counts"] is None or _frames["pbp"] is None or _frames["ftn_charting"] is None:
        pytest.skip("snap_counts/pbp/ftn_charting not cached")
    out = _build(pos, _frames)
    pre = out.filter(pl.col("season") <= 2022)
    if pre.height == 0:
        pytest.skip(f"no season<=2022 {pos.upper()} rows to check")
    assert pre.get_column("catchable_target_rate_n1").null_count() == pre.height
