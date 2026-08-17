"""Tests for the v1.5 (Phase C) pbp-derived red-zone/goal-line share features:

WR/TE's ``rz_target_share``/``ez_target_share`` and RB's
``rz_carry_share``/``goal_line_carry_share`` (``src.features.shared.redzone_share_table``,
wired into ``src.features.{wr,rb,te}.build_raw_stat_table``). Same two guarantees every
other feature in this pipeline gets, per the brief: a structural leakage test (pbp-derived
features go through the identical strip-season-N mechanism
``tests/test_features_positions.py`` uses for player_stats) and a nullable-by-design check
(pbp absent -> null columns, never a raised error). Build-dependent tests read the
gitignored raw/processed parquet files and skip when absent, matching every other test
file's pattern in this repo.
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl
import pytest

from src.features import shared as sh
from src.features.rb import build_features_rb
from src.features.te import build_features_te
from src.features.wr import build_features_wr
from src.labels.build import OUT_PATH as LABELS_PATH

PBP_PATH = sh.RAW_DIR / "pbp.parquet"

_REQUIRED_PATHS = [
    LABELS_PATH,
    sh.PATHS["player_stats"],
    sh.PATHS["rosters"],
    sh.PATHS["rosters_weekly"],
    sh.PATHS["draft_picks"],
    sh.PATHS["schedules"],
]

_BUILDERS = {
    "wr": {"fn": build_features_wr, "cols": ["rz_target_share_n1", "ez_target_share_n1"]},
    "rb": {"fn": build_features_rb, "cols": ["rz_carry_share_n1", "goal_line_carry_share_n1"]},
    "te": {"fn": build_features_te, "cols": ["rz_target_share_n1", "ez_target_share_n1"]},
}


def _raw_available() -> bool:
    return all(p.exists() for p in _REQUIRED_PATHS)


def _pbp_available() -> bool:
    return PBP_PATH.exists()


def _skip_if_raw_missing() -> None:
    if not _raw_available():
        pytest.skip("raw data not cached; run `python -m src.ingest.nflverse` then `python -m src.labels.build`")


def _skip_if_pbp_missing() -> None:
    if not _pbp_available():
        pytest.skip("data/raw/pbp.parquet not cached; run `python -m src.ingest.nflverse --only pbp`")


@lru_cache(maxsize=1)
def _default_frames() -> dict[str, pl.DataFrame | None]:
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
    }


def _build(pos: str, **overrides) -> pl.DataFrame:
    frames = dict(_default_frames())
    kwargs = {
        "labels": frames["labels"],
        "player_stats": frames["player_stats"],
        "ff_opportunity": frames["ff_opportunity"],
        "rosters": frames["rosters"],
        "rosters_weekly": frames["rosters_weekly"],
        "draft_picks": frames["draft_picks"],
        "schedules": frames["schedules"],
        "snap_counts": frames["snap_counts"],
        "coaching_changes": frames["coaching_changes"],
        "pbp": frames["pbp"],
    }
    kwargs.update(overrides)
    return _BUILDERS[pos]["fn"](**kwargs)


def _assert_frames_close(a: pl.DataFrame, b: pl.DataFrame, *, context: str) -> None:
    assert a.columns == b.columns, f"{context}: column sets differ"
    assert a.height == b.height, f"{context}: row counts differ ({a.height} vs {b.height})"
    for col in a.columns:
        ca, cb = a.get_column(col), b.get_column(col)
        if ca.dtype in (pl.Float32, pl.Float64):
            diff = (ca - cb).abs()
            both_null = ca.is_null() & cb.is_null()
            bad = (~both_null) & ((diff > 1e-6) | (ca.is_null() != cb.is_null()))
            assert bad.sum() == 0, f"{context}: column {col!r} differs beyond float tolerance ({bad.sum()} rows)"
        else:
            assert ca.equals(cb), f"{context}: column {col!r} differs"


# --------------------------------------------------------------------------
# Structural leakage test: strip season-2023 pbp rows, rebuild season-2023
# feature rows, assert byte-identical to the full build. Season-2023 label
# rows read season-2022 pbp (the N-1 shift) -- they can only be identical
# if nothing in them was ever computed from season-2023 pbp rows.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", sorted(_BUILDERS))
def test_pbp_leakage_structural_2023_stripped_matches_full_build(pos: str) -> None:
    _skip_if_raw_missing()
    _skip_if_pbp_missing()

    full = _build(pos).filter(pl.col("season") == 2023).sort(["season", "gsis_id"])
    if full.height == 0:
        pytest.skip(f"no season-2023 {pos.upper()} rows to check")

    stripped_pbp = _default_frames()["pbp"].filter(pl.col("season") != 2023)
    stripped = _build(pos, pbp=stripped_pbp).filter(pl.col("season") == 2023).sort(["season", "gsis_id"])

    _assert_frames_close(
        full, stripped, context=f"{pos}: season-2023 feature rows changed when 2023 pbp rows were removed"
    )


# --------------------------------------------------------------------------
# Nullable by design: pbp=None -> every rz/ez/goal-line column is entirely
# null, never a raised error (same optional-source contract as
# snap_counts/ngs_*).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", sorted(_BUILDERS))
def test_pbp_absent_yields_null_columns_not_an_error(pos: str) -> None:
    _skip_if_raw_missing()
    out = _build(pos, pbp=None)
    for col in _BUILDERS[pos]["cols"]:
        assert out.get_column(col).null_count() == out.height, f"{pos}: {col} should be entirely null when pbp=None"


# --------------------------------------------------------------------------
# With real pbp present: columns are not entirely null, and every non-null
# value is a valid share in [0, 1].
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", sorted(_BUILDERS))
def test_pbp_present_shares_are_valid_and_not_all_null(pos: str) -> None:
    _skip_if_raw_missing()
    _skip_if_pbp_missing()
    out = _build(pos)
    for col in _BUILDERS[pos]["cols"]:
        series = out.get_column(col)
        assert series.null_count() < out.height, f"{pos}: {col} is entirely null with pbp present"
        non_null = series.drop_nulls()
        assert (non_null >= 0).all() and (non_null <= 1).all(), f"{pos}: {col} has values outside [0, 1]"


# --------------------------------------------------------------------------
# redzone_share_table itself: direct unit tests against a small synthetic
# pbp frame (no dependency on cached data).
# --------------------------------------------------------------------------


@pytest.fixture()
def synthetic_pbp() -> pl.DataFrame:
    # Team "A", season 2023, REG: two receivers, three qualifying (rz) targets total
    # for the team at threshold 20 (WR1 gets 2, WR2 gets 1), one of WR1's two is also
    # inside the tighter threshold 10 (goal-line-adjacent).
    return pl.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023, 2023],
            "season_type": ["REG"] * 5,
            "posteam": ["A", "A", "A", "A", "A"],
            "yardline_100": [15, 8, 18, 25, 5],
            "pass_attempt": [1, 1, 1, 1, 0],
            "rush_attempt": [0, 0, 0, 0, 1],
            "receiver_player_id": ["wr1", "wr1", "wr2", "wr3", None],
            "rusher_player_id": [None, None, None, None, "rb1"],
            "complete_pass": [1, 0, 1, 1, 0],
            "touchdown": [0, 0, 0, 0, 0],
            "week": [1, 1, 1, 1, 1],
        }
    )


@pytest.fixture()
def synthetic_team_assign() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "gsis_id": ["wr1", "wr2", "wr3", "rb1"],
            "primary_team": ["A", "A", "A", "A"],
            "games": [16, 16, 16, 16],
        }
    )


def test_redzone_share_table_target_shares(synthetic_pbp, synthetic_team_assign) -> None:
    out = sh.redzone_share_table(
        synthetic_pbp,
        synthetic_team_assign,
        play_col="pass_attempt",
        player_col="receiver_player_id",
        thresholds={"rz_target_share": 20, "ez_target_share": 10},
    )
    row = out.filter(pl.col("gsis_id") == "wr1").row(0, named=True)
    # WR1: 2 of the 3 rz (<=20) team targets -> 2/3
    assert row["rz_target_share"] == pytest.approx(2 / 3)
    # WR1: 1 of the 1 ez (<=10) team target -> 1/1 (the 25-yardline target is excluded from this threshold entirely)
    assert row["ez_target_share"] == pytest.approx(1.0)

    wr3 = out.filter(pl.col("gsis_id") == "wr3").row(0, named=True)
    # WR3's only target is at yardline 25 -- outside both thresholds -> explicit 0.0, not null
    # (the team itself had qualifying rz/ez targets that season, so 0/team_n is a real zero).
    assert wr3["rz_target_share"] == pytest.approx(0.0)
    assert wr3["ez_target_share"] == pytest.approx(0.0)


def test_redzone_share_table_carry_shares(synthetic_pbp, synthetic_team_assign) -> None:
    out = sh.redzone_share_table(
        synthetic_pbp,
        synthetic_team_assign,
        play_col="rush_attempt",
        player_col="rusher_player_id",
        thresholds={"rz_carry_share": 20, "goal_line_carry_share": 5},
    )
    row = out.filter(pl.col("gsis_id") == "rb1").row(0, named=True)
    assert row["rz_carry_share"] == pytest.approx(1.0)  # RB1's only carry, at yardline 5, is inside <=20
    assert row["goal_line_carry_share"] == pytest.approx(1.0)  # also inside <=5


def test_redzone_share_table_excludes_non_reg_and_null_ids() -> None:
    pbp = pl.DataFrame(
        {
            "season": [2023, 2023],
            "season_type": ["POST", "REG"],
            "posteam": ["A", "A"],
            "yardline_100": [10, 10],
            "pass_attempt": [1, 1],
            "receiver_player_id": ["wr1", None],
        }
    )
    team_assign = pl.DataFrame({"season": [2023], "gsis_id": ["wr1"], "primary_team": ["A"], "games": [16]})
    out = sh.redzone_share_table(pbp, team_assign, play_col="pass_attempt", player_col="receiver_player_id", thresholds={"rz": 20})
    # Neither row qualifies (POST excluded, REG row has a null receiver) -- team never appears
    # in team_n, so the share is null (0/0), not a fabricated 0.
    row = out.filter(pl.col("gsis_id") == "wr1").row(0, named=True)
    assert row["rz"] is None


def test_empty_redzone_share_table_shape() -> None:
    out = sh.empty_redzone_share_table(["rz_target_share", "ez_target_share"])
    assert out.height == 0
    assert set(out.columns) == {"season", "gsis_id", "rz_target_share", "ez_target_share"}
