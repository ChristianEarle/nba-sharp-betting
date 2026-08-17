"""Schema and coverage guards for the Phase 1a nflverse cache.

The raw cache is gitignored, so these skip when it is absent (fresh clone /
CI) and run whenever ``python -m src.ingest.nflverse`` has been executed.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.ingest.nflverse import RAW_DIR, load_config, season_range

CFG = load_config()
SEASONS = season_range(CFG)

# Datasets the pipeline cannot proceed without, and the columns downstream
# phases key on. Guards against an upstream schema rename landing silently.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "player_stats": [
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "season_type",
        "fantasy_points_ppr",
        "targets",
        "receptions",
        "receiving_yards",
        "rushing_yards",
        "passing_yards",
    ],
    "snap_counts": ["season", "week", "player", "position"],
    "rosters": ["season", "gsis_id", "position"],
    "draft_picks": ["season", "round", "pick", "gsis_id"],
    "ff_opportunity": ["season", "week", "player_id"],
    "ff_playerids": ["gsis_id", "fantasypros_id", "sleeper_id", "name", "position"],
}


SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
OVERRIDES_PATH = RAW_DIR.parents[1] / "configs" / "id_overrides.csv"


def _override_mfl_ids() -> list[str]:
    """mfl_ids with a documented manual correction; excluded from duplicate checks."""
    if not OVERRIDES_PATH.exists():
        return []
    df = pl.read_csv(OVERRIDES_PATH, comment_prefix="#", infer_schema_length=0)
    if "mfl_id" not in df.columns:
        return []
    return df.get_column("mfl_id").drop_nulls().to_list()


def _path(name: str):
    return RAW_DIR / f"{name}.parquet"


def _read(name: str) -> pl.DataFrame:
    p = _path(name)
    if not p.exists():
        pytest.skip(f"{name}.parquet not cached; run `python -m src.ingest.nflverse`")
    return pl.read_parquet(p)


@pytest.mark.parametrize("name", sorted(CFG["datasets"]))
def test_required_datasets_cached_and_nonempty(name: str) -> None:
    spec = CFG["datasets"][name]
    if not spec.get("required", False):
        pytest.skip(f"{name} is optional")
    df = _read(name)
    assert df.height > 0, f"{name} cached but empty"


@pytest.mark.parametrize("name,cols", sorted(REQUIRED_COLUMNS.items()))
def test_expected_columns_present(name: str, cols: list[str]) -> None:
    df = _read(name)
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"{name} missing expected columns: {missing}"


def test_player_stats_covers_all_configured_seasons() -> None:
    df = _read("player_stats")
    present = set(df.get_column("season").unique().to_list())
    missing = [s for s in SEASONS if s not in present]
    assert not missing, f"player_stats missing seasons: {missing}"


def test_player_stats_has_regular_and_post_season_split() -> None:
    """Phase 2 finish ranks must be REG-only.

    Weekly stats ship regular and postseason rows together, so a label built
    without filtering silently credits playoff production. Nacua's 2023 reads
    18 games unfiltered versus 17 REG. Lock the column that makes the filter
    possible.
    """
    df = _read("player_stats")
    kinds = set(df.get_column("season_type").unique().to_list())
    assert "REG" in kinds, "no REG rows; finish ranks cannot be computed"
    assert "POST" in kinds, "no POST rows; expected split not present"


def test_regular_season_weeks_are_bounded() -> None:
    df = _read("player_stats").filter(pl.col("season_type") == "REG")
    assert df.get_column("week").max() <= 18, "REG week exceeds 18"
    # 17-game era begins 2021; before that a REG season is 17 weeks.
    pre = df.filter(pl.col("season") < 2021).get_column("week").max()
    assert pre <= 17, f"pre-2021 REG week max is {pre}, expected <= 17"


def test_id_map_gsis_ids_are_unique_among_skill_positions() -> None:
    """A duplicated gsis_id in the crosswalk silently fans out joins.

    The upstream dynastyprocess map assigns one gsis_id to two players in ~10
    same-name cases (a CB and an RB both named Tony Carter, etc). All but one
    pair sit outside QB/RB/WR/TE and cannot reach this model. The survivor is
    documented in configs/id_overrides.csv; anything beyond that is new
    upstream breakage and should fail here.
    """
    df = _read("ff_playerids").filter(
        pl.col("gsis_id").is_not_null() & pl.col("position").is_in(SKILL_POSITIONS)
    )
    known = _override_mfl_ids()
    if "mfl_id" in df.columns and known:
        df = df.filter(~pl.col("mfl_id").is_in(known))

    dupes = df.group_by("gsis_id").len().filter(pl.col("len") > 1).get_column("gsis_id").to_list()
    assert not dupes, (
        f"undocumented duplicate gsis_id among skill positions: {dupes[:10]}. "
        "Verify against nflverse and add to configs/id_overrides.csv."
    )


def test_known_breakout_season_is_sane() -> None:
    """Named-player canary: catches a broken stats join before it reaches labels."""
    df = _read("player_stats").filter(
        (pl.col("player_display_name") == "Puka Nacua")
        & (pl.col("season") == 2023)
        & (pl.col("season_type") == "REG")
    )
    assert df.height == 17, f"expected 17 REG games, got {df.height}"
    ppg = df.get_column("fantasy_points_ppr").sum() / df.height
    assert 15.0 < ppg < 20.0, f"Nacua 2023 PPR ppg {ppg:.1f} outside sane range"
