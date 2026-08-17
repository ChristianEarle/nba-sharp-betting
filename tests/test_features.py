"""Tests for the Phase 3 WR feature builders (src/features/shared.py, src/features/wr.py).

Build-dependent tests read the gitignored raw/processed parquet files and
skip when they're absent (fresh clone / CI), matching the pattern in
test_labels.py. Run, in order: ``python -m src.ingest.nflverse``,
``python -m src.labels.build``, ``python -m src.features.wr``.

The leakage test is the important one here — see its docstring below for
the two independent mechanisms it uses.
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl
import pytest

from src.features import shared as sh
from src.features.wr import (
    NGS_RECEIVING_PATH,
    OUT_PATH,
    build_features_wr,
)
from src.labels.build import OUT_PATH as LABELS_PATH

_REQUIRED_PATHS = [
    LABELS_PATH,
    sh.PATHS["player_stats"],
    sh.PATHS["rosters"],
    sh.PATHS["rosters_weekly"],
    sh.PATHS["draft_picks"],
    sh.PATHS["schedules"],
]


def _raw_available() -> bool:
    return all(p.exists() for p in _REQUIRED_PATHS)


def _skip_if_raw_missing() -> None:
    if not _raw_available():
        pytest.skip(
            "raw data not cached; run `python -m src.ingest.nflverse` then "
            "`python -m src.labels.build`"
        )


@lru_cache(maxsize=1)
def _default_frames() -> dict[str, pl.DataFrame | None]:
    """Every default input frame, loaded once and cached across tests."""
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
        "ngs_receiving": pl.read_parquet(NGS_RECEIVING_PATH) if NGS_RECEIVING_PATH.exists() else None,
        "coaching_changes": sh.load_coaching_changes(),
    }


def _build(**overrides) -> pl.DataFrame:
    frames = dict(_default_frames())
    frames.update(overrides)
    return build_features_wr(**frames)


def _df() -> pl.DataFrame:
    if not OUT_PATH.exists():
        pytest.skip(f"{OUT_PATH.name} not built; run `python -m src.features.wr`")
    return pl.read_parquet(OUT_PATH)


# --------------------------------------------------------------------------
# Leakage test — the important one
# --------------------------------------------------------------------------
#
# Two independent mechanisms:
#
# 1. Recompute-and-compare: independently recompute a handful of N-1-derived
#    values for known players straight from player_stats (bypassing every
#    builder in src/features/) and assert they equal the stored row exactly
#    — e.g. Jefferson's 2022 row's target_share_n1 must equal his own 2021
#    season target_share, not anything from 2022.
#
# 2. Structural: build season-2023 WR rows twice — once from the full
#    player_stats frame, once from a copy with every 2023 row *deleted* —
#    and assert the season-2023 output rows are byte-identical between the
#    two builds. player_stats is the only season-N-shaped input every
#    season-N-labeled feature could possibly read game-level 2023 data
#    from (ff_opportunity/schedules/rosters/draft_picks are all read
#    exclusively for N-1-or-earlier / preseason-N facts elsewhere in the
#    pipeline); deleting its 2023 rows and getting identical output proves
#    nothing in features_wr.parquet's season-2023 rows depends on them.


def test_leakage_recompute_jefferson_target_share() -> None:
    _skip_if_raw_missing()
    player_stats = pl.read_parquet(sh.PATHS["player_stats"])
    reg = sh.filter_reg(player_stats)

    # Independent, from-scratch recomputation: Jefferson's 2021 target share
    # against his 2021 primary team's 2021 total targets.
    jefferson_id = "00-0036322"
    p21 = reg.filter((pl.col("season") == 2021) & (pl.col("player_id") == jefferson_id))
    if p21.is_empty():
        pytest.skip("Justin Jefferson 2021 not in player_stats")
    team = p21.group_by("team").len().sort("len", descending=True).get_column("team")[0]
    player_targets = p21.filter(pl.col("team") == team).get_column("targets").sum()
    team_targets = reg.filter((pl.col("season") == 2021) & (pl.col("team") == team)).get_column("targets").sum()
    expected_share = player_targets / team_targets

    out = _df()
    row = out.filter((pl.col("season") == 2022) & (pl.col("gsis_id") == jefferson_id))
    if row.height == 0:
        pytest.skip("Justin Jefferson 2022 not in features_wr output")
    stored = row.get_column("target_share_n1").item()
    assert stored == pytest.approx(expected_share, abs=1e-9)


def _assert_frames_close(a: pl.DataFrame, b: pl.DataFrame, *, context: str) -> None:
    """Column-wise equality, tolerant of float64 summation-order noise (~1e-9).

    polars aggregation order (group_by hashing / internal parallel chunking)
    is not guaranteed stable across two structurally different input
    frames, so a ``.sum()``-derived float column built from the *same*
    underlying 2022 rows can differ from the full build in the last couple
    ULPs even though nothing 2023-derived touched it (verified by hand for
    ``vacated_target_share``: identical up to ~5e-17). Non-float columns
    (ids, strings, ints, bools) must still match exactly.
    """
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


def test_leakage_structural_2023_stripped_matches_full_build() -> None:
    _skip_if_raw_missing()
    frames = dict(_default_frames())

    full = _build().filter(pl.col("season") == 2023).sort(["season", "gsis_id"])
    if full.height == 0:
        pytest.skip("no season-2023 WR rows to check")

    stripped_stats = frames["player_stats"].filter(pl.col("season") != 2023)
    stripped = _build(player_stats=stripped_stats).filter(pl.col("season") == 2023).sort(["season", "gsis_id"])

    _assert_frames_close(
        full, stripped, context="season-2023 feature rows changed when 2023 player_stats rows were removed"
    )


# --------------------------------------------------------------------------
# new_hc correctness on known cases
# --------------------------------------------------------------------------


def test_new_hc_known_cases() -> None:
    """Verified directly against schedules.parquet's week-1 home/away coach columns.

    2024 ATL: Arthur Smith (fired after 2023) -> Raheem Morris, a genuinely
    new Week-1 coach -> 1. 2024 KC: Andy Reid continuing -> 0.
    """
    _skip_if_raw_missing()
    schedules = pl.read_parquet(sh.PATHS["schedules"])
    nh = sh.new_hc_table(schedules)
    atl = nh.filter((pl.col("season") == 2024) & (pl.col("team") == "ATL")).get_column("new_hc").item()
    kc = nh.filter((pl.col("season") == 2024) & (pl.col("team") == "KC")).get_column("new_hc").item()
    assert atl == 1
    assert kc == 0


# --------------------------------------------------------------------------
# Vacated-share sanity: Davante Adams leaving GB after 2021
# --------------------------------------------------------------------------


def test_vacated_target_share_davante_adams_departure() -> None:
    """GB's 2022 vacated_target_share must be >= Adams' own 2021 GB target share.

    Verified in data: Adams (00-0031381) target-shared ~29.6% of GB's 2021
    targets, then signed with LV before the 2022 season (his season_team
    flips to LV) — GB's 2022 vacated_target_share must be at least that
    large (other 2021 departures, e.g. Marquez Valdes-Scantling to KC, add
    on top of it).
    """
    _skip_if_raw_missing()
    player_stats = pl.read_parquet(sh.PATHS["player_stats"])
    reg = sh.filter_reg(player_stats)
    adams_id = "00-0031381"
    p21 = reg.filter((pl.col("season") == 2021) & (pl.col("player_id") == adams_id) & (pl.col("team") == "GB"))
    if p21.is_empty():
        pytest.skip("Davante Adams 2021 GB not in player_stats")
    adams_targets = p21.get_column("targets").sum()
    gb_targets_2021 = reg.filter((pl.col("season") == 2021) & (pl.col("team") == "GB")).get_column("targets").sum()
    adams_share = adams_targets / gb_targets_2021

    rosters = pl.read_parquet(sh.PATHS["rosters"])
    rosters_weekly = pl.read_parquet(sh.PATHS["rosters_weekly"])
    season_team = sh.season_roster_team(rosters, rosters_weekly)
    adams_2022_team = season_team.filter((pl.col("season") == 2022) & (pl.col("gsis_id") == adams_id))
    if adams_2022_team.height == 0 or adams_2022_team.get_column("team").item() == "GB":
        pytest.skip("Davante Adams 2022 roster team not resolved away from GB in this data snapshot")

    vs = sh.vacated_shares(reg, season_team)
    gb_2022 = vs.filter((pl.col("season") == 2022) & (pl.col("team") == "GB"))
    assert gb_2022.height == 1
    assert gb_2022.get_column("vacated_target_share").item() >= adams_share - 1e-9


# --------------------------------------------------------------------------
# Regression: 2024 Davante Adams (LV -> NYJ October trade) must key off his
# Week-1 team (LV), not his season-final team (NYJ). This is the leak a
# season-level-rosters-sourced team assignment produced in an earlier
# version of this module — see src.features.shared.season_roster_team's
# docstring.
# --------------------------------------------------------------------------


def test_adams_2024_keys_off_week1_team_not_final_team() -> None:
    """Demonstrates the leak directly: swap Adams' resolved 2024 team to his
    real final team (NYJ) and show LV's 2024 vacated_target_share inflates
    by exactly his 2023 LV target share — then confirm the actual (fixed)
    build does not have that inflation.
    """
    _skip_if_raw_missing()
    adams_id = "00-0031381"

    rosters = pl.read_parquet(sh.PATHS["rosters"])
    rosters_weekly = pl.read_parquet(sh.PATHS["rosters_weekly"])
    season_team = sh.season_roster_team(rosters, rosters_weekly)
    row = season_team.filter((pl.col("season") == 2024) & (pl.col("gsis_id") == adams_id))
    assert row.height == 1
    assert row.get_column("team").item() == "LV", "season_roster_team must resolve Adams 2024 to LV (Week 1), not NYJ"
    assert row.get_column("preseason_team_fallback").item() == 0

    player_stats = pl.read_parquet(sh.PATHS["player_stats"])
    reg = sh.filter_reg(player_stats)
    usage = sh.player_team_usage(reg)
    totals = sh.team_usage_totals(reg)
    adams_2023_lv = usage.filter(
        (pl.col("season") == 2023) & (pl.col("gsis_id") == adams_id) & (pl.col("team") == "LV")
    )
    if adams_2023_lv.height == 0:
        pytest.skip("Davante Adams 2023 LV usage not in player_stats")
    lv_2023_total = totals.filter((pl.col("season") == 2023) & (pl.col("team") == "LV"))
    adams_2023_share = adams_2023_lv.get_column("targets").item() / lv_2023_total.get_column("team_targets").item()

    vs_correct = sh.vacated_shares(reg, season_team)
    lv_correct = vs_correct.filter((pl.col("season") == 2024) & (pl.col("team") == "LV"))
    assert lv_correct.height == 1
    lv_correct_share = lv_correct.get_column("vacated_target_share").item()

    # Simulate the leak this fix removes: resolve Adams' season-2024 team to
    # his real *final* 2024 team (NYJ, his October trade destination) instead
    # of his real Week-1 team (LV) — exactly what the season-level
    # rosters.parquet table did before rosters_weekly.parquet was available.
    leaky_season_team = season_team.with_columns(
        pl.when((pl.col("season") == 2024) & (pl.col("gsis_id") == adams_id))
        .then(pl.lit("NYJ"))
        .otherwise(pl.col("team"))
        .alias("team")
    )
    vs_leaky = sh.vacated_shares(reg, leaky_season_team)
    lv_leaky = vs_leaky.filter((pl.col("season") == 2024) & (pl.col("team") == "LV"))
    assert lv_leaky.height == 1
    lv_leaky_share = lv_leaky.get_column("vacated_target_share").item()

    assert lv_leaky_share - lv_correct_share == pytest.approx(adams_2023_share, abs=1e-9), (
        "the leaky (final-team) resolution should inflate LV's vacated share by exactly "
        "Adams' own 2023 LV target share, confirming the fixed (Week-1) resolution removes that inflation"
    )


# --------------------------------------------------------------------------
# Nullability
# --------------------------------------------------------------------------

_CORE_COLUMNS = [
    "age",
    "age_sq",
    "years_exp",
    "year_in_league",
    "draft_round",
    "draft_pick",
    "log_draft_pick",
    "undrafted",
    "team_pass_att_pg_prior",
    "team_plays_pg_prior",
    "team_pass_rate_prior",
    "vacated_target_share",
    "vacated_carry_share",
    "label_season_2020",
    "target_share_n1",
    "air_yards_share_n1",
    "wopr_n1",
    "targets_pg_n1",
    "receptions_pg_n1",
    "rec_yards_pg_n1",
    "adot_n1",
    "ppr_ppg_n1",
    "yards_per_reception_n1",
    "td_rate_n1",
]

_NGS_ONLY_COLUMNS = ["avg_separation_n1", "avg_cushion_n1", "catch_percentage_n1"]


def test_pre_2016_rows_have_null_ngs_but_nonnull_core_columns() -> None:
    df = _df()
    pre2016 = df.filter(pl.col("season") < 2016)
    if pre2016.is_empty():
        pytest.skip("no pre-2016 WR rows in features_wr output")

    for col in _NGS_ONLY_COLUMNS:
        assert pre2016.get_column(col).null_count() == pre2016.height, (
            f"{col} should be entirely null pre-2016 (NGS starts 2016)"
        )

    for col in _CORE_COLUMNS:
        assert pre2016.get_column(col).null_count() < pre2016.height, (
            f"{col} is entirely null for pre-2016 rows — should have real values"
        )


def test_no_all_null_core_columns() -> None:
    """The frame has zero all-null core columns (across the whole table, not just pre-2016)."""
    df = _df()
    for col in _CORE_COLUMNS:
        assert df.get_column(col).null_count() < df.height, f"{col} is entirely null across the whole output"


# --------------------------------------------------------------------------
# Uniqueness
# --------------------------------------------------------------------------


def test_no_duplicate_season_gsis_id() -> None:
    df = _df()
    dupes = df.group_by(["season", "gsis_id"]).len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"duplicate (season, gsis_id) rows: {dupes}"


# --------------------------------------------------------------------------
# Population correctness: rookies excluded, non-rookie WRs included
# --------------------------------------------------------------------------


def test_rookie_seasons_excluded() -> None:
    df = _df()
    labels = pl.read_parquet(LABELS_PATH)
    rookie_wr = labels.filter((pl.col("position") == "WR") & pl.col("is_rookie")).select("season", "gsis_id")
    overlap = df.join(rookie_wr, on=["season", "gsis_id"], how="inner")
    assert overlap.is_empty(), f"rookie WR seasons leaked into features_wr: {overlap.select('season','gsis_id')}"


def test_non_rookie_wr_population_matches_labels() -> None:
    df = _df()
    labels = pl.read_parquet(LABELS_PATH)
    expected = labels.filter((pl.col("position") == "WR") & (~pl.col("is_rookie"))).select("season", "gsis_id")
    assert df.height == expected.height
    missing = expected.join(df, on=["season", "gsis_id"], how="anti")
    assert missing.is_empty(), f"non-rookie WR labels rows missing from features_wr: {missing.height}"
