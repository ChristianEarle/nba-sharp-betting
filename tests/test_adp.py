"""Schema and coverage guards for the Phase 1b market-expectation table.

Skips when data/processed/market_expectation.parquet is absent (fresh clone
/ CI), matching the pattern in test_ingest.py. Run
``python -m src.ingest.adp`` first to build it.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.ingest.adp import OUT_PATH, SCHEDULES_PATH


def _df() -> pl.DataFrame:
    if not OUT_PATH.exists():
        pytest.skip(f"{OUT_PATH.name} not built; run `python -m src.ingest.adp`")
    return pl.read_parquet(OUT_PATH)


def _first_reg_dates() -> dict[int, object]:
    if not SCHEDULES_PATH.exists():
        pytest.skip("schedules.parquet not cached; run `python -m src.ingest.nflverse`")
    sched = pl.read_parquet(SCHEDULES_PATH).filter(pl.col("game_type") == "REG")
    sched = sched.with_columns(pl.col("gameday").str.to_date())
    first = sched.group_by("season").agg(pl.col("gameday").min().alias("first_reg"))
    return dict(zip(first.get_column("season").to_list(), first.get_column("first_reg").to_list()))


def test_exactly_one_snapshot_date_per_season_source() -> None:
    df = _df()
    per_group = df.group_by(["season", "adp_source"]).agg(pl.col("scrape_date").n_unique().alias("n"))
    bad = per_group.filter(pl.col("n") != 1)
    assert bad.is_empty(), f"season/source with != 1 snapshot date: {bad}"


def test_snapshot_date_precedes_season_kickoff() -> None:
    df = _df()
    first_reg = _first_reg_dates()
    for season, snap_date in (
        df.select("season", "scrape_date").unique().iter_rows()
    ):
        if snap_date is None:
            continue  # manual source with no dated snapshot
        kickoff = first_reg.get(season)
        if kickoff is None:
            continue
        assert snap_date < kickoff, f"season {season}: snapshot {snap_date} not before kickoff {kickoff}"


def test_pos_rank_contiguous_within_season_position() -> None:
    df = _df()
    for (season, position), sub in df.group_by(["season", "position"]):
        ranks = sorted(sub.get_column("pos_rank").to_list())
        expected = list(range(1, sub.height + 1))
        assert ranks == expected, f"{season} {position}: pos_rank not contiguous 1..{sub.height}"


def test_top200_gsis_match_rate_at_least_95_percent() -> None:
    """The gate: if this drops, the crosswalk or ECR filter broke upstream."""
    df = _df()
    failures = []
    for season, sub in df.sort("ecr_overall").group_by("season"):
        top200 = sub.sort("ecr_overall").head(200)
        rate = top200.filter(pl.col("gsis_id").is_not_null()).height / top200.height
        if rate < 0.95:
            failures.append(f"season {season}: top-200 match rate {rate:.1%}")
    assert not failures, "top-200 gsis match rate below 95%:\n" + "\n".join(failures)


def test_canary_puka_nacua_2023_deep_pos_rank() -> None:
    df = _df()
    row = df.filter((pl.col("season") == 2023) & (pl.col("player_name") == "Puka Nacua"))
    assert row.height == 1, "Puka Nacua 2023 missing from market_expectation"
    assert row.get_column("pos_rank").item() >= 40


def test_canary_justin_jefferson_2022_top5() -> None:
    df = _df()
    row = df.filter((pl.col("season") == 2022) & (pl.col("player_name") == "Justin Jefferson"))
    assert row.height == 1, "Justin Jefferson 2022 missing from market_expectation"
    assert row.get_column("pos_rank").item() <= 5


def test_canary_jamarr_chase_2021_present_top40() -> None:
    df = _df()
    row = df.filter((pl.col("season") == 2021) & (pl.col("player_name") == "Ja'Marr Chase"))
    assert row.height == 1, "Ja'Marr Chase 2021 missing from market_expectation"
    assert row.get_column("pos_rank").item() <= 40


def test_regression_rondale_moore_xx_position_resolves() -> None:
    """Rondale Moore's ff_playerids row carries position "XX" (he's since left

    the Cardinals' active roster picture in dynastyprocess's snapshot), which
    used to make build_id_map drop him before the fantasypros_id join ever
    ran. He shows up across three preseasons (2021-2023) and must resolve on
    the id, not fall through to fuzzy/unmatched.
    """
    df = _df()
    for season in (2021, 2022, 2023):
        row = df.filter((pl.col("season") == season) & (pl.col("player_name") == "Rondale Moore"))
        assert row.height == 1, f"Rondale Moore {season} missing from market_expectation"
        assert row.get_column("gsis_id").item() == "00-0036936", f"season {season}: wrong/missing gsis_id"
        assert row.get_column("match_method").item() == "fantasypros_id", f"season {season}: wrong match_method"
