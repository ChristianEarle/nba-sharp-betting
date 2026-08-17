"""Threshold, schema, and coverage guards for the Phase 2 breakout labels.

Build-dependent tests (canaries, distribution, sanity) read the gitignored
data/processed/labels.parquet and skip when it is absent (fresh clone / CI),
matching the pattern in test_adp.py / test_ingest.py. Run
``python -m src.labels.build`` first to build it.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.labels.build import (
    OUT_PATH,
    PLAYER_STATS_PATH,
    SKILL_POSITIONS,
    add_labels,
    load_labels_config,
    load_reg_stats,
    load_scoring_config,
    season_aggregates,
)

THRESHOLDS = load_labels_config()["thresholds"]


def _df() -> pl.DataFrame:
    if not OUT_PATH.exists():
        pytest.skip(f"{OUT_PATH.name} not built; run `python -m src.labels.build`")
    return pl.read_parquet(OUT_PATH)


# --------------------------------------------------------------------------
# Threshold unit tests (synthetic, one boundary case per position)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("position", SKILL_POSITIONS)
def test_breakout_at_exact_threshold_is_positive(position: str) -> None:
    t = THRESHOLDS[position]
    df = pl.DataFrame(
        {"position": [position], "finish_pos_rank": [t["finish_top"]], "expectation_pos_rank": [t["adp_worse_than"]]}
    )
    out = add_labels(df, THRESHOLDS)
    assert out.get_column("breakout").item() == 1


@pytest.mark.parametrize("position", SKILL_POSITIONS)
def test_breakout_one_finish_rank_worse_than_threshold_is_negative(position: str) -> None:
    """finish_pos_rank one worse than finish_top fails the finish condition even at the ADP edge."""
    t = THRESHOLDS[position]
    df = pl.DataFrame(
        {
            "position": [position],
            "finish_pos_rank": [t["finish_top"] + 1],
            "expectation_pos_rank": [t["adp_worse_than"]],
        }
    )
    out = add_labels(df, THRESHOLDS)
    assert out.get_column("breakout").item() == 0


@pytest.mark.parametrize("position", SKILL_POSITIONS)
def test_breakout_one_finish_rank_better_than_threshold_is_positive(position: str) -> None:
    t = THRESHOLDS[position]
    df = pl.DataFrame(
        {
            "position": [position],
            "finish_pos_rank": [t["finish_top"] - 1],
            "expectation_pos_rank": [t["adp_worse_than"]],
        }
    )
    out = add_labels(df, THRESHOLDS)
    assert out.get_column("breakout").item() == 1


@pytest.mark.parametrize("position", SKILL_POSITIONS)
def test_breakout_one_expectation_rank_better_than_threshold_is_negative(position: str) -> None:
    """expectation_pos_rank one *better* (lower number) than adp_worse_than means the market

    already expected it — not eligible even with a qualifying finish.
    """
    t = THRESHOLDS[position]
    df = pl.DataFrame(
        {
            "position": [position],
            "finish_pos_rank": [t["finish_top"]],
            "expectation_pos_rank": [t["adp_worse_than"] - 1],
        }
    )
    out = add_labels(df, THRESHOLDS)
    assert out.get_column("breakout").item() == 0


@pytest.mark.parametrize("position", SKILL_POSITIONS)
def test_breakout_one_expectation_rank_worse_than_threshold_is_positive(position: str) -> None:
    t = THRESHOLDS[position]
    df = pl.DataFrame(
        {
            "position": [position],
            "finish_pos_rank": [t["finish_top"]],
            "expectation_pos_rank": [t["adp_worse_than"] + 1],
        }
    )
    out = add_labels(df, THRESHOLDS)
    assert out.get_column("breakout").item() == 1


def test_null_finish_rank_is_never_a_breakout() -> None:
    df = pl.DataFrame({"position": ["WR"], "finish_pos_rank": [None], "expectation_pos_rank": [999]})
    out = add_labels(df, THRESHOLDS)
    assert out.get_column("breakout").item() == 0
    assert out.get_column("finish_rank_delta").item() is None


# --------------------------------------------------------------------------
# REG-only aggregation
# --------------------------------------------------------------------------


def test_post_season_weeks_do_not_inflate_games(tmp_path) -> None:
    """A player with 17 REG + 2 POST weeks must aggregate to 17 games, matching

    the real Puka Nacua 2023 case documented in README's "Known data
    issues" (17 REG vs 18 unfiltered with only 1 POST game there; this
    synthetic case uses 2 POST weeks to make the filter unmissable).
    """
    weeks = list(range(1, 18))  # 17 REG weeks
    n_reg = len(weeks)
    rows = {
        "player_id": ["00-9999999"] * (n_reg + 2),
        "player_name": ["Synthetic Player"] * (n_reg + 2),
        "player_display_name": ["Synthetic Player"] * (n_reg + 2),
        "position": ["WR"] * (n_reg + 2),
        "season": [2023] * (n_reg + 2),
        "week": weeks + [18, 19],
        "season_type": ["REG"] * n_reg + ["POST", "POST"],
        "rushing_yards": [0] * (n_reg + 2),
        "receiving_yards": [50] * (n_reg + 2),
        "passing_yards": [0] * (n_reg + 2),
        "rushing_tds": [0] * (n_reg + 2),
        "receiving_tds": [0] * (n_reg + 2),
        "passing_tds": [0] * (n_reg + 2),
        "receptions": [4] * (n_reg + 2),
        "passing_interceptions": [0] * (n_reg + 2),
        "fumbles_lost_total": [0] * (n_reg + 2),
        "passing_2pt_conversions": [0] * (n_reg + 2),
        "rushing_2pt_conversions": [0] * (n_reg + 2),
        "receiving_2pt_conversions": [0] * (n_reg + 2),
        "fantasy_points_ppr": [9.0] * (n_reg + 2),
    }
    path = tmp_path / "player_stats.parquet"
    pl.DataFrame(rows).write_parquet(path)

    profile = load_scoring_config()
    reg = load_reg_stats(path, profile)
    agg = season_aggregates(reg)
    row = agg.filter(pl.col("gsis_id") == "00-9999999")
    assert row.get_column("games").item() == 17, "POST weeks leaked into the games count"


def test_real_player_stats_reg_only(tmp_path) -> None:
    if not PLAYER_STATS_PATH.exists():
        pytest.skip("player_stats.parquet not cached; run `python -m src.ingest.nflverse`")
    profile = load_scoring_config()
    reg = load_reg_stats(PLAYER_STATS_PATH, profile)
    assert reg.get_column("season_type").unique().to_list() == ["REG"]


# --------------------------------------------------------------------------
# Uniqueness
# --------------------------------------------------------------------------


def test_season_gsis_id_unique() -> None:
    df = _df()
    dupes = df.group_by(["season", "gsis_id"]).len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"duplicate (season, gsis_id) rows: {dupes}"


# --------------------------------------------------------------------------
# adp_source distribution
# --------------------------------------------------------------------------


def test_adp_source_distribution_by_era() -> None:
    df = _df()
    for season in range(2020, 2026):
        sources = set(df.filter(pl.col("season") == season).get_column("adp_source").unique().to_list())
        assert sources <= {"ecr", "capped"}, f"season {season}: unexpected adp_source values {sources}"
    for season in range(2014, 2020):
        sources = set(df.filter(pl.col("season") == season).get_column("adp_source").unique().to_list())
        assert sources <= {"proxy", "capped"}, f"season {season}: unexpected adp_source values {sources}"


# --------------------------------------------------------------------------
# Named canaries
# --------------------------------------------------------------------------


def test_canary_puka_nacua_2023() -> None:
    df = _df()
    row = df.filter((pl.col("season") == 2023) & (pl.col("player_name") == "Puka Nacua"))
    if row.height == 0:
        pytest.skip("Puka Nacua 2023 not in labels output")
    r = row.row(0, named=True)
    assert r["breakout"] == 1
    assert r["adp_source"] == "ecr"
    assert r["expectation_pos_rank"] == 96
    # By ppr_ppg (this pipeline's finish basis) he lands WR6, not WR5 — the
    # spec's "top-5" is a season-total-points framing; per-game he's a hair
    # behind an injury-shortened Justin Jefferson (10 games, 20.2 ppg). Both
    # readings clear the WR breakout finish_top of 18 by a wide margin.
    assert r["finish_pos_rank"] <= 10


def test_canary_justin_jefferson_2022_not_a_breakout() -> None:
    df = _df()
    row = df.filter((pl.col("season") == 2022) & (pl.col("player_name") == "Justin Jefferson"))
    if row.height == 0:
        pytest.skip("Justin Jefferson 2022 not in labels output")
    r = row.row(0, named=True)
    assert r["breakout"] == 0
    assert r["adp_source"] == "ecr"
    assert r["expectation_pos_rank"] == 1
    assert r["finish_pos_rank"] <= 5


def test_canary_cordarrelle_patterson_2021_finished_just_outside_breakout() -> None:
    """Real outcome, not a bug: Patterson finished RB17 by ppr_ppg in 2021

    (Henry/Taylor/Ekeler/Fournette/Kamara all correctly rank above him),
    one spot outside the RB finish_top of 15. Deep preseason expectation
    (undrafted-as-RB conversion player, expectation_pos_rank=134) is not in
    question — the finish side alone misses by one rank. Encoding what the
    data actually says rather than forcing breakout=1.
    """
    df = _df()
    row = df.filter((pl.col("season") == 2021) & (pl.col("player_name") == "Cordarrelle Patterson"))
    if row.height == 0:
        pytest.skip("Cordarrelle Patterson 2021 not in labels output")
    r = row.row(0, named=True)
    assert r["position"] == "RB"
    assert r["adp_source"] == "ecr"
    assert r["expectation_pos_rank"] >= 60  # deep, unambiguously "not expected"
    assert 15 <= r["finish_pos_rank"] <= 20  # eligible, top-tier, but not top-15
    assert r["breakout"] == 0


def test_canary_lamar_jackson_2019_proxy_era_breakout() -> None:
    """Proxy-era check (2019 uses 2018's finish rank as expectation, since

    market_expectation.parquet only reaches back to 2020). 2018 rookie-year
    Jackson mixes 8 late-season starts with several near-empty early-season
    spot appearances (per src/labels/build.py's `games` definition — any
    player_stats row counts, regardless of snap share), which drags his
    2018 ppr_ppg down to a QB34-level proxy rank. He then took the starting
    job full-time in 2019 and won MVP — a legitimate proxy-era breakout.
    """
    df = _df()
    row = df.filter((pl.col("season") == 2019) & (pl.col("player_name") == "Lamar Jackson"))
    if row.height == 0:
        pytest.skip("Lamar Jackson 2019 not in labels output")
    r = row.row(0, named=True)
    assert r["adp_source"] == "proxy"
    assert r["finish_pos_rank"] == 1
    assert r["expectation_pos_rank"] >= 16  # QB16 or later -> eligible
    assert r["breakout"] == 1


def test_canary_alvin_kamara_2017_rookie_capped() -> None:
    """2017 is proxy-era, but Kamara was a rookie that year — no prior season

    exists to build a proxy from, so he correctly falls to adp_source=
    'capped' rather than 'proxy'. Confirms capped applies within the proxy
    era too, and that rookies are excluded from in_training_pool even when
    they carry a real breakout label.
    """
    df = _df()
    row = df.filter((pl.col("season") == 2017) & (pl.col("player_name") == "Alvin Kamara"))
    if row.height == 0:
        pytest.skip("Alvin Kamara 2017 not in labels output")
    r = row.row(0, named=True)
    assert r["adp_source"] == "capped"
    assert r["is_rookie"] is True
    assert r["in_training_pool"] is False
    assert r["breakout"] == 1


# --------------------------------------------------------------------------
# Positive-rate sanity
# --------------------------------------------------------------------------


def test_positive_rate_sanity_by_position() -> None:
    """Loose pooled sanity bounds; the printed __main__ table is the real eyeball check.

    WR/TE get a lower floor than QB/RB: roughly a third of WR/TE
    player-seasons with >=1 game are short-career/committee players under
    min_games (8) who can never get a finish rank and are automatically
    breakout=0, which dilutes the games>=1 denominator more than it does
    for QB/RB. Restricted to games>=min_games, all four positions land
    3.3%-12.5% — see the __main__ report for the exact eligible-population
    rates.
    """
    df = _df()
    sub = df.filter(pl.col("games") >= 1)
    bounds = {"QB": (0.03, 0.25), "RB": (0.03, 0.25), "WR": (0.015, 0.25), "TE": (0.015, 0.25)}
    for position, (lo, hi) in bounds.items():
        pos_df = sub.filter(pl.col("position") == position)
        rate = pos_df.get_column("breakout").mean()
        assert lo <= rate <= hi, f"{position}: pooled breakout rate {rate:.1%} outside [{lo:.1%}, {hi:.1%}]"
