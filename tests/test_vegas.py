"""Tests for src/ingest/vegas.py (v1.5 Phase C, Deliverable 1).

``build_vegas_team`` is exercised entirely against synthetic ``team_lines``-
shaped frames (no dependency on the real gitignored odds_api pull).
``build_vegas_props`` is exercised against synthetic raw-JSON fixtures
(same shape the-odds-api's event-props endpoint returns, verified directly
against the real files in data/external/odds_api/raw/) plus a synthetic,
hand-built crosswalk -- fully isolated from data/raw/ff_playerids.parquet,
matching tests/test_odds_api.py's "zero dependency on the real cache"
convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.ingest import vegas


# --------------------------------------------------------------------------
# Team name normalization
# --------------------------------------------------------------------------


def test_team_name_to_code_covers_washington_both_eras() -> None:
    assert vegas.TEAM_NAME_TO_CODE["Washington Football Team"] == "WAS"
    assert vegas.TEAM_NAME_TO_CODE["Washington Commanders"] == "WAS"


def test_team_name_to_code_has_32_distinct_codes() -> None:
    assert len(set(vegas.TEAM_NAME_TO_CODE.values())) == 32


def test_team_name_to_code_matches_real_schedules_codes_if_cached() -> None:
    schedules_path = vegas.REPO_ROOT / "data" / "raw" / "schedules.parquet"
    if not schedules_path.exists():
        pytest.skip("data/raw/schedules.parquet not cached; run `python -m src.ingest.nflverse`")
    df = pl.read_parquet(schedules_path).filter(pl.col("season") >= 2020)
    real_codes = set(df.get_column("home_team").unique().to_list()) | set(df.get_column("away_team").unique().to_list())
    mapped_codes = set(vegas.TEAM_NAME_TO_CODE.values())
    assert mapped_codes <= real_codes, f"codes not in real schedules: {mapped_codes - real_codes}"


# --------------------------------------------------------------------------
# De-vig math
# --------------------------------------------------------------------------


def test_american_to_prob_matches_hand_math() -> None:
    df = pl.DataFrame({"odds": [150.0, -200.0, 100.0, -100.0]})
    out = df.with_columns(vegas.american_to_prob(pl.col("odds")).alias("p"))
    got = out.get_column("p").to_list()
    assert got[0] == pytest.approx(100 / 250)  # +150 -> 0.4
    assert got[1] == pytest.approx(200 / 300)  # -200 -> 0.6667
    assert got[2] == pytest.approx(0.5)  # +100 -> 0.5
    assert got[3] == pytest.approx(0.5)  # -100 -> 0.5


# --------------------------------------------------------------------------
# build_vegas_team -- implied-points math on a synthetic fixture
# --------------------------------------------------------------------------


def _team_lines(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "season": pl.Int64,
        "snapshot": pl.Utf8,
        "event_id": pl.Utf8,
        "commence_time": pl.Utf8,
        "home_team": pl.Utf8,
        "away_team": pl.Utf8,
        "home_ml": pl.Float64,
        "away_ml": pl.Float64,
        "home_spread": pl.Float64,
        "total": pl.Float64,
    }
    return pl.DataFrame(rows, schema_overrides=schema)


def test_build_vegas_team_implied_points_formula() -> None:
    # Chiefs (home) -6.5 favorites, total 47 -> home 26.75, away 20.25.
    rows = [
        {
            "season": 2024, "snapshot": "s", "event_id": "e1", "commence_time": "t",
            "home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens",
            "home_ml": -280.0, "away_ml": 230.0, "home_spread": -6.5, "total": 47.0,
        }
    ]
    out = vegas.build_vegas_team(_team_lines(rows))
    kc = out.filter(pl.col("team") == "KC").row(0, named=True)
    bal = out.filter(pl.col("team") == "BAL").row(0, named=True)
    assert kc["implied_ppg"] == pytest.approx(26.75)
    assert bal["implied_ppg"] == pytest.approx(20.25)
    assert kc["n_events_priced"] == 1
    assert kc["has_vegas"] == 1
    # de-vigged win prob: p_home_raw=280/380, p_away_raw=100/330; devig sums to 1.
    assert kc["implied_win_prob"] + bal["implied_win_prob"] == pytest.approx(1.0)
    assert kc["implied_win_prob"] > 0.5  # -280 favorite should devig to a >50% favorite


def test_build_vegas_team_averages_across_multiple_events_equally() -> None:
    rows = [
        {
            "season": 2024, "snapshot": "s", "event_id": "e1", "commence_time": "t1",
            "home_team": "Buffalo Bills", "away_team": "New York Jets",
            "home_ml": -150.0, "away_ml": 130.0, "home_spread": -3.0, "total": 44.0,
        },
        {
            "season": 2024, "snapshot": "s", "event_id": "e2", "commence_time": "t2",
            "home_team": "Miami Dolphins", "away_team": "Buffalo Bills",
            "home_ml": 120.0, "away_ml": -140.0, "home_spread": 2.5, "total": 48.0,
        },
    ]
    out = vegas.build_vegas_team(_team_lines(rows))
    buf = out.filter(pl.col("team") == "BUF").row(0, named=True)
    # game 1 (home): 44/2 - (-3)/2 = 23.5; game 2 (away, spread is Miami's home spread
    # so Buffalo's implied points = total/2 + home_spread/2 = 24 + 1.25 = 25.25.
    assert buf["implied_ppg"] == pytest.approx((23.5 + 25.25) / 2)
    assert buf["n_events_priced"] == 2


def test_build_vegas_team_nullable_when_total_or_spread_missing() -> None:
    rows = [
        {
            "season": 2020, "snapshot": "s", "event_id": "e1", "commence_time": "t",
            "home_team": "Denver Broncos", "away_team": "Tennessee Titans",
            "home_ml": None, "away_ml": None, "home_spread": 1.5, "total": None,
        }
    ]
    out = vegas.build_vegas_team(_team_lines(rows))
    den = out.filter(pl.col("team") == "DEN").row(0, named=True)
    assert den["implied_ppg"] is None
    assert den["implied_win_prob"] is None
    assert den["n_events_priced"] == 0
    assert den["has_vegas"] == 1  # still listed, just no priced total/ml this event


def test_build_vegas_team_raises_on_unmapped_team_name() -> None:
    rows = [
        {
            "season": 2024, "snapshot": "s", "event_id": "e1", "commence_time": "t",
            "home_team": "Fictional City Team", "away_team": "Buffalo Bills",
            "home_ml": -150.0, "away_ml": 130.0, "home_spread": -3.0, "total": 44.0,
        }
    ]
    with pytest.raises(ValueError, match="unmapped team name"):
        vegas.build_vegas_team(_team_lines(rows))


def test_build_vegas_team_empty_input() -> None:
    out = vegas.build_vegas_team(pl.DataFrame())
    assert out.height == 0
    assert set(out.columns) == {"season", "team", "implied_ppg", "implied_win_prob", "n_events_priced", "has_vegas"}


# --------------------------------------------------------------------------
# extract_player_market_lines -- median across bookmakers
# --------------------------------------------------------------------------


def _write_event_props_file(path: Path, season: int, event: dict) -> None:
    path.write_text(json.dumps({"timestamp": "t", "data": event}))


def test_extract_player_market_lines_takes_median_over_outcome_only() -> None:
    event = {
        "id": "ev1", "home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_rush_yds",
                        "outcomes": [
                            {"name": "Over", "description": "Derrick Henry", "point": 85.5, "price": -110},
                            {"name": "Under", "description": "Derrick Henry", "point": 85.5, "price": -110},
                        ],
                    }
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_rush_yds",
                        "outcomes": [
                            {"name": "Over", "description": "Derrick Henry", "point": 90.5, "price": -115},
                            {"name": "Under", "description": "Derrick Henry", "point": 90.5, "price": -105},
                        ],
                    }
                ],
            },
        ],
    }

    def run(tmp_path: Path) -> pl.DataFrame:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _write_event_props_file(raw_dir / "2024_week1_props_ev1_2024-01-01T000000Z.json", 2024, event)
        return vegas.extract_player_market_lines(raw_dir)

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = run(Path(td))
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["player_name"] == "Derrick Henry"
    assert row["market"] == "player_rush_yds"
    # median of {85.5, 90.5} (Under duplicates are never read -- see docstring) = 88.0
    assert row["line"] == pytest.approx(88.0)


def test_extract_player_market_lines_empty_bookmakers_contributes_nothing(tmp_path: Path) -> None:
    event = {"id": "ev1", "home_team": "A", "away_team": "B", "bookmakers": []}
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_event_props_file(raw_dir / "2023_week1_props_ev1_2023-01-01T000000Z.json", 2023, event)
    out = vegas.extract_player_market_lines(raw_dir)
    assert out.height == 0


def test_extract_player_market_lines_empty_dir(tmp_path: Path) -> None:
    out = vegas.extract_player_market_lines(tmp_path / "nonexistent")
    assert out.height == 0
    assert set(out.columns) == {"season", "player_name", "market", "line"}


# --------------------------------------------------------------------------
# build_vegas_props -- scoring math + gsis matching, synthetic crosswalk
# --------------------------------------------------------------------------


def _synthetic_crosswalk() -> pl.DataFrame:
    rows = [
        {"mfl_id": "1", "gsis_id": "00-0000001", "fantasypros_id": None, "sleeper_id": None, "name": "Derrick Henry", "position": "RB", "draft_year": 2016, "birthdate": None},
        {"mfl_id": "2", "gsis_id": "00-0000002", "fantasypros_id": None, "sleeper_id": None, "name": "CeeDee Lamb", "position": "WR", "draft_year": 2020, "birthdate": None},
        {"mfl_id": "3", "gsis_id": "00-0000003", "fantasypros_id": None, "sleeper_id": None, "name": "Patrick Mahomes", "position": "QB", "draft_year": 2017, "birthdate": None},
        {"mfl_id": "4", "gsis_id": "00-0000004", "fantasypros_id": None, "sleeper_id": None, "name": "Travis Kelce", "position": "TE", "draft_year": 2013, "birthdate": None},
    ]
    df = pl.DataFrame(rows)
    return df.with_columns(pl.col("name").alias("merge_name").str.to_lowercase())


def _event_for_props(markets_by_player: dict[str, dict[str, float]]) -> dict:
    """{"Derrick Henry": {"player_rush_yds": 85.5}, ...} -> one bookmaker's markets block."""
    by_market: dict[str, list[dict]] = {}
    for player, markets in markets_by_player.items():
        for market, point in markets.items():
            by_market.setdefault(market, []).append(
                {"name": "Over", "description": player, "point": point, "price": -110}
            )
    return {
        "id": "ev1", "home_team": "A", "away_team": "B",
        "bookmakers": [{"key": "draftkings", "markets": [{"key": m, "outcomes": o} for m, o in by_market.items()]}],
    }


def test_build_vegas_props_scores_via_scoring_yaml_weights(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    event = _event_for_props(
        {
            "CeeDee Lamb": {"player_receptions": 6.5, "player_reception_yds": 75.5},
            "Patrick Mahomes": {"player_pass_yds": 275.5, "player_pass_tds": 2.5},
        }
    )
    _write_event_props_file(raw_dir / "2024_week1_props_ev1_2024-01-01T000000Z.json", 2024, event)

    out = vegas.build_vegas_props(raw_dir, crosswalk=_synthetic_crosswalk())
    assert set(out.columns) == {"season", "gsis_id", "player_name", "prop_implied_ppr", "markets_priced", "match_method"}

    lamb = out.filter(pl.col("player_name") == "CeeDee Lamb").row(0, named=True)
    # 6.5 receptions * 1 + 75.5 rec_yards * 0.1 = 6.5 + 7.55 = 14.05
    assert lamb["prop_implied_ppr"] == pytest.approx(14.05)
    assert lamb["markets_priced"] == 2
    assert lamb["gsis_id"] == "00-0000002"
    assert lamb["match_method"] == "exact_name"

    mahomes = out.filter(pl.col("player_name") == "Patrick Mahomes").row(0, named=True)
    # 275.5 pass_yards * 0.04 + 2.5 pass_tds * 4 = 11.02 + 10.0 = 21.02
    assert mahomes["prop_implied_ppr"] == pytest.approx(21.02)
    assert mahomes["gsis_id"] == "00-0000003"


def test_build_vegas_props_te_retry_for_receiving_only_players(tmp_path: Path) -> None:
    """A receiving-only player (no pass/rush market) whose real position is TE, not WR
    (the position-guess ladder's default first try), must still resolve via the TE retry.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    event = _event_for_props({"Travis Kelce": {"player_receptions": 5.5, "player_reception_yds": 55.5}})
    _write_event_props_file(raw_dir / "2024_week1_props_ev1_2024-01-01T000000Z.json", 2024, event)

    out = vegas.build_vegas_props(raw_dir, crosswalk=_synthetic_crosswalk())
    kelce = out.filter(pl.col("player_name") == "Travis Kelce").row(0, named=True)
    assert kelce["gsis_id"] == "00-0000004"
    assert kelce["match_method"] == "exact_name"


def test_build_vegas_props_unmatched_player_kept_with_null_gsis(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    event = _event_for_props({"Totally Unknown Player": {"player_rush_yds": 40.5}})
    _write_event_props_file(raw_dir / "2024_week1_props_ev1_2024-01-01T000000Z.json", 2024, event)

    out = vegas.build_vegas_props(raw_dir, crosswalk=_synthetic_crosswalk())
    row = out.row(0, named=True)
    assert row["gsis_id"] is None
    assert row["match_method"] == "unmatched"


def test_build_vegas_props_empty_raw_dir(tmp_path: Path) -> None:
    out = vegas.build_vegas_props(tmp_path / "nonexistent")
    assert out.height == 0
    assert set(out.columns) == set(vegas._VEGAS_PROPS_SCHEMA)
