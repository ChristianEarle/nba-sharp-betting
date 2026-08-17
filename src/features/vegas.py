"""Preseason Vegas team-line features (v1.6), from the Odds API acquisition.

Derives per-(season, team) market-expectation features from
``data/external/odds_api/team_lines.parquet`` — the preseason snapshot
(4 days before each season's first kickoff, 2020-2025) pulled by
``src.ingest.odds_api`` — and attaches them to the four position feature
tables. This is the v1.5 README's "No Vegas data anywhere in training"
limitation being retired.

Leakage rule: every number here was published by sportsbooks *before*
Week 1 of the labeled season N — a preseason fact in exactly the sense of
``src.features.shared``'s convention (same class as draft results and
Week-1 rosters), not N-season outcome data. No shifting is needed or
wanted: season-N lines describe season N.

Features (null for label seasons before 2020 — the API's historical
coverage floor — and for any team-season the snapshot didn't list):

- ``vegas_implied_pts_pg``  mean implied team points across the team's
  listed games: ``total/2 - team_spread/2`` (team_spread is the team's own
  spread, negative when favored — the standard implied-total identity).
  Computed only over games where both spread and total exist.
- ``vegas_total_pg``        mean over/under total across listed games.
- ``vegas_spread_pg``       mean team spread (negative = favored on
  average — a season-strength prior in its own right).
- ``vegas_win_prob``        mean de-vigged moneyline win probability
  across listed games (each game's two implied probabilities renormalized
  to sum to 1; games missing either side's price are skipped).
- ``vegas_n_games``         how many of the team's games the snapshot
  listed — an explicit coverage signal (2020's snapshot listed partial
  slates; 2023-2025 list all 272 games).

Week-1 player props are deliberately NOT modeled here: their historical
coverage is 2024-2025 plus a single 2023 game (measured during the
acquisition — see data/external/odds_api's commit), which is entirely
inside the holdout years. A feature with no non-null training rows before
2024 cannot be learned from under the expanding-window CV (train <=2023);
it would ride along as noise. The raw props JSON stays in
``data/external/odds_api/raw/`` for when a future season's coverage makes
them trainable (or for a 2026-board overlay, which needs no training).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_LINES_PATH = REPO_ROOT / "data" / "external" / "odds_api" / "team_lines.parquet"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# The Odds API's full team names -> the present-day franchise codes
# player_stats.parquet (and therefore every feature table's `team` column)
# uses -- see src.features.shared's team-code writeup. Both Washington
# names map to WAS: the franchise renamed (Football Team 2020-2021,
# Commanders 2022-) without relocating.
TEAM_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
}

VEGAS_FEATURES = [
    "vegas_implied_pts_pg",
    "vegas_total_pg",
    "vegas_spread_pg",
    "vegas_win_prob",
    "vegas_n_games",
]


def _ml_prob(ml: pl.Expr) -> pl.Expr:
    """American moneyline -> raw implied probability (vig included)."""
    return (
        pl.when(ml.is_null())
        .then(None)
        .when(ml < 0)
        .then((-ml) / ((-ml) + 100.0))
        .otherwise(100.0 / (ml + 100.0))
    )


def team_game_rows(team_lines: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, team, event): the team's own spread, the game total, de-vigged win prob.

    Unknown team names (a lookup miss) raise rather than silently dropping —
    a new franchise name in a future season's pull should fail loudly here.
    """
    names = set(team_lines.get_column("home_team").to_list()) | set(
        team_lines.get_column("away_team").to_list()
    )
    unknown = sorted(n for n in names if n not in TEAM_NAME_TO_CODE)
    assert not unknown, f"team_lines carries team names with no code mapping: {unknown}"

    base = team_lines.with_columns(
        _ml_prob(pl.col("home_ml")).alias("_p_home_raw"),
        _ml_prob(pl.col("away_ml")).alias("_p_away_raw"),
    ).with_columns(
        (pl.col("_p_home_raw") / (pl.col("_p_home_raw") + pl.col("_p_away_raw"))).alias("_p_home"),
    )

    home = base.select(
        "season",
        "event_id",
        pl.col("home_team").replace_strict(TEAM_NAME_TO_CODE).alias("team"),
        pl.col("home_spread").alias("team_spread"),
        "total",
        pl.col("_p_home").alias("win_prob"),
    )
    away = base.select(
        "season",
        "event_id",
        pl.col("away_team").replace_strict(TEAM_NAME_TO_CODE).alias("team"),
        (-pl.col("home_spread")).alias("team_spread"),
        "total",
        (1.0 - pl.col("_p_home")).alias("win_prob"),
    )
    return pl.concat([home, away], how="vertical")


def vegas_team_features(team_lines: pl.DataFrame) -> pl.DataFrame:
    """season, team -> the VEGAS_FEATURES columns (see module docstring)."""
    rows = team_game_rows(team_lines)
    rows = rows.with_columns(
        pl.when(pl.col("total").is_null() | pl.col("team_spread").is_null())
        .then(None)
        .otherwise(pl.col("total") / 2.0 - pl.col("team_spread") / 2.0)
        .alias("_implied_pts")
    )
    return (
        rows.group_by(["season", "team"])
        .agg(
            pl.col("_implied_pts").mean().alias("vegas_implied_pts_pg"),
            pl.col("total").mean().alias("vegas_total_pg"),
            pl.col("team_spread").mean().alias("vegas_spread_pg"),
            pl.col("win_prob").mean().alias("vegas_win_prob"),
            pl.len().alias("vegas_n_games"),
        )
        .with_columns(pl.col("vegas_n_games").cast(pl.Int64))
        .sort(["season", "team"])
    )


def attach_vegas(features: pl.DataFrame, vegas: pl.DataFrame) -> pl.DataFrame:
    """Left-join vegas features onto a position feature frame on (season, team).

    Idempotent: any existing vegas_* columns are dropped first, so re-running
    the __main__ attach over an already-augmented parquet doesn't stack
    duplicate columns.
    """
    existing = [c for c in features.columns if c in VEGAS_FEATURES]
    if existing:
        features = features.drop(existing)
    return features.join(vegas, on=["season", "team"], how="left")


def main() -> int:
    if not TEAM_LINES_PATH.exists():
        print(f"vegas | {TEAM_LINES_PATH} not found -- run src.ingest.odds_api --normalize first")
        return 1
    vegas = vegas_team_features(pl.read_parquet(TEAM_LINES_PATH))
    print(f"vegas | {vegas.height} (season, team) rows from {TEAM_LINES_PATH.name}")

    for pos in ("wr", "rb", "te", "qb"):
        path = PROCESSED_DIR / f"features_{pos}.parquet"
        if not path.exists():
            print(f"vegas | {path.name} missing -- build it first (src.features.{pos}); skipped")
            continue
        df = attach_vegas(pl.read_parquet(path), vegas)
        covered = df.filter((pl.col("season") >= 2020) & pl.col("vegas_implied_pts_pg").is_not_null())
        in_era = df.filter(pl.col("season") >= 2020)
        df.write_parquet(path)
        print(
            f"vegas | {path.name}: {df.height} rows | 2020+ rows with implied pts: "
            f"{covered.height}/{in_era.height}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
