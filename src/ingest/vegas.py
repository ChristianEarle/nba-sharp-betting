"""Vegas-derived team and player features (v1.5 Phase C, Deliverable 1).

Two independent outputs, both nullable / has_vegas-gated by design --
partial snapshot coverage 2020-2022 and thin 2023 props coverage are real,
not bugs; see each builder's docstring for exactly how they're tolerated:

1. ``data/processed/vegas_team.parquet`` -- per (season, team): market-
   implied points-per-game and a de-vigged moneyline win probability,
   derived from ``data/external/odds_api/team_lines.parquet`` (the
   ``--pull-team`` / ``--normalize`` output of ``src.ingest.odds_api``).
2. ``data/processed/vegas_props.parquet`` -- per (season, gsis_id): a
   Week-1 PPR-points rate proxy built from the props raw JSON
   (``data/external/odds_api/raw/*week1_props*.json``), median betting
   line per market scored through ``configs/scoring.yaml``'s
   ``standard_ppr`` weights, matched to gsis_id via
   ``src.ingest.id_map.match_to_gsis``.

Team name normalization
------------------------
``team_lines.parquet`` carries full team names ("Kansas City Chiefs"),
never nflverse's abbreviation codes. ``TEAM_NAME_TO_CODE`` is a static
dict, built by enumerating every distinct ``home_team``/``away_team`` the
real 2020-2025 pull returned (33 names: 32 present-day franchises plus
"Washington Football Team", the franchise's 2020-2021 name before it
became "Washington Commanders" in 2022 -- both map to nflverse's "WAS",
the code ``data/raw/schedules.parquet`` and ``src.features.shared`` use)
against ``schedules.parquet``'s own team codes for the same seasons.
``build_vegas_team`` raises on any name it doesn't recognize rather than
silently dropping a team, since a silent drop here would show up
downstream only as an unexplained null.

Implied points formula
------------------------
For one priced event with a known ``total`` and ``home_spread`` (the
book's home-team-perspective spread: negative means the home team is
favored), the market's own implied final score follows directly:
``home_implied_pts = total / 2 - home_spread / 2`` and
``away_implied_pts = total / 2 + home_spread / 2`` (a -3 home favorite in
a 47-point game is priced to win 25-22, and 47/2 - (-3)/2 = 25). Averaged
equally across every one of that team's own listed events in
``team_lines`` for the season (the preseason historical-odds snapshot
returns whatever games the book had already lined at that date -- see
``src.ingest.odds_api``'s module docstring -- which for 2023-2025 is
close to the full 17-game slate and for 2020-2022 is a handful of
early-marquee games only; either way, every available game counts once,
un-weighted by how early or late in the season it falls).

De-vigged win probability
---------------------------
American odds -> implied probability via the standard formula
(``100 / (odds + 100)`` for positive odds, ``-odds / (-odds + 100)`` for
negative), then the two-way vig removed by dividing each side by their
sum (``p_home / (p_home + p_away)``) -- the textbook two-outcome devig,
not fitted or tuned.

Nullability
------------
A team can appear in ``team_lines`` (i.e. be listed for at least one
event that season) with some events missing ``total``/``home_spread``
(2020-2023's real gap -- see ``src.ingest.odds_api``'s coverage note) or
missing moneylines. Both ``implied_ppg`` and ``implied_win_prob`` are
computed only from that team's events actually carrying the needed
fields (a plain ``pl.mean()`` already skips nulls); a team with zero
qualifying events for a given metric gets a null for that metric, not a
zero. ``n_events_priced`` counts only the events that fed ``implied_ppg``
specifically (documented, not the win-prob event count, which can
differ). ``has_vegas`` is 1 for every row this table emits (the team was
listed at all that season); a team/season entirely absent from
``team_lines`` -- pre-2020, or a team the snapshot never priced -- simply
has no row here at all. Callers that need the has_vegas=0 case do a left
join against this table and fill 0, per the brief (see
``src.features.shared`` callers in Deliverable 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.ingest.id_map import build_id_map, match_to_gsis
from src.labels.build import load_scoring_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = REPO_ROOT / "data" / "external" / "odds_api"
RAW_PROPS_DIR = EXTERNAL_DIR / "raw"
TEAM_LINES_PATH = EXTERNAL_DIR / "team_lines.parquet"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

VEGAS_TEAM_PATH = PROCESSED_DIR / "vegas_team.parquet"
VEGAS_PROPS_PATH = PROCESSED_DIR / "vegas_props.parquet"

# Every home_team/away_team spelling the real 2020-2025 pull returned in
# data/external/odds_api/team_lines.parquet, mapped onto the present-day
# nflverse code src.features.shared.TEAM_ALIASES/player_stats.parquet use.
TEAM_NAME_TO_CODE: dict[str, str] = {
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


# --------------------------------------------------------------------------
# vegas_team.parquet
# --------------------------------------------------------------------------


def american_to_prob(odds: pl.Expr) -> pl.Expr:
    """American odds -> raw (vig-included) implied probability."""
    return pl.when(odds > 0).then(100.0 / (odds + 100.0)).otherwise((-odds) / ((-odds) + 100.0))


def _long_team_events(team_lines: pl.DataFrame) -> pl.DataFrame:
    """One team_lines row (one event) -> two rows (home side, away side).

    Each side carries its own per-event implied_pts (null unless both
    total and home_spread are present) and de-vigged win_prob (null
    unless both moneylines are present) -- see module docstring.
    """
    p_home = american_to_prob(pl.col("home_ml"))
    p_away = american_to_prob(pl.col("away_ml"))
    p_sum = p_home + p_away
    has_ml = pl.col("home_ml").is_not_null() & pl.col("away_ml").is_not_null()

    base = team_lines.with_columns(
        (pl.col("total") / 2 - pl.col("home_spread") / 2).alias("_home_pts"),
        (pl.col("total") / 2 + pl.col("home_spread") / 2).alias("_away_pts"),
        pl.when(has_ml).then(p_home / p_sum).otherwise(None).alias("_home_wp"),
        pl.when(has_ml).then(p_away / p_sum).otherwise(None).alias("_away_wp"),
    )
    home = base.select(
        "season",
        pl.col("home_team").alias("team_name"),
        pl.col("_home_pts").alias("implied_pts"),
        pl.col("_home_wp").alias("win_prob"),
    )
    away = base.select(
        "season",
        pl.col("away_team").alias("team_name"),
        pl.col("_away_pts").alias("implied_pts"),
        pl.col("_away_wp").alias("win_prob"),
    )
    long = pl.concat([home, away], how="vertical_relaxed")

    unmapped = long.filter(~pl.col("team_name").is_in(list(TEAM_NAME_TO_CODE))).select("team_name").unique()
    if unmapped.height > 0:
        raise ValueError(f"vegas.py: unmapped team name(s) in team_lines -- add to TEAM_NAME_TO_CODE: {sorted(unmapped.get_column('team_name').to_list())}")

    return long.with_columns(pl.col("team_name").replace(TEAM_NAME_TO_CODE).alias("team"))


def build_vegas_team(team_lines: pl.DataFrame) -> pl.DataFrame:
    """team_lines.parquet -> season, team, implied_ppg, implied_win_prob, n_events_priced, has_vegas.

    See module docstring for the implied-points formula, the de-vig
    formula, and the nullability contract.
    """
    if team_lines.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "team": pl.Utf8,
                "implied_ppg": pl.Float64,
                "implied_win_prob": pl.Float64,
                "n_events_priced": pl.Int64,
                "has_vegas": pl.Int64,
            }
        )
    long = _long_team_events(team_lines)
    out = long.group_by(["season", "team"]).agg(
        pl.col("implied_pts").mean().alias("implied_ppg"),
        pl.col("implied_pts").drop_nulls().len().alias("n_events_priced"),
        pl.col("win_prob").mean().alias("implied_win_prob"),
    )
    out = out.with_columns(pl.lit(1, dtype=pl.Int64).alias("has_vegas"))
    return out.select(
        "season", "team", "implied_ppg", "implied_win_prob", "n_events_priced", "has_vegas"
    ).sort(["season", "team"])


def run_build_vegas_team(team_lines_path: Path = TEAM_LINES_PATH, out_path: Path = VEGAS_TEAM_PATH) -> pl.DataFrame:
    team_lines = pl.read_parquet(team_lines_path) if team_lines_path.exists() else pl.DataFrame()
    out = build_vegas_team(team_lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return out


# --------------------------------------------------------------------------
# vegas_props.parquet
# --------------------------------------------------------------------------

# market key (the-odds-api) -> configs/scoring.yaml standard_ppr rule name.
# Verified directly against the raw JSON (data/external/odds_api/raw/
# *week1_props*.json): every bookmaker/market present uses exactly these
# five keys, no others.
PROP_MARKET_TO_SCORING_RULE: dict[str, str] = {
    "player_receptions": "reception",
    "player_reception_yds": "rec_yards",
    "player_rush_yds": "rush_yards",
    "player_pass_yds": "pass_yards",
    "player_pass_tds": "pass_td",
}

_VEGAS_PROPS_SCHEMA = {
    "season": pl.Int64,
    "gsis_id": pl.Utf8,
    "player_name": pl.Utf8,
    "prop_implied_ppr": pl.Float64,
    "markets_priced": pl.Int64,
    "match_method": pl.Utf8,
}


def extract_player_market_lines(raw_dir: Path = RAW_PROPS_DIR) -> pl.DataFrame:
    """Every raw/*week1_props*.json event -> one row per (season, player_name, market): the
    median betting line across bookmakers.

    Reads only the "Over" outcome's ``point`` per (player, market,
    bookmaker) -- Over and Under always carry the identical line (verified
    directly against the raw JSON), so reading both would just duplicate
    every value into the median rather than adding information. An event
    with an empty ``bookmakers`` list (the real gap: 2023 Week 1 has
    exactly one event -- the Thursday opener -- with any bookmaker data at
    all, the other 14 events returned nothing at this snapshot; 2024/2025
    are fully populated) contributes nothing, not an error -- see
    ``src.ingest.odds_api``'s module docstring for why.
    """
    rows: list[dict] = []
    for path in sorted(raw_dir.glob("*week1_props*.json")):
        try:
            season = int(path.stem.split("_")[0])
        except ValueError:
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        event = payload.get("data")
        if not isinstance(event, dict):
            continue
        for bk in event.get("bookmakers", []):
            book = bk.get("key")
            for mk in bk.get("markets", []):
                market = mk.get("key")
                if market not in PROP_MARKET_TO_SCORING_RULE:
                    continue
                for outcome in mk.get("outcomes", []):
                    if outcome.get("name") != "Over":
                        continue
                    player = outcome.get("description")
                    point = outcome.get("point")
                    if player is None or point is None:
                        continue
                    rows.append(
                        {"season": season, "player_name": player, "market": market, "book": book, "point": float(point)}
                    )

    if not rows:
        return pl.DataFrame(schema={"season": pl.Int64, "player_name": pl.Utf8, "market": pl.Utf8, "line": pl.Float64})

    df = pl.DataFrame(rows)
    return df.group_by(["season", "player_name", "market"]).agg(pl.col("point").median().alias("line"))


def _position_guess_expr() -> pl.Expr:
    """Which position(s) to try src.ingest.id_map.match_to_gsis against, from which prop
    markets a player was priced in -- the props JSON carries no position column of its own.

    Priority: a passing market -> QB; else a rushing market -> RB; else a
    receiving-only market -> WR (tried first) with a TE retry for whatever
    stays unmatched (see ``match_props_to_gsis`` -- WR and TE are
    genuinely indistinguishable from market composition alone, this is a
    documented heuristic, not a real position signal).
    """
    has_pass = pl.col("player_pass_yds").is_not_null() | pl.col("player_pass_tds").is_not_null()
    has_rush = pl.col("player_rush_yds").is_not_null()
    has_rec = pl.col("player_receptions").is_not_null() | pl.col("player_reception_yds").is_not_null()
    return (
        pl.when(has_pass)
        .then(pl.lit("QB"))
        .when(has_rush)
        .then(pl.lit("RB"))
        .when(has_rec)
        .then(pl.lit("WR"))
        .otherwise(None)
    )


def _score_prop_implied_ppr(wide: pl.DataFrame, scoring_cfg: dict) -> pl.DataFrame:
    ppr_expr = None
    priced_expr = None
    for market, rule in PROP_MARKET_TO_SCORING_RULE.items():
        pts_per_unit = scoring_cfg[rule]["points"]
        term = pl.col(market).fill_null(0.0) * pts_per_unit
        ppr_expr = term if ppr_expr is None else ppr_expr + term
        present = pl.col(market).is_not_null().cast(pl.Int64)
        priced_expr = present if priced_expr is None else priced_expr + present
    return wide.with_columns(ppr_expr.alias("prop_implied_ppr"), priced_expr.alias("markets_priced"))


def match_props_to_gsis(props_wide: pl.DataFrame, crosswalk: pl.DataFrame | None = None) -> pl.DataFrame:
    """Attach gsis_id/match_method via the position-guess ladder documented in
    ``_position_guess_expr``: match at the first guessed position, then retry
    whatever's still unmatched/ambiguous at TE for rows whose guess was WR
    (the receiving-only, WR-vs-TE-ambiguous bucket).
    """
    cw = crosswalk if crosswalk is not None else build_id_map()
    df = props_wide.with_columns(_position_guess_expr().alias("pos_guess"))

    parts = []
    for season in sorted(df.get_column("season").unique().to_list()):
        sub = df.filter(pl.col("season") == season)
        matched = match_to_gsis(sub, name_col="player_name", pos_col="pos_guess", season=season, crosswalk=cw)

        retry_mask = matched["pos_guess"].eq("WR") & matched["match_method"].is_in(["unmatched", "ambiguous"])
        retry_rows = matched.filter(retry_mask)
        if retry_rows.height > 0:
            retry_rows = retry_rows.with_columns(pl.lit("TE").alias("pos_guess"))
            retried = match_to_gsis(
                retry_rows.drop(["gsis_id", "match_method"]), name_col="player_name", pos_col="pos_guess", season=season, crosswalk=cw
            )
            retried = retried.filter(pl.col("match_method").is_in(["fantasypros_id", "exact_name", "fuzzy"]))
            if retried.height > 0:
                matched = matched.join(
                    retried.select("season", "player_name", pl.col("gsis_id").alias("_gsis_te"), pl.col("match_method").alias("_method_te")),
                    on=["season", "player_name"],
                    how="left",
                )
                matched = matched.with_columns(
                    pl.when(pl.col("_gsis_te").is_not_null()).then(pl.col("_gsis_te")).otherwise(pl.col("gsis_id")).alias("gsis_id"),
                    pl.when(pl.col("_gsis_te").is_not_null()).then(pl.col("_method_te")).otherwise(pl.col("match_method")).alias("match_method"),
                ).drop(["_gsis_te", "_method_te"])
        parts.append(matched)

    return pl.concat(parts, how="vertical_relaxed") if parts else df.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("gsis_id"), pl.lit("unmatched").alias("match_method")
    )


def build_vegas_props(raw_dir: Path = RAW_PROPS_DIR, crosswalk: pl.DataFrame | None = None) -> pl.DataFrame:
    """raw/*week1_props*.json -> season, gsis_id, player_name, prop_implied_ppr, markets_priced,
    match_method.

    A Week-1 rate proxy, not a season-long projection -- see module
    docstring's prop_implied_ppr formula. Rows that never resolve to a
    gsis_id keep gsis_id=null and match_method="unmatched"/"ambiguous"
    (never dropped, so the caller can see and report the miss rate).
    """
    lines = extract_player_market_lines(raw_dir)
    if lines.is_empty():
        return pl.DataFrame(schema=_VEGAS_PROPS_SCHEMA)

    wide = lines.pivot(index=["season", "player_name"], on="market", values="line")
    for market in PROP_MARKET_TO_SCORING_RULE:
        if market not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(market))

    scoring_cfg = load_scoring_config()
    wide = _score_prop_implied_ppr(wide, scoring_cfg)

    matched = match_props_to_gsis(wide, crosswalk=crosswalk)
    return matched.select("season", "gsis_id", "player_name", "prop_implied_ppr", "markets_priced", "match_method").sort(
        ["season", "player_name"]
    )


def run_build_vegas_props(raw_dir: Path = RAW_PROPS_DIR, out_path: Path = VEGAS_PROPS_PATH) -> pl.DataFrame:
    out = build_vegas_props(raw_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    team_out = run_build_vegas_team()
    print(f"vegas_team | wrote {VEGAS_TEAM_PATH} | {team_out.height:,} rows")
    coverage = team_out.group_by("season").agg(pl.len().alias("n_teams_priced")).sort("season")
    print(coverage)

    props_out = run_build_vegas_props()
    print(f"vegas_props | wrote {VEGAS_PROPS_PATH} | {props_out.height:,} rows")
    if props_out.height > 0:
        rates = props_out.group_by(["season", "match_method"]).agg(pl.len().alias("n")).sort(["season", "match_method"])
        print(rates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
