"""Position-agnostic feature builders (Phase 3).

Every function here returns a small, purpose-named frame keyed by either
``(season, gsis_id)`` or ``(season, team)``, built strictly from data a
model could see *before* Week 1 of the labeled season: prior-season (N-1
and earlier) production, and offseason facts (draft results, preseason
roster, Week-1 coaching staff). See the module docstring in
``src.features.wr`` for the full leakage-rule writeup and the
season-shifting convention every function below follows.

Builders take already-loaded polars frames (not paths) so tests can pass in
a doctored frame — e.g. a ``player_stats`` frame with a season stripped out
— to prove nothing downstream depends on that season's rows. Default paths
live in ``PATHS`` for callers (``__main__`` blocks) that want to load from
disk.

Team-code normalization
------------------------
``player_stats.parquet`` is the ground truth here: verified against its
own team column (season 2013-2025, no filtering), it carries **only**
present-day franchise codes — ``LA`` (Rams), ``LAC`` (Chargers), ``LV``
(Raiders) — for every season, never the historical ``STL`` / ``SD`` /
``OAK`` a fan would recognize from those years. Every other team-bearing
source disagrees with it in its own way:

- ``draft_picks.parquet`` ships a different abbreviation style entirely
  (``GNB``, ``KAN``, ``LAR``, ``LVR``, ``NOR``, ``NWE``, ``SDG``, ``SFO``,
  ``TAM``) plus era-correct legacy codes for historical drafts (``OAK``,
  ``STL``).
- ``rosters.parquet`` and ``schedules.parquet`` both use era-correct
  historical codes for the relocated franchises (``OAK`` through 2019,
  ``SD`` through 2016, ``STL``/``SL`` through 2015) rather than
  present-day ones, plus (rosters only) three seasons of legacy short
  codes for unmoved teams (``ARZ``, ``BLT``, ``CLV``, ``HST``, 2013-2015).

``normalize_team`` maps every variant — draft_picks style and era-correct
historical alike — onto player_stats' present-day set. Every cross-source
team join in this module goes through it, on **every** source
(draft_picks, rosters, schedules) before it touches a player_stats-derived
frame; skipping it on schedules in particular was an early bug here (an
un-normalized "OAK" in ``new_hc_table`` silently failed to match a
rosters-normalized "LV" row for 2013-2019 seasons) caught by the
null-rate report's team-context columns and fixed by normalizing
schedules' home/away team columns too.

Team assignment: preseason team vs. primary team
--------------------------------------------------
Two different "team" concepts are needed, from two different sources:

- ``season_roster_team`` (preseason/current-team proxy) = the player's
  season-N **Week-1** team, used for every team-*context* feature (new_hc,
  new_oc, vacated shares, team pass volume, competition draft capital).
  Sourced from ``rosters_weekly.parquet`` (week==1, REG), *not*
  ``rosters.parquet``'s season-level table — verified against the 2024
  Davante Adams trade: his season-level ``rosters.parquet`` row reads
  ``team=NYJ`` (his October trade destination — genuinely future
  information relative to Week 1), while ``rosters_weekly.parquet``'s
  week==1 row correctly reads ``team=LV``. An earlier version of this
  module used the season-level table as a "close enough" proxy and
  documented the resulting leak as a known limitation; that was wrong — a
  mid-season trade is exactly the kind of hindsight signal the leakage
  rule exists to keep out, and ``rosters_weekly.parquet`` removes the
  need for the tradeoff entirely once it's available. See
  ``season_roster_team``'s docstring for the two-level fallback (earliest
  available REG week, then season-level rosters as a last resort with
  ``preseason_team_fallback=1``) and why it's still fine relative to the
  leakage rule.

  Deliberately **not** derived from ``player_stats`` either way — a
  season-N game-log-derived team assignment (e.g. "his Week-1
  opponent") would depend on season-N game *result* rows, and the
  structural leakage test rebuilds features with season-N stripped out of
  ``player_stats`` and asserts identical output. Both
  ``rosters.parquet`` and ``rosters_weekly.parquet`` are pure roster
  facts (who's on the roster, not what happened in the game), finalized
  before kickoff — using them is not a leak the way reading season-N
  ``player_stats`` would be, which is why the structural leakage test
  perturbs only ``player_stats`` and not either rosters table.
- ``primary_team`` (prior-season production scope, sourced from
  ``player_stats``) = the team with the most REG game rows in a season
  (mode; ties broken by earliest week). Used only to scope N-1
  team-relative shares (target_share, etc.): a player traded mid-season
  gets his shares computed against the team he produced the bulk of his
  N-1 stats for, per the documented simplification in
  ``src.features.wr``'s honesty-rules note. This is intentionally
  ``player_stats``-derived — for a season-N label row it is always an
  N-1 concept (shifted by ``games_played_prior`` / used inside
  ``src.features.wr``'s share builder), so it is never read from the
  label season's own rows.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_DIR = REPO_ROOT / "configs"

PATHS = {
    "player_stats": RAW_DIR / "player_stats.parquet",
    "rosters": RAW_DIR / "rosters.parquet",
    "rosters_weekly": RAW_DIR / "rosters_weekly.parquet",
    "draft_picks": RAW_DIR / "draft_picks.parquet",
    "schedules": RAW_DIR / "schedules.parquet",
    "snap_counts": RAW_DIR / "snap_counts.parquet",
    "coaching_changes": CONFIG_DIR / "coaching_changes.csv",
}

# Every variant -> the present-day code player_stats.parquet uses
# unconditionally for every season (see module docstring). Covers both
# draft_picks' own abbreviation style and every source's era-correct
# historical code for a relocated franchise.
TEAM_ALIASES = {
    "GNB": "GB",
    "KAN": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "NOR": "NO",
    "NWE": "NE",
    "SDG": "LAC",
    "SFO": "SF",
    "TAM": "TB",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SL": "LA",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
    # pre-2013 codes retained for completeness; harmless if never matched.
    "PHO": "ARI",
    "RAI": "LV",
    "RAM": "LA",
}

# Undrafted-player sentinel (spec): one round past the real 7, and a pick
# number worse than any real pick (last real pick is in the low 260s).
UNDRAFTED_ROUND = 8
UNDRAFTED_PICK = 300


def normalize_team(df: pl.DataFrame, col: str = "team") -> pl.DataFrame:
    """Map every known legacy/alternate team code in ``col`` onto the standard set."""
    return df.with_columns(pl.col(col).replace(TEAM_ALIASES).alias(col))


def filter_reg(df: pl.DataFrame) -> pl.DataFrame:
    """Restrict a player_stats-shaped frame to REG rows with a real player_id."""
    return df.filter((pl.col("season_type") == "REG") & pl.col("player_id").is_not_null())


# --------------------------------------------------------------------------
# Age, experience, draft capital
# --------------------------------------------------------------------------


def age_and_experience(rosters: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> age (at Sept 1 of season), age_sq, years_exp.

    ``years_exp`` is read straight from rosters' own per-season-row column
    (0 in a player's rookie season, verified against known rookies e.g.
    Puka Nacua 2023 / Justin Jefferson 2020) rather than recomputed from
    ``rookie_year`` — same underlying fact, one less place to get it wrong.
    Both are pure roster metadata knowable long before Week 1.

    Birth date is looked up from *any* season row for that gsis_id (a
    handful of season-rows carry a null birth_date even though the player's
    other seasons have it) so a single missing year doesn't null out age
    for seasons that do have it.

    ``rosters`` is not strictly one row per (season, gsis_id) — a player
    who changed teams mid-season (e.g. a practice-squad claim) can carry
    two rows for the same season, one per team, with identical birth_date/
    years_exp. Deduped here (keep first) since only those two
    team-independent columns are read.
    """
    birth = (
        rosters.filter(pl.col("birth_date").is_not_null())
        .sort("season")
        .group_by("gsis_id")
        .agg(pl.col("birth_date").first())
    )
    exp = (
        rosters.select("season", "gsis_id", "years_exp")
        .filter(pl.col("gsis_id").is_not_null())
        .unique(subset=["season", "gsis_id"], keep="first")
    )
    out = exp.join(birth, on="gsis_id", how="left")
    sept1 = pl.date(pl.col("season"), 9, 1)
    age = (sept1 - pl.col("birth_date")).dt.total_days() / 365.25
    out = out.with_columns(age.alias("age"))
    out = out.with_columns((pl.col("age") ** 2).alias("age_sq"))
    return out.select("season", "gsis_id", "age", "age_sq", "years_exp")


def draft_capital(draft_picks: pl.DataFrame) -> pl.DataFrame:
    """gsis_id -> draft_round, draft_pick, log_draft_pick, undrafted (career-constant).

    Only carries drafted players; ``attach_draft_capital`` fills the
    undrafted sentinel (round 8 / pick 300) for every gsis_id absent here.
    Deduped by gsis_id (earliest draft season wins) as a defensive measure
    against a duplicated crosswalk row — draft_picks should already be
    one-row-per-player.
    """
    dp = draft_picks.filter(pl.col("gsis_id").is_not_null() & pl.col("pick").is_not_null())
    dp = normalize_team(dp, "team")
    dp = dp.sort("season").unique(subset=["gsis_id"], keep="first")
    return dp.select(
        "gsis_id",
        pl.col("round").alias("draft_round"),
        pl.col("pick").alias("draft_pick"),
        pl.col("pick").log().alias("log_draft_pick"),
        pl.lit(0).alias("undrafted"),
    )


def attach_draft_capital(df: pl.DataFrame, capital: pl.DataFrame) -> pl.DataFrame:
    """Left-join draft_capital onto ``df`` (must carry gsis_id), filling the undrafted sentinel."""
    out = df.join(capital, on="gsis_id", how="left")
    out = out.with_columns(
        pl.col("draft_round").fill_null(UNDRAFTED_ROUND),
        pl.col("draft_pick").fill_null(UNDRAFTED_PICK),
        pl.col("log_draft_pick").fill_null(math.log(UNDRAFTED_PICK)),
        pl.col("undrafted").fill_null(1),
    )
    return out


# --------------------------------------------------------------------------
# Team assignment (preseason team vs. prior-season primary team)
# --------------------------------------------------------------------------


def team_assignments(reg: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> primary_team, games — both purely N-1 (prior-production) concepts.

    ``primary_team`` is the team with the most REG game rows that season
    (mode; ties broken by earliest week), used to scope a player's N-1
    target_share/air_yards_share. ``games`` is total REG games that season
    across every team the player appeared for (matches
    ``src.labels.build.season_aggregates``' definition) — used as "games
    played" for the games-N-1 shared feature and as the denominator for
    whole-season rate stats in ``src.features.wr``.

    Deliberately does **not** carry a "current team" column — see
    ``season_roster_team`` below for why that's sourced from
    ``rosters.parquet`` instead of here.
    """
    by_team = reg.group_by(["season", "player_id", "team"]).agg(
        pl.len().alias("n_weeks"), pl.col("week").min().alias("first_week")
    )
    primary = (
        by_team.sort(["season", "player_id", "n_weeks", "first_week"], descending=[False, False, True, False])
        .group_by(["season", "player_id"], maintain_order=True)
        .first()
        .select("season", "player_id", pl.col("team").alias("primary_team"))
    )
    games = reg.group_by(["season", "player_id"]).agg(pl.len().alias("games"))
    out = primary.join(games, on=["season", "player_id"], how="left")
    return out.rename({"player_id": "gsis_id"})


def season_roster_team(rosters: pl.DataFrame, rosters_weekly: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> team (Week-1 roster), preseason_team_fallback.

    Primary source is ``rosters_weekly.parquet``, REG rows, week==1 — a
    real Week-1 snapshot, finalized before any season-N game is played
    (verified against the 2024 Davante Adams trade: week==1 correctly
    reads LV, not his October NYJ destination — see module docstring for
    why the season-level ``rosters.parquet`` table got this wrong).

    Two-level fallback, in order:

    1. If a player has no week==1 row that season (e.g. his team's
       Week-1 game was postponed — 2017 TB@MIA, Hurricane Irma — or he
       signed after Week 1), use his *earliest available* REG week that
       season instead. Still a pure preseason-or-early-season roster
       fact, not a game result.
    2. If a player has no REG row in ``rosters_weekly`` for that season at
       all (extremely rare — a late-season-only signee who never
       registered a weekly snapshot), fall back to ``rosters.parquet``'s
       season-level ``team`` column and set ``preseason_team_fallback=1``
       so that residual is visible and auditable rather than silently
       blended in. Every other row gets ``preseason_team_fallback=0``.

    A rare handful of (season, gsis_id, week) triples carry duplicate
    rows differing only in roster ``status`` (e.g. ACT + TRT the same
    week) — deduped by preferring the ``ACT`` (active) row.
    """
    reg = rosters_weekly.filter((pl.col("game_type") == "REG") & pl.col("gsis_id").is_not_null())
    reg = reg.with_columns(pl.when(pl.col("status") == "ACT").then(0).otherwise(1).alias("_status_rank"))
    reg = reg.sort(["season", "gsis_id", "week", "_status_rank"]).unique(
        subset=["season", "gsis_id", "week"], keep="first"
    )

    # Week 1 if present, else the earliest REG week available that season —
    # sorting by week ascending and keeping the first row per (season,
    # gsis_id) implements both in one pass.
    weekly_team = (
        reg.sort(["season", "gsis_id", "week"])
        .unique(subset=["season", "gsis_id"], keep="first")
        .select("season", "gsis_id", "team")
    )
    weekly_team = normalize_team(weekly_team, "team").with_columns(pl.lit(0).alias("preseason_team_fallback"))

    season_level = (
        rosters.filter(pl.col("gsis_id").is_not_null())
        .sort(["season", "gsis_id", "week"])
        .unique(subset=["season", "gsis_id"], keep="last")
        .select("season", "gsis_id", "team")
    )
    season_level = normalize_team(season_level, "team").with_columns(pl.lit(1).alias("preseason_team_fallback"))
    fallback_only = season_level.join(
        weekly_team.select("season", "gsis_id"), on=["season", "gsis_id"], how="anti"
    )

    return pl.concat([weekly_team, fallback_only], how="vertical_relaxed").select(
        "season", "gsis_id", "team", "preseason_team_fallback"
    )


def games_played_prior(team_assign: pl.DataFrame) -> pl.DataFrame:
    """season (N), gsis_id -> games_prior = games played in season N-1."""
    prior = team_assign.select("season", "gsis_id", pl.col("games").alias("games_prior"))
    return prior.with_columns((pl.col("season") + 1).alias("season"))


def team_change_flag(season_team: pl.DataFrame, team_assign: pl.DataFrame) -> pl.DataFrame:
    """season (N), gsis_id -> team_change: season_team[N] (rosters) != primary_team[N-1] (player_stats).

    Null when the player has no N-1 row at all (no games played the prior
    season) — there's no "old team" to compare against.
    """
    current = season_team.select("season", "gsis_id", "team")
    prior = team_assign.select("season", "gsis_id", pl.col("primary_team").alias("prior_team")).with_columns(
        (pl.col("season") + 1).alias("season")
    )
    out = current.join(prior, on=["season", "gsis_id"], how="left")
    out = out.with_columns(
        pl.when(pl.col("prior_team").is_null())
        .then(None)
        .otherwise((pl.col("team") != pl.col("prior_team")).cast(pl.Int64))
        .alias("team_change")
    )
    return out.select("season", "gsis_id", "team_change")


# --------------------------------------------------------------------------
# Usage totals (targets/carries), shared by target_share-style features and
# vacated-share features
# --------------------------------------------------------------------------


def player_team_usage(reg: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id, team -> targets, receptions, rec_yards, air_yards, rec_tds, carries.

    Scoped to a single team-season (a player traded mid-season gets one row
    per team he played for), the building block for both team_usage_totals
    below and target_share/air_yards_share in src.features.wr.
    """
    return reg.group_by(["season", "player_id", "team"]).agg(
        pl.col("targets").sum().alias("targets"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("receiving_yards").sum().alias("rec_yards"),
        pl.col("receiving_air_yards").sum().alias("air_yards"),
        pl.col("receiving_tds").sum().alias("rec_tds"),
        pl.col("carries").sum().alias("carries"),
    ).rename({"player_id": "gsis_id"})


def team_usage_totals(reg: pl.DataFrame) -> pl.DataFrame:
    """season, team -> team_targets, team_carries, team_air_yards, team_rush_yards (all positions, all rows).

    ``team_rush_yards`` (added for QB's rush_yard_share — see
    ``src.features.qb``) is every rusher's yards on the team, QB scrambles
    and designed runs included, same "every position, every row" scope as
    the other three totals.
    """
    return reg.group_by(["season", "team"]).agg(
        pl.col("targets").sum().alias("team_targets"),
        pl.col("carries").sum().alias("team_carries"),
        pl.col("receiving_air_yards").sum().alias("team_air_yards"),
        pl.col("rushing_yards").sum().alias("team_rush_yards"),
    )


def vacated_shares(reg: pl.DataFrame, season_team: pl.DataFrame) -> pl.DataFrame:
    """season (N), team -> vacated_target_share, vacated_carry_share.

    For team X: sum of season-N-1 target/carry share (relative to X's N-1
    team totals) belonging to every player who touched the ball for X in
    N-1 but is NOT on X's season-N roster (``season_team`` — see
    ``season_roster_team``, rosters-sourced — differs from X, including
    players who left the league entirely and have no season-N roster row
    at all).
    """
    usage = player_team_usage(reg)
    totals = team_usage_totals(reg)
    shares = usage.join(totals, on=["season", "team"], how="left")
    shares = shares.with_columns(
        (pl.col("targets") / pl.col("team_targets")).alias("player_target_share"),
        (pl.col("carries") / pl.col("team_carries")).alias("player_carry_share"),
    )

    # For each N-1 usage row, look up that player's season-N team (N-1 + 1).
    lookup = season_team.select("season", "gsis_id", pl.col("team").alias("season_n_team")).with_columns(
        (pl.col("season") - 1).alias("season")
    )
    shares = shares.join(lookup, on=["season", "gsis_id"], how="left")
    departed = shares.filter(
        (pl.col("season_n_team").is_null()) | (pl.col("season_n_team") != pl.col("team"))
    )

    out = departed.group_by(["season", "team"]).agg(
        pl.col("player_target_share").sum().alias("vacated_target_share"),
        pl.col("player_carry_share").sum().alias("vacated_carry_share"),
    )
    return out.with_columns((pl.col("season") + 1).alias("season"))


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """num / den, null (not NaN) when den is 0 or null.

    A plain ``num / den`` on a genuine 0/0 (e.g. a special-teamer with 0
    targets and 0 air yards) yields float NaN, not null — a real
    correctness bug (NaN survives null-checks, breaks null-rate reporting,
    and most downstream ML libraries treat it differently from a missing
    value). Every ratio-shaped feature across ``src.features.wr``/``rb``/
    ``te``/``qb`` goes through this (WR keeps its own private copy,
    predating this shared one; behavior is identical).
    """
    return pl.when(den == 0).then(None).otherwise(num / den)


def reg_with_points(player_stats: pl.DataFrame, profile: dict) -> pl.DataFrame:
    """REG rows + computed_points, using the src.labels.build scoring formula (imported, not duplicated).

    Position-agnostic — every position's ppr_ppg feature starts here (WR
    keeps its own private copy, predating this shared one; behavior is
    identical).
    """
    from src.labels.build import compute_weekly_points

    reg = filter_reg(player_stats)
    return reg.with_columns(compute_weekly_points(reg, profile).alias("computed_points"))


def expected_points_total(ff_opportunity: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> expected_total: sum of ff_opportunity's total_fantasy_points_exp, REG only.

    Position-agnostic: ``total_fantasy_points_exp`` already blends pass +
    rush + rec expected points for whichever of those a player's role
    involves, so this one aggregation serves every position's expected-PPR
    feature (WR keeps its own private copy, predating this shared one;
    behavior is identical). ff_opportunity ships no season_type column;
    REG rows are identified by joining game_id against schedules (verified:
    every ff_opportunity game_id resolves to a schedules game_type, no
    unmatched rows).
    """
    ff = ff_opportunity.with_columns(pl.col("season").cast(pl.Int64))
    reg_games = schedules.filter(pl.col("game_type") == "REG").select("game_id")
    ff_reg = ff.join(reg_games, on="game_id", how="inner").filter(pl.col("player_id").is_not_null())
    agg = ff_reg.group_by(["season", "player_id"]).agg(
        pl.col("total_fantasy_points_exp").sum().alias("expected_total")
    )
    return agg.rename({"player_id": "gsis_id"})


def snap_share_from_counts(snap_counts: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> snap_share: mean REG-season offense_pct.

    Position-agnostic (WR keeps its own private copy, predating this
    shared one; behavior is identical). snap_counts carries no gsis_id,
    only pfr_player_id; resolved via rosters' own (season, pfr_id, gsis_id)
    crosswalk. Rows that don't resolve (older/deeper-roster players with no
    pfr_id on file) are dropped here, not guessed — they simply stay null
    downstream.
    """
    reg = snap_counts.filter(pl.col("game_type") == "REG")
    xwalk = (
        rosters.filter(pl.col("pfr_id").is_not_null())
        .select("season", "pfr_id", "gsis_id")
        .unique()
    )
    j = reg.join(xwalk, left_on=["season", "pfr_player_id"], right_on=["season", "pfr_id"], how="left")
    j = j.filter(pl.col("gsis_id").is_not_null())
    return j.group_by(["season", "gsis_id"]).agg(pl.col("offense_pct").mean().alias("snap_share"))


# --------------------------------------------------------------------------
# Team pass / rush volume (N-1)
# --------------------------------------------------------------------------


def team_pass_volume_prior(reg: pl.DataFrame) -> pl.DataFrame:
    """season (N), team -> team_pass_att_pg, team_plays_pg, team_pass_rate, from season N-1.

    Built from player_stats weekly passing/rushing rows (no team_stats
    table ships in data/raw): a "pass play" is a dropback (attempts +
    sacks_suffered, i.e. including sacks, standard offensive-play-rate
    convention); a "play" adds rushing carries. Rates are season totals
    (sum / sum), not an average of weekly rates, so a bye week doesn't
    distort them.
    """
    weekly = reg.group_by(["season", "team", "week"]).agg(
        pl.col("attempts").sum().alias("att"),
        pl.col("sacks_suffered").sum().alias("sacks"),
        pl.col("carries").sum().alias("rush"),
    )
    season_tot = weekly.group_by(["season", "team"]).agg(
        pl.len().alias("n_games"),
        pl.col("att").sum().alias("att_sum"),
        pl.col("sacks").sum().alias("sacks_sum"),
        pl.col("rush").sum().alias("rush_sum"),
    )
    season_tot = season_tot.with_columns(
        (pl.col("att_sum") / pl.col("n_games")).alias("team_pass_att_pg_prior"),
        ((pl.col("att_sum") + pl.col("sacks_sum") + pl.col("rush_sum")) / pl.col("n_games")).alias(
            "team_plays_pg_prior"
        ),
        (
            (pl.col("att_sum") + pl.col("sacks_sum"))
            / (pl.col("att_sum") + pl.col("sacks_sum") + pl.col("rush_sum"))
        ).alias("team_pass_rate_prior"),
    )
    out = season_tot.select(
        "season", "team", "team_pass_att_pg_prior", "team_plays_pg_prior", "team_pass_rate_prior"
    )
    return out.with_columns((pl.col("season") + 1).alias("season"))


def team_rush_volume_prior(reg: pl.DataFrame) -> pl.DataFrame:
    """season (N), team -> team_rush_att_pg_prior, from season N-1.

    RB's analogue of ``team_pass_volume_prior`` above: season-total carries
    (every position — a team's rush attempts include QB scrambles/designed
    runs, not just RB carries) divided by season-total games, then shifted
    N-1 -> N. Same "sum / sum, not mean-of-weekly-rates" convention so a
    bye week doesn't distort it.
    """
    weekly = reg.group_by(["season", "team", "week"]).agg(pl.col("carries").sum().alias("rush"))
    season_tot = weekly.group_by(["season", "team"]).agg(
        pl.len().alias("n_games"), pl.col("rush").sum().alias("rush_sum")
    )
    season_tot = season_tot.with_columns(
        (pl.col("rush_sum") / pl.col("n_games")).alias("team_rush_att_pg_prior")
    )
    out = season_tot.select("season", "team", "team_rush_att_pg_prior")
    return out.with_columns((pl.col("season") + 1).alias("season"))


# --------------------------------------------------------------------------
# Competition draft capital
# --------------------------------------------------------------------------


def competition_draft_capital(draft_picks: pl.DataFrame, position: str | list[str]) -> pl.DataFrame:
    """season (N), team -> competition_draft_capital: sum(max(0, 300 - pick)) for `position` picks.

    "Competition" a returning player faces from his own team's season-N
    draft class at his position — entirely a preseason (draft-day) fact.

    ``position`` accepts either a single position code (WR's own-position
    call) or a list (QB's supporting_cast_capital: draft capital the
    season-N team spent on WR/TE/RB, i.e. every *other* skill position) —
    same underlying formula either way, ``.is_in`` subsumes ``==``.
    """
    positions = [position] if isinstance(position, str) else list(position)
    dp = draft_picks.filter(pl.col("position").is_in(positions) & pl.col("pick").is_not_null())
    dp = normalize_team(dp, "team")
    dp = dp.with_columns((pl.lit(UNDRAFTED_PICK) - pl.col("pick")).clip(lower_bound=0).alias("capital"))
    return dp.group_by(["season", "team"]).agg(pl.col("capital").sum().alias("competition_draft_capital"))


# --------------------------------------------------------------------------
# Coaching
# --------------------------------------------------------------------------


def new_hc_table(schedules: pl.DataFrame) -> pl.DataFrame:
    """season (N), team -> new_hc: Week-1 REG coach differs from every REG coach that team had in N-1.

    Built entirely from public, pre-Week-1-knowable facts (who is
    coaching the season opener) compared against the prior season's full
    coaching record (covers in-season interim-coach changes too — an
    interim coach who ran the back half of N-1 counts as "a coach that
    team had in N-1").
    """
    reg = schedules.filter(pl.col("game_type") == "REG")
    home = normalize_team(
        reg.select("season", "week", pl.col("home_team").alias("team"), pl.col("home_coach").alias("coach")),
        "team",
    )
    away = normalize_team(
        reg.select("season", "week", pl.col("away_team").alias("team"), pl.col("away_coach").alias("coach")),
        "team",
    )
    long = pl.concat([home, away], how="vertical_relaxed").filter(pl.col("coach").is_not_null())

    week1_coach = long.filter(pl.col("week") == 1).select("season", "team", pl.col("coach").alias("week1_coach"))
    prior_coaches = (
        long.group_by(["season", "team"])
        .agg(pl.col("coach").unique().alias("coaches"))
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({"coaches": "prior_coaches"})
    )

    out = week1_coach.join(prior_coaches, on=["season", "team"], how="left")
    out = out.with_columns(
        pl.when(pl.col("prior_coaches").is_null())
        .then(None)
        .otherwise(
            ~pl.col("prior_coaches").list.contains(pl.col("week1_coach"))
        )
        .cast(pl.Int64)
        .alias("new_hc")
    )
    return out.select("season", "team", "new_hc")


def new_oc_table(coaching_changes: pl.DataFrame | None) -> pl.DataFrame:
    """season, team -> new_oc (0/1), nullable.

    ``coaching_changes`` lists only teams that GOT a new OC/play-caller
    that season. Per configs/coaching_changes.csv's own header: if a
    season appears anywhere in the file, every unlisted team that season
    is new_oc=0; a season entirely absent from the file (currently
    2014-2018) is new_oc=null for every team (no signal collected yet, not
    "no change"). ``coaching_changes=None`` (file not seeded at all) yields
    an all-null-safe empty table — every join against it produces null.
    """
    if coaching_changes is None or coaching_changes.is_empty():
        return pl.DataFrame(
            schema={"season": pl.Int64, "team": pl.Utf8, "new_oc": pl.Int64}
        )
    cc = coaching_changes.with_columns(pl.col("new_oc").cast(pl.Int64))
    return cc.select("season", "team", "new_oc")


def attach_new_oc(df: pl.DataFrame, new_oc: pl.DataFrame, season_col: str = "season", team_col: str = "team") -> pl.DataFrame:
    """Left-join new_oc_table onto df, applying the "season present -> unlisted is 0" rule."""
    seasons_in_file = set(new_oc.get_column("season").unique().to_list()) if not new_oc.is_empty() else set()
    out = df.join(
        new_oc.rename({"season": season_col, "team": team_col}), on=[season_col, team_col], how="left"
    )
    out = out.with_columns(
        pl.when(pl.col("new_oc").is_not_null())
        .then(pl.col("new_oc"))
        .when(pl.col(team_col).is_not_null() & pl.col(season_col).is_in(list(seasons_in_file)))
        .then(0)
        .otherwise(None)
        .alias("new_oc")
    )
    return out


def load_coaching_changes(path: Path = PATHS["coaching_changes"]) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_csv(path, comment_prefix="#")


# --------------------------------------------------------------------------
# COVID-2020 flag
# --------------------------------------------------------------------------


def add_covid_flag(df: pl.DataFrame, season_col: str = "season") -> pl.DataFrame:
    """label_season_2020: 1 if the label season N is 2020 OR its N-1 stats season is 2020 (N==2021).

    Both label seasons carry a COVID-disrupted signal: 2020 itself
    (16-game, empty-stadium, opt-outs) is the label season, and 2021's
    N-1-derived features are built entirely from that disrupted 2020 data.
    One flag covers both, per spec.
    """
    return df.with_columns(
        (pl.col(season_col).is_in([2020, 2021])).cast(pl.Int64).alias("label_season_2020")
    )
