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
    # rosters.parquet uses "AZ" for Arizona starting with the 2026 season
    # snapshot (every other season, and every other 2026 source --
    # schedules, draft_picks, player_stats -- uses "ARI"); verified directly
    # against the cached 2026 data while building Phase 6.
    "AZ": "ARI",
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
    """season, gsis_id, team -> targets, receptions, rec_yards, air_yards, rec_tds, rush_tds,

    carries, attempts. Scoped to a single team-season (a player traded mid-season gets one
    row per team he played for), the building block for both team_usage_totals below and
    target_share/air_yards_share in src.features.wr. ``rush_tds``/``attempts`` (v2.4) feed
    the vacated_td_share / qb_continuity team-context builders below -- same "every
    position, every row" scope as every other column here (a caller filters ``reg`` to one
    position first when it wants a position-scoped numerator, e.g. QB attempts for
    qb_continuity).
    """
    return reg.group_by(["season", "player_id", "team"]).agg(
        pl.col("targets").sum().alias("targets"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("receiving_yards").sum().alias("rec_yards"),
        pl.col("receiving_air_yards").sum().alias("air_yards"),
        pl.col("receiving_tds").sum().alias("rec_tds"),
        pl.col("rushing_tds").sum().alias("rush_tds"),
        pl.col("carries").sum().alias("carries"),
        pl.col("attempts").sum().alias("attempts"),
    ).rename({"player_id": "gsis_id"})


def team_usage_totals(reg: pl.DataFrame) -> pl.DataFrame:
    """season, team -> team_targets, team_carries, team_air_yards, team_rush_yards,

    team_attempts, team_off_tds (all positions, all rows).

    ``team_rush_yards`` (added for QB's rush_yard_share — see
    ``src.features.qb``) is every rusher's yards on the team, QB scrambles
    and designed runs included, same "every position, every row" scope as
    the other three totals. ``team_attempts`` (v2.4, qb_continuity) and
    ``team_off_tds`` (v2.4, vacated_td_share -- rushing + receiving TDs,
    every scorer) follow the identical whole-team convention.
    """
    return reg.group_by(["season", "team"]).agg(
        pl.col("targets").sum().alias("team_targets"),
        pl.col("carries").sum().alias("team_carries"),
        pl.col("receiving_air_yards").sum().alias("team_air_yards"),
        pl.col("rushing_yards").sum().alias("team_rush_yards"),
        pl.col("attempts").sum().alias("team_attempts"),
        (pl.col("receiving_tds").sum() + pl.col("rushing_tds").sum()).alias("team_off_tds"),
    )


# Decimal places every *SUM* of already-computed float shares (never a MAX, and never a
# sum of raw integer counts -- both exact/order-independent already) is rounded to on the
# way out. Diagnosed this session (v2.4): polars' group_by().agg(.sum()) over floats is
# NOT guaranteed bit-identical across repeated process runs on identical input --
# floating-point addition is non-associative, and polars' parallel/vectorized reduction
# order isn't pinned to a fixed thread/chunk schedule, so a share value that logically
# should read exactly 0.1607142857142857 on every run can differ in the ~15th significant
# digit run to run. Harmless for the share's own meaning (12 decimal places is far past
# any real precision this business quantity carries) but NOT harmless for anything that
# groups by exact float equality downstream -- confirmed via
# ``src.dashboard.build.feature_profiles``'s ``rank(pct=True)`` (tied players' shares seen
# as equal on one run, no longer tied on the next, shifting displayed percentiles) is
# exactly what broke ``tests/test_dashboard.py::test_build_twice_is_byte_identical``
# after ``vacated_td_share`` started shipping as a live SHAP feature. Verified BOTH this
# new column and the pre-existing ``vacated_shares``/``returning_incumbent_share`` (same
# group_by+sum shape) reproduce the jitter on two back-to-back calls with identical
# inputs -- this is a systemic, pre-existing property of the aggregation pattern, not
# something unique to the new columns; every SUM-shaped vacated_*/returning_incumbent_share
# builder below rounds its output for this reason.
_FLOAT_SUM_ROUND_DP = 12


def _departed_player_rows(shares: pl.DataFrame, season_team: pl.DataFrame) -> pl.DataFrame:
    """Given a per-(season, team, gsis_id) N-1 usage-share frame (unshifted), attach

    ``season_n_team`` (the player's season-N team, per ``season_roster_team``) and filter
    to rows where the player is NOT on that (season, team)'s season-N roster -- i.e.
    "departed" (including a player who left the league entirely and has no season-N
    roster row at all). The shared "who left this team" join every ``vacated_*``
    team-context builder below reuses (``vacated_shares``' own target/carry columns,
    ``vacated_td_share_table``, ``vacated_goal_line_carry_share_table``,
    ``max_single_vacated_share``) -- computed exactly once so a future vacated-* feature
    doesn't re-derive this join a fourth way. Does NOT apply the N-1 -> N season shift --
    callers do that themselves after aggregating, matching ``vacated_shares``' own
    pre-v2.4 convention.
    """
    lookup = season_team.select("season", "gsis_id", pl.col("team").alias("season_n_team")).with_columns(
        (pl.col("season") - 1).alias("season")
    )
    joined = shares.join(lookup, on=["season", "gsis_id"], how="left")
    return joined.filter((pl.col("season_n_team").is_null()) | (pl.col("season_n_team") != pl.col("team")))


def _target_carry_player_shares(reg: pl.DataFrame) -> pl.DataFrame:
    """season, team, gsis_id -> player_target_share, player_carry_share: each player's N-1

    team-relative target/carry share (plain division, matching ``vacated_shares``' original
    formula exactly -- NOT ``safe_div``, so a genuine 0/0 team-total stays float NaN here,
    unchanged pre-v2.4 behavior). Shared building block for ``vacated_shares`` (sums this
    over the departed population) and ``max_single_vacated_share`` (takes the max instead).
    """
    usage = player_team_usage(reg)
    totals = team_usage_totals(reg)
    shares = usage.join(totals, on=["season", "team"], how="left")
    return shares.with_columns(
        (pl.col("targets") / pl.col("team_targets")).alias("player_target_share"),
        (pl.col("carries") / pl.col("team_carries")).alias("player_carry_share"),
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
    shares = _target_carry_player_shares(reg)
    departed = _departed_player_rows(shares, season_team)

    out = departed.group_by(["season", "team"]).agg(
        pl.col("player_target_share").sum().round(_FLOAT_SUM_ROUND_DP).alias("vacated_target_share"),
        pl.col("player_carry_share").sum().round(_FLOAT_SUM_ROUND_DP).alias("vacated_carry_share"),
    )
    return out.with_columns((pl.col("season") + 1).alias("season"))


def max_single_vacated_share(reg: pl.DataFrame, season_team: pl.DataFrame) -> pl.DataFrame:
    """season (N), team -> max_single_vacated_target_share, max_single_vacated_carry_share (v2.4).

    ``vacated_shares``' MAX instead of SUM: the single largest departed player's own N-1
    target/carry share, not the whole departed population's combined share. Distinguishes a
    star departure (one big number) from diffuse churn (several small departures summing to
    the same ``vacated_*_share`` total) -- a signal ``vacated_shares`` alone can't carry,
    since summing loses which departure did the work. Built on the identical
    ``_target_carry_player_shares``/``_departed_player_rows`` machinery ``vacated_shares``
    uses (same departed population, same per-player shares), so the two are always
    consistent with each other (max <= sum for any team by construction). Null wherever a
    team had no departed players with any usage at all (no row to take a max over), same
    "nullable, not guessed" contract as every other vacated-* column.
    """
    shares = _target_carry_player_shares(reg)
    departed = _departed_player_rows(shares, season_team)

    out = departed.group_by(["season", "team"]).agg(
        pl.col("player_target_share").max().alias("max_single_vacated_target_share"),
        pl.col("player_carry_share").max().alias("max_single_vacated_carry_share"),
    )
    return out.with_columns((pl.col("season") + 1).alias("season"))


def vacated_td_share_table(reg: pl.DataFrame, season_team: pl.DataFrame) -> pl.DataFrame:
    """season (N), team -> vacated_td_share (v2.4).

    Share of the team's N-1 offensive TDs (rushing + receiving, ``team_off_tds`` --
    ``team_usage_totals``) belonging to players absent from the Week-1 season-N roster
    (the ``_departed_player_rows`` population ``vacated_shares`` also uses). E.g. a goal-
    line back or a team's top red-zone receiver departing directly frees up TD equity for
    whoever inherits the touches, independent of whether the *volume* (targets/carries)
    share moved by the same amount -- a player can absorb a departed teammate's touches
    without absorbing his scoring rate. ``safe_div`` (not plain division, unlike
    ``vacated_shares``): a genuine team with zero N-1 offensive TDs is a real, if rare,
    0/0 that should read as null, not NaN.
    """
    usage = player_team_usage(reg)
    totals = team_usage_totals(reg)
    shares = usage.join(totals, on=["season", "team"], how="left")
    shares = shares.with_columns(
        safe_div(pl.col("rec_tds") + pl.col("rush_tds"), pl.col("team_off_tds")).alias("player_td_share")
    )
    departed = _departed_player_rows(shares, season_team)

    out = departed.group_by(["season", "team"]).agg(
        pl.col("player_td_share").sum().round(_FLOAT_SUM_ROUND_DP).alias("vacated_td_share")
    )
    return out.with_columns((pl.col("season") + 1).alias("season"))


# yardline_100 threshold for the goal-line vacated-carry-share feature below -- same
# "<=5" definition src.features.rb.REDZONE_THRESHOLDS' own goal_line_carry_share uses.
GOAL_LINE_THRESHOLD = 5


def vacated_goal_line_carry_share_table(pbp: pl.DataFrame, season_team: pl.DataFrame) -> pl.DataFrame:
    """season (N), team -> vacated_goal_line_carry_share (v2.4).

    The pbp-derived, goal-line-scoped (yardline_100<=5) analogue of ``vacated_shares``'
    ``vacated_carry_share``: share of the team's N-1 goal-line carries belonging to players
    absent from that team's Week-1 season-N roster (the identical ``_departed_player_rows``
    population every other vacated_* builder uses) -- e.g. a goal-line back departing frees
    up short-yardage carries for whoever's left in the backfield, a signal
    ``vacated_carry_share`` (which covers ALL carries, not just goal-line ones) can miss
    entirely if the departed back was a committee/passing-down specialist everywhere else
    on the field but the team's primary short-yardage option. Same slim-pbp-cache
    filtering ``redzone_share_table`` uses (``rusher_player_id``/``yardline_100``/
    ``posteam``, REG only); a team-season with zero qualifying goal-line carries at all
    that season is a real, if rare, 0/0 that reads as null via ``safe_div``, never NaN.
    """
    reg = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("rush_attempt") == 1)
        & pl.col("rusher_player_id").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & (pl.col("yardline_100") <= GOAL_LINE_THRESHOLD)
        & pl.col("posteam").is_not_null()
    )
    player_n = (
        reg.group_by(["season", "rusher_player_id", "posteam"])
        .agg(pl.len().alias("n"))
        .rename({"rusher_player_id": "gsis_id", "posteam": "team"})
    )
    team_n = reg.group_by(["season", "posteam"]).agg(pl.len().alias("team_n")).rename({"posteam": "team"})
    shares = player_n.join(team_n, on=["season", "team"], how="left")
    shares = shares.with_columns(safe_div(pl.col("n"), pl.col("team_n")).alias("player_gl_carry_share"))

    departed = _departed_player_rows(shares, season_team)
    out = departed.group_by(["season", "team"]).agg(
        pl.col("player_gl_carry_share").sum().round(_FLOAT_SUM_ROUND_DP).alias("vacated_goal_line_carry_share")
    )
    return out.with_columns((pl.col("season") + 1).alias("season"))


def qb_continuity_table(reg: pl.DataFrame, season_team: pl.DataFrame) -> pl.DataFrame:
    """season (N), team -> qb_continuity (v2.4).

    Fraction of the team's N-1 pass attempts (``team_attempts`` -- every attempt thrown by
    the team, ``team_usage_totals``) thrown by QBs who are STILL on that team's Week-1
    season-N roster -- ``returning_incumbent_share_table``'s exact "stayed" construction
    (position="QB", usage_col="attempts", team_total_col="team_attempts"), the mirror
    image of ``vacated_shares``' "departed" half for the same underlying question (who's
    still throwing the ball for this team). A new-starting-QB situation (the old starter
    left, nobody who threw meaningfully for this team last year is still around) reads as
    LOW; an offense running back its same signal-caller (or a backup who saw real N-1
    snaps) reads as HIGH. QB rows themselves skip this column entirely (see
    ``src.features.qb``'s module docstring) -- a QB's own continuity with himself is not a
    meaningful signal for the thrower; it is wired into WR/TE/RB only, where it answers
    "how much does this pass-catcher's team's *passing infrastructure* carry over."
    """
    out = returning_incumbent_share_table(reg, season_team, position="QB", usage_col="attempts", team_total_col="team_attempts")
    return out.rename({"returning_incumbent_share": "qb_continuity"})


# Every v2.4 vacancy/continuity column, by which position family carries it (see each
# builder's docstring above). WR/TE/RB get all five; QB gets only vacated_td_share (a
# team-context fact, not thrower-specific) -- qb_continuity is deliberately skipped for
# QB rows themselves (meaningless for the thrower, per the brief), and
# max_single_vacated_share / vacated_goal_line_carry_share have no QB-specific meaning
# the brief defines. Both gated (src.models.vacancy_gate) before shipping in any
# position's actual model -- see configs/model_{pos}.yaml's excluded_features.
VACANCY_COLUMNS_WR_TE_RB = [
    "qb_continuity",
    "vacated_td_share",
    "vacated_goal_line_carry_share",
    "max_single_vacated_target_share",
    "max_single_vacated_carry_share",
]
VACANCY_COLUMNS_QB = ["vacated_td_share"]


def attach_wr_te_rb_vacancy_features(
    out: pl.DataFrame, reg: pl.DataFrame, season_team: pl.DataFrame, pbp: pl.DataFrame | None
) -> pl.DataFrame:
    """Join every VACANCY_COLUMNS_WR_TE_RB column onto ``out`` (must carry season, team) --

    identical wiring for WR/TE/RB (see each module's ``build_features_*`` call site),
    factored out here so the join order / pbp-absent null-fallback behavior is defined
    exactly once rather than copy-pasted three times. QB does not use this -- see
    ``VACANCY_COLUMNS_QB``/``src.features.qb`` (only ``vacated_td_share`` applies there,
    wired directly at that module's own call site).
    """
    out = out.join(qb_continuity_table(reg, season_team), on=["season", "team"], how="left")
    out = out.join(vacated_td_share_table(reg, season_team), on=["season", "team"], how="left")
    out = out.join(max_single_vacated_share(reg, season_team), on=["season", "team"], how="left")
    if pbp is not None:
        out = out.join(vacated_goal_line_carry_share_table(pbp, season_team), on=["season", "team"], how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("vacated_goal_line_carry_share"))
    return out


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
# Red-zone / goal-line shares (v1.5, Phase C) -- built from play-by-play
# (data/raw/pbp.parquet, src.ingest.nflverse's slim 9-column pull; see
# configs/data.yaml's `pbp` entry). v1 skipped these entirely (no pbp
# pulled at all -- see README's "Play-by-play is deliberately not pulled"
# note); pbp is `required: false` and genuinely absent for 2026 (no games
# played yet) and for any environment that hasn't run the ingest, so
# every caller of this function must tolerate ``pbp=None`` (see
# ``src.features.wr.build_raw_stat_table`` / ``src.features.rb`` 's own
# wiring) -- this function itself is never called with None; callers
# branch to an all-null column set instead, matching every other optional
# source in this module (snap_counts, ngs_*).
# --------------------------------------------------------------------------


def redzone_share_table(
    pbp: pl.DataFrame, team_assign: pl.DataFrame, *, play_col: str, player_col: str, thresholds: dict[str, int]
) -> pl.DataFrame:
    """season, gsis_id -> one column per (name, threshold) in ``thresholds``: the player's

    share of his N-1 **primary team's** own red-zone-or-tighter usage inside that yardline
    threshold -- e.g. ``rz_target_share`` (yardline_100<=20) is the player's red-zone targets
    divided by his team's total red-zone targets that season, not the player's own share of
    his season-total targets. Both numerator and denominator use the identical threshold, so
    a tighter (e.g. <=10, "end-zone-adjacent") name captures a different, tighter-usage slice
    of the same underlying concept, not just a rescaled version of the wider one.

    ``play_col`` is ``pass_attempt`` (WR/TE, ``player_col="receiver_player_id"``) or
    ``rush_attempt`` (RB, ``player_col="rusher_player_id"``). ``pbp`` must carry
    ``season``, ``season_type``, ``posteam``, ``yardline_100`` plus ``play_col``/``player_col``
    -- exactly the slim 9-column set ``configs/data.yaml``'s ``pbp`` entry pulls.

    Scoped to the N-1 primary team the same way ``src.features.wr``'s ``_shares`` scopes
    target_share/air_yards_share (see that function + ``team_assignments``' docstring) — a
    player traded mid-season gets this computed against the team he produced the bulk of his
    snaps for, not a per-week blend. A player who was on the primary team but had zero
    qualifying plays inside a given threshold gets an explicit share of 0.0 (not null) — a
    real "he wasn't targeted/didn't carry inside the red zone," not missing data; a share is
    null only when the team itself had zero qualifying plays at that threshold all season
    (the 0/0 case, vanishingly rare for these thresholds but handled via ``safe_div``).
    """
    reg = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col(play_col) == 1)
        & pl.col(player_col).is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("posteam").is_not_null()
    )
    out: pl.DataFrame | None = None
    for name, threshold in thresholds.items():
        sub = reg.filter(pl.col("yardline_100") <= threshold)
        player_n = (
            sub.group_by(["season", player_col, "posteam"])
            .agg(pl.len().alias("n"))
            .rename({player_col: "gsis_id", "posteam": "team"})
        )
        team_n = sub.group_by(["season", "posteam"]).agg(pl.len().alias("team_n")).rename({"posteam": "team"})

        primary = team_assign.select("season", "gsis_id", pl.col("primary_team").alias("team"))
        scoped = primary.join(player_n, on=["season", "gsis_id", "team"], how="left")
        scoped = scoped.join(team_n, on=["season", "team"], how="left")
        scoped = scoped.with_columns(pl.col("n").fill_null(0))
        scoped = scoped.with_columns(safe_div(pl.col("n"), pl.col("team_n")).alias(name))
        scoped = scoped.select("season", "gsis_id", name)

        out = scoped if out is None else out.join(scoped, on=["season", "gsis_id"], how="left")

    assert out is not None, "redzone_share_table called with an empty thresholds dict"
    return out


def empty_redzone_share_table(names: list[str]) -> pl.DataFrame:
    """season, gsis_id -> all-null columns for every name in ``names`` -- pbp-absent fallback,

    same shape ``redzone_share_table`` would produce, matching every other optional-source
    fallback in this module (snap_counts/ngs_* -> null columns, never a raised error).
    """
    schema = {"season": pl.Int64, "gsis_id": pl.Utf8, **{n: pl.Float64 for n in names}}
    return pl.DataFrame(schema=schema)


# --------------------------------------------------------------------------
# Efficiency proxies (v2.2, inference-safe YPRR substitutes) -- true
# yards-per-route-run is impossible to compute (route-participation data is
# dead after 2023, see README's Known Limitations), so these approximate
# "how efficient is this player with the offensive snaps/targets he
# actually gets" from data that IS still fresh:
#   - yards_per_snap / targets_per_snap: season yards-or-targets divided by
#     that player-season's total offensive snaps (``snap_counts.parquet``,
#     resolved to gsis_id via the same pfr_id crosswalk
#     ``snap_share_from_counts`` uses). Nullable wherever the snap
#     crosswalk doesn't resolve -- never guessed.
#   - catchable_target_rate: share of a player's targeted pass plays FTN
#     charted as a catchable ball (``data/raw/ftn_charting.parquet``,
#     2022+ only -- FTN's own charting-coverage start), joined onto pbp by
#     play id (see ``ftn_catchable_target_rate_table``'s docstring for the
#     exact join keys). Nullable pre-2022 and wherever FTN has no coverage
#     for that game.
# Both go through the capacity gate (``src.models.efficiency_gate``) before
# shipping in any position's model: kept only if they don't regress
# holdout top-10 precision and don't regress PR-AUC by more than 0.01 at
# FROZEN hyperparameters, per position.
# --------------------------------------------------------------------------


def season_offense_snaps_table(snap_counts: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> offense_snaps: total REG-season offensive snaps played.

    Same pfr_player_id -> gsis_id resolution ``snap_share_from_counts`` uses (rosters' own
    (season, pfr_id, gsis_id) crosswalk, deduped). A row with no crosswalk match is dropped
    here, not guessed -- joining a numerator against this table then produces null (never
    0) for that player-season, a real "no snap count on file," not "played zero snaps."
    """
    reg = snap_counts.filter(pl.col("game_type") == "REG")
    xwalk = (
        rosters.filter(pl.col("pfr_id").is_not_null())
        .select("season", "pfr_id", "gsis_id")
        .unique()
    )
    j = reg.join(xwalk, left_on=["season", "pfr_player_id"], right_on=["season", "pfr_id"], how="left")
    j = j.filter(pl.col("gsis_id").is_not_null())
    return j.group_by(["season", "gsis_id"]).agg(pl.col("offense_snaps").sum().alias("offense_snaps"))


def per_snap_rate_table(
    season_totals: pl.DataFrame,
    snap_counts: pl.DataFrame,
    rosters: pl.DataFrame,
    *,
    numerator_col: str,
    out_col: str,
) -> pl.DataFrame:
    """season, gsis_id -> {out_col}: season_totals[numerator_col] / that player-season's

    total offensive snaps (``season_offense_snaps_table``). ``season_totals`` must carry
    ``season``, ``gsis_id``, ``numerator_col`` (a season-level SUM, e.g. season receiving
    yards or season targets -- not a per-game rate). Null wherever either side is missing
    or the snap total is 0 (``safe_div``).
    """
    snaps = season_offense_snaps_table(snap_counts, rosters)
    out = season_totals.select("season", "gsis_id", numerator_col).join(snaps, on=["season", "gsis_id"], how="left")
    out = out.with_columns(safe_div(pl.col(numerator_col), pl.col("offense_snaps")).alias(out_col))
    return out.select("season", "gsis_id", out_col)


# FTN's charting coverage begins here (configs/data.yaml's `ftn_charting` entry:
# first_available: 2022) -- catchable_target_rate is structurally null before this season,
# same "nullable, not guessed" contract as every other optional-source proxy in this module.
FTN_CHARTING_FIRST_SEASON = 2022


def ftn_catchable_target_rate_table(ftn_charting: pl.DataFrame, pbp_full: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> catchable_target_rate: share of a player's TARGETED pass plays

    FTN charted as ``is_catchable_ball`` (2022+ only -- FTN_CHARTING_FIRST_SEASON).

    Join keys (verified directly against this build's cached files: every 2023 FTN row
    matches exactly one pbp play, 48,225/48,225): FTN's own
    (``nflverse_game_id``, ``nflverse_play_id``) == pbp's (``game_id``, ``play_id``) --
    ``pbp_full`` needs those two columns on top of the slim 9-column pull
    ``src.features.shared.redzone_share_table`` uses; see ``configs/data.yaml``'s ``pbp``
    entry's v2.2 `select_columns` addition. The player is identified by pbp's own
    ``receiver_player_id``, which IS already a gsis_id (same as ``redzone_share_table`` --
    no separate crosswalk needed here, unlike snap_counts' pfr_id).

    A play with no charted receiver (``receiver_player_id`` null -- a run, spike, or
    untagged throwaway) is excluded from both numerator and denominator; a player-season
    with zero charted targets that year has no row in this table at all (left-joins
    against it null out, never zero -- "we don't know," not "he wasn't catchable").
    """
    ftn = ftn_charting.select(
        "season",
        "nflverse_game_id",
        pl.col("nflverse_play_id").cast(pl.Int32).alias("nflverse_play_id"),
        "is_catchable_ball",
    )
    pbp_keys = pbp_full.select(
        pl.col("game_id"),
        pl.col("play_id").cast(pl.Int32),
        "receiver_player_id",
    ).filter(pl.col("receiver_player_id").is_not_null())
    joined = ftn.join(
        pbp_keys, left_on=["nflverse_game_id", "nflverse_play_id"], right_on=["game_id", "play_id"], how="inner"
    )
    out = joined.group_by(["season", "receiver_player_id"]).agg(
        pl.col("is_catchable_ball").cast(pl.Float64).mean().alias("catchable_target_rate")
    )
    return out.rename({"receiver_player_id": "gsis_id"}).select("season", "gsis_id", "catchable_target_rate")


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


# --------------------------------------------------------------------------
# Vegas team implied points / win probability (v1.5, Phase C -- Deliverable 3)
# built from src.ingest.vegas.build_vegas_team's output
# (data/processed/vegas_team.parquet). Same "optional source -> empty,
# not raise" contract as new_oc_table/snap_counts/ngs_*: vegas_team=None
# (file not built yet in this environment) yields an all-null-safe empty
# table, and attach_vegas_team fills has_vegas=0 -- never null -- for every
# row with no matching (season, team) priced row, whatever the reason
# (pre-2020, a team/season the snapshot never priced, or the source file
# absent entirely).
# --------------------------------------------------------------------------


def vegas_team_features(vegas_team: pl.DataFrame | None) -> pl.DataFrame:
    """season, team -> implied_ppg, implied_win_prob, has_vegas."""
    if vegas_team is None or vegas_team.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "team": pl.Utf8,
                "implied_ppg": pl.Float64,
                "implied_win_prob": pl.Float64,
                "has_vegas": pl.Int64,
            }
        )
    return vegas_team.select("season", "team", "implied_ppg", "implied_win_prob", "has_vegas")


def attach_vegas_team(
    df: pl.DataFrame, vegas_team: pl.DataFrame | None, season_col: str = "season", team_col: str = "team"
) -> pl.DataFrame:
    """Left-join vegas_team_features onto df (must carry season_col/team_col); has_vegas
    defaults to 0 (never null) for any row with no matching priced (season, team) row --
    implied_ppg/implied_win_prob stay null in that case, per the brief's "nullable."
    """
    feats = vegas_team_features(vegas_team).rename({"season": season_col, "team": team_col})
    out = df.join(feats, on=[season_col, team_col], how="left")
    return out.with_columns(pl.col("has_vegas").fill_null(0))



# --------------------------------------------------------------------------
# v2.1 derived depth/competition features. The nflverse ``depth_charts``
# table stops at 2024 (see README's "Known data issues") and ships a
# schema-incompatible payload for 2026 -- unusable for a pipeline that must
# score the current offseason. Everything below derives an equivalent
# "who's ahead of whom on the depth chart" signal purely from data that
# exists for every season including 2026: prior-season (N-1) usage plus the
# season-N Week-1 roster (``season_roster_team``/``season_roster_position``
# below) -- the same two ingredients ``vacated_shares`` already combines,
# just read for the "stayed" half of the population instead of the
# "departed" half.
# --------------------------------------------------------------------------


def season_roster_position(rosters: pl.DataFrame, rosters_weekly: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> position: Week-1 roster position.

    Same two-level fallback as ``season_roster_team`` (earliest available
    REG week that season, else ``rosters.parquet``'s season-level
    ``position`` column when a player has no ``rosters_weekly`` REG row at
    all that season) -- deliberately mirrors that function rather than
    reusing its output, because depth/competition features need each
    player's position *on the season-N roster itself*, not the position he
    produced N-1 stats at (``player_stats.position`` has nothing for 2026 —
    no games played yet — while ``rosters_weekly``/``rosters`` both do).
    """
    reg = rosters_weekly.filter((pl.col("game_type") == "REG") & pl.col("gsis_id").is_not_null())
    reg = reg.with_columns(pl.when(pl.col("status") == "ACT").then(0).otherwise(1).alias("_status_rank"))
    reg = reg.sort(["season", "gsis_id", "week", "_status_rank"]).unique(
        subset=["season", "gsis_id", "week"], keep="first"
    )
    weekly_pos = (
        reg.sort(["season", "gsis_id", "week"])
        .unique(subset=["season", "gsis_id"], keep="first")
        .select("season", "gsis_id", "position")
    )

    season_level = (
        rosters.filter(pl.col("gsis_id").is_not_null())
        .sort(["season", "gsis_id", "week"])
        .unique(subset=["season", "gsis_id"], keep="last")
        .select("season", "gsis_id", "position")
    )
    fallback_only = season_level.join(weekly_pos.select("season", "gsis_id"), on=["season", "gsis_id"], how="anti")

    return pl.concat([weekly_pos, fallback_only], how="vertical_relaxed").select("season", "gsis_id", "position")


def returning_incumbent_share_table(
    reg: pl.DataFrame,
    season_team: pl.DataFrame,
    *,
    position: str,
    usage_col: str,
    team_total_col: str,
) -> pl.DataFrame:
    """season (N), team -> returning_incumbent_share.

    Sum of N-1 team-relative usage share (``usage_col`` / ``team_total_col``
    -- targets/team_targets for WR/TE, carries/team_carries for RB) of
    ``position`` players who played for that team in season N-1 AND are
    still on that team's season-N roster (per ``season_team`` --
    ``season_roster_team``). Structurally the mirror image of
    ``vacated_shares`` (same ``player_team_usage``/``team_usage_totals``
    building blocks, opposite half of the N-1 population: "stayed", not
    "departed"). A team-context fact broadcast to every row on that team,
    not player-specific -- a returning player's own N-1 share counts toward
    his own team's total the same self-inclusive way ``target_share``'s own
    team-total denominator already includes the player's own targets; this
    is a deliberate simplification (a returning starter's own share is
    "returning competition" for depth-chart purposes exactly as much as
    anyone else's), not an oversight.
    """
    pos_reg = reg.filter(pl.col("position") == position)
    usage = player_team_usage(pos_reg)
    totals = team_usage_totals(reg)
    shares = usage.join(totals, on=["season", "team"], how="left")
    shares = shares.with_columns(safe_div(pl.col(usage_col), pl.col(team_total_col)).alias("_share"))

    lookup = season_team.select("season", "gsis_id", pl.col("team").alias("season_n_team")).with_columns(
        (pl.col("season") - 1).alias("season")
    )
    shares = shares.join(lookup, on=["season", "gsis_id"], how="left")
    returning = shares.filter(pl.col("season_n_team") == pl.col("team"))

    out = returning.group_by(["season", "team"]).agg(
        pl.col("_share").sum().round(_FLOAT_SUM_ROUND_DP).alias("returning_incumbent_share")
    )
    return out.with_columns((pl.col("season") + 1).alias("season"))


def depth_rank_table(
    season_roster_team_pos: pl.DataFrame,
    raw_share_n1: pl.DataFrame,
    *,
    position: str,
    share_col: str,
) -> pl.DataFrame:
    """season (N), gsis_id -> depth_rank_derived: rank within the player's season-N team's

    same-``position`` group, ordered descending by each player's own N-1 ``share_col`` value
    (his real share wherever he actually played in N-1 -- ``raw_share_n1`` is the position
    module's own unshifted raw stat table, e.g. ``target_share``/``carry_share`` before the
    ``_n1``-suffix shift is applied; this function does the N-1 -> N shift itself). 1 = highest
    N-1 share on the roster, i.e. presumptive top of the depth chart.

    ``season_roster_team_pos`` is (season, gsis_id, team, position) for the *entire* season-N
    roster at ``position`` -- not just the labeled (non-rookie) population, since a depth chart
    includes rookies too (they simply rank last: ``pl.Expr.rank`` leaves a null ``share_col``
    row's rank null rather than assigning it a number, which is exactly "nulls last" and also
    means including/excluding rookies from the ranking pool never changes a non-null player's
    rank). Nullable by design: a player with no N-1 share at all (rookie, or out of the league
    in N-1) gets a null rank, per the brief.
    """
    roster = season_roster_team_pos.filter(pl.col("position") == position).select("season", "team", "gsis_id")
    share_n1 = raw_share_n1.select("season", "gsis_id", share_col).with_columns((pl.col("season") + 1).alias("season"))
    # De-dupe defensively before the join: season_roster_team_pos should already be one row
    # per (season, gsis_id), but guards against a stray duplicate silently double-weighting
    # a tie. method="min" (not "ordinal"): two players tied on N-1 share must get the SAME
    # rank -- "ordinal" would break the tie by incidental row order, which is not guaranteed
    # stable across two structurally different builds of the same underlying values (verified
    # by the structural leakage test failing under "ordinal" when unrelated pbp rows were
    # perturbed -- share values were identical, only tie-break row order differed).
    roster = roster.unique(subset=["season", "gsis_id"], keep="first")
    j = roster.join(share_n1, on=["season", "gsis_id"], how="left")
    j = j.with_columns(
        pl.col(share_col).rank(method="min", descending=True).over(["season", "team"]).alias("depth_rank_derived")
    )
    return j.select("season", "gsis_id", pl.col("depth_rank_derived").cast(pl.Int64))


def attach_depth_movement(df: pl.DataFrame, depth_all: pl.DataFrame) -> pl.DataFrame:
    """Join depth_rank_derived (season N) + depth_rank_derived_n1 (season N-1's own rank,

    read off the *same* ``depth_rank_table`` output one season earlier and shifted forward)
    onto ``df``, plus the two composites: ``depth_moved_up`` = depth_rank_derived_n1 -
    depth_rank_derived (positive = climbed the pecking order into season N -- e.g. #3 in the
    room last year, #1 now) and ``became_presumptive_starter`` = (depth_rank_derived == 1) AND
    (depth_rank_derived_n1 >= 2) (the classic "backup inherits the job" breakout setup).

    ``depth_all`` is ``depth_rank_table``'s full output (every season it covers, not just the
    label seasons) -- its own row for season N already used N-1 usage + season-N roster (see
    that function's docstring); its row for season N-1 (attached here, shifted +1 to
    ``depth_rank_derived_n1``) used N-2 usage + season-N-1 roster by the identical
    construction. Both null-safe: a player with no derivable rank in either season yields null
    ``depth_moved_up``/``became_presumptive_starter``, never a false 0/1.
    """
    out = df.join(depth_all, on=["season", "gsis_id"], how="left")
    n1 = depth_all.select(
        "season", "gsis_id", pl.col("depth_rank_derived").alias("depth_rank_derived_n1")
    ).with_columns((pl.col("season") + 1).alias("season"))
    out = out.join(n1, on=["season", "gsis_id"], how="left")
    out = out.with_columns(
        pl.when(pl.col("depth_rank_derived").is_null() | pl.col("depth_rank_derived_n1").is_null())
        .then(None)
        .otherwise(pl.col("depth_rank_derived_n1") - pl.col("depth_rank_derived"))
        .alias("depth_moved_up")
    )
    out = out.with_columns(
        pl.when(pl.col("depth_rank_derived").is_null() | pl.col("depth_rank_derived_n1").is_null())
        .then(None)
        .otherwise((pl.col("depth_rank_derived") == 1) & (pl.col("depth_rank_derived_n1") >= 2))
        .cast(pl.Int64)
        .alias("became_presumptive_starter")
    )
    return out


def returning_qb_starter_flag(reg: pl.DataFrame, season_team: pl.DataFrame) -> pl.DataFrame:
    """season (N), gsis_id -> returning_starter: QB's analogue of returning_incumbent_share.

    1 if a QB *other than the row's own player* who threw more than 50% of the team's season
    N-1 pass attempts is still on that team's season-N roster; 0 if no such returning starter
    exists; null if the team has no N-1 QB pass-attempt data at all (first-data-year/expansion
    edge case) to determine a starter from in the first place -- "nullable where underlying
    data is missing," per the brief. Unlike ``returning_incumbent_share`` (a team-level share
    sum, self-inclusive), this is a boolean flag that explicitly excludes the row's own player
    per the brief's own QB wording ("who isn't the player himself") -- a returning starter QB
    cannot be his own competition.
    """
    qb_reg = reg.filter(pl.col("position") == "QB")
    qb_att = (
        qb_reg.group_by(["season", "player_id", "team"])
        .agg(pl.col("attempts").sum().alias("qb_attempts"))
        .rename({"player_id": "gsis_id"})
    )
    team_att = qb_reg.group_by(["season", "team"]).agg(pl.col("attempts").sum().alias("team_qb_attempts"))
    joined = qb_att.join(team_att, on=["season", "team"], how="left")
    joined = joined.with_columns(safe_div(pl.col("qb_attempts"), pl.col("team_qb_attempts")).alias("share"))
    starters = joined.filter(pl.col("share") > 0.5).select("season", "team", "gsis_id")
    starters_n = starters.with_columns((pl.col("season") + 1).alias("season"))

    roster_check = season_team.select("season", "gsis_id", "team")
    starters_returning = starters_n.join(roster_check, on=["season", "team", "gsis_id"], how="inner").rename(
        {"gsis_id": "starter_id"}
    )

    team_had_n1 = (
        team_att.select("season", "team")
        .with_columns((pl.col("season") + 1).alias("season"), pl.lit(1).alias("_had_n1"))
    )

    all_rows = season_team.select("season", "gsis_id", "team")
    out = all_rows.join(starters_returning.select("season", "team", "starter_id"), on=["season", "team"], how="left")
    out = out.join(team_had_n1, on=["season", "team"], how="left")
    out = out.with_columns(
        pl.when(pl.col("_had_n1").is_null())
        .then(None)
        .when(pl.col("starter_id").is_null())
        .then(0)
        .otherwise((pl.col("starter_id") != pl.col("gsis_id")).cast(pl.Int64))
        .alias("returning_starter")
    )
    return out.group_by(["season", "gsis_id"]).agg(pl.col("returning_starter").max())


def attach_young_stayer(df: pl.DataFrame) -> pl.DataFrame:
    """young_stayer = (age <= 24) AND (team_change == 0); null if either input is null.

    Explicit composite (per the brief: trees can't reliably learn interactions at these
    sample sizes -- 27-48 positives per position). ``df`` must carry ``age``/``team_change``.
    """
    return df.with_columns(
        pl.when(pl.col("age").is_null() | pl.col("team_change").is_null())
        .then(None)
        .otherwise((pl.col("age") <= 24) & (pl.col("team_change") == 0))
        .cast(pl.Int64)
        .alias("young_stayer")
    )


def attach_path_to_volume(df: pl.DataFrame, vacated_col: str) -> pl.DataFrame:
    """path_to_volume = df[vacated_col] - returning_incumbent_share (signed).

    Positive = the season-N team's vacated opportunity (departed players' N-1 share) exceeds
    the returning competition already on the roster -- i.e. the path to volume is relatively
    open. Null if either input is null. ``df`` must carry ``returning_incumbent_share`` and
    ``vacated_col`` (``vacated_target_share`` for WR/TE, ``vacated_carry_share`` for RB).
    """
    return df.with_columns(
        pl.when(pl.col(vacated_col).is_null() | pl.col("returning_incumbent_share").is_null())
        .then(None)
        .otherwise(pl.col(vacated_col) - pl.col("returning_incumbent_share"))
        .alias("path_to_volume")
    )


def attach_moved_into_vacancy(df: pl.DataFrame, vacated_col: str) -> pl.DataFrame:
    """moved_into_vacancy = (team_change == 1) AND (df[vacated_col] > 0.25); null if either input is null."""
    return df.with_columns(
        pl.when(pl.col("team_change").is_null() | pl.col(vacated_col).is_null())
        .then(None)
        .otherwise((pl.col("team_change") == 1) & (pl.col(vacated_col) > 0.25))
        .cast(pl.Int64)
        .alias("moved_into_vacancy")
    )


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
