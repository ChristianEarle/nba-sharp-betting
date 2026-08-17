"""WR feature table for breakout modeling (Phase 3).

Builds ``data/processed/features_wr.parquet``: one row per (season N,
gsis_id) for every WR in ``labels.parquet`` with ``is_rookie=False``
(rookies get a separate rookie heuristic later, per Phase 2's design —
they have no N-1 season to build any of this from anyway).

Non-negotiable leakage rule
---------------------------
Every feature here must be knowable *before Week 1 of season N*: season
N-1 (and earlier) production, plus offseason facts (draft results,
preseason roster, Week-1 coaching staff). Two kinds of feature obey two
different timing rules:

- **Player-performance features** (targets/game, target_share, ppr_ppg,
  ...) come from season N-1, scoped to whatever team the player produced
  for that year (see "traded players" below) — never season N.
- **Team-context features** (new_hc, new_oc, team pass volume, vacated
  shares, competition draft capital) describe the player's season-**N**
  team, but only using facts knowable pre-Week-1: the Week-1 coaching
  staff, the completed season-N draft, and the *prior* season's (N-1)
  team-level production. A traded player gets his old team's N-1
  production and his new team's preseason context — exactly the brief's
  "old production, new environment."

Every one of those N-1/N-only joins is a season *shift* (``season + 1``)
applied to an otherwise ordinary season-N-1 aggregate, so nothing here
ever reads season-N *game* rows for a feature value. ``tests/test_features.
py``'s leakage test proves this structurally: it rebuilds season-2023 WR
rows from a ``player_stats`` frame with every 2023 row deleted and asserts
byte-identical output — those rows can only be identical if nothing in
them was ever computed from 2023 data.

Traded players (honesty-rules note)
------------------------------------
``target_share`` / ``air_yards_share`` / ``wopr`` are scoped to the
player's **primary team** in N-1 (the team he played the most weeks for
that season — see ``src.features.shared.team_assignments``), not a
per-week-weighted blend across two teams. This is the simpler of the two
options the spec allows ("compute against the team he was on per-week,
... or document a simpler choice"): a player who played 3 games for his
old team and 14 for his new one gets his share computed entirely against
the new team, which is materially more informative than either splitting
the season or ignoring the trade. Rate stats that don't need a team
denominator (targets/game, receptions/game, yards/game, aDOT, ppr_ppg,
yards/reception, td_rate) use the player's *whole* N-1 season across every
team he played for — no scoping issue there.

Column groups in the output
----------------------------
- identity: season, gsis_id, player_name, team (season-N **Week-1** roster
  team, from ``src.features.shared.season_roster_team`` — sourced from
  ``rosters_weekly.parquet`` week==1, deliberately *not* derived from any
  season-N ``player_stats`` row nor from ``rosters.parquet``'s season-level
  table, which reflects a player's *final* team of the season and leaked a
  real mid-season trade in an earlier version of this module — see that
  function's docstring), preseason_team_fallback (1 when a player had no
  weekly-roster row that season at all and the season-level table was used
  instead — see ``season_roster_team``)
- shared (``src.features.shared``): age, age_sq, years_exp,
  draft_round/draft_pick/log_draft_pick/undrafted, games_prior,
  team_change, new_hc, new_oc, team_pass_att_pg_prior/team_plays_pg_prior/
  team_pass_rate_prior, vacated_target_share/vacated_carry_share,
  competition_draft_capital, label_season_2020
- WR-specific: year_in_league (years_exp capped at 10)
- base metrics, each as ``{name}_n1`` (season N-1 value) and
  ``{name}_yoy_delta`` (N-1 minus N-2, null with no N-2 season): target_share,
  air_yards_share, wopr, targets_pg, receptions_pg, rec_yards_pg, adot,
  ppr_ppg, expected_ppr_ppg, efficiency_residual_pg, yards_per_reception,
  td_rate, snap_share (nullable), avg_separation/avg_cushion/
  catch_percentage (nullable, NGS 2016+)

Optional inputs (snap_counts, ngs_receiving, coaching_changes) are
genuinely optional: every function here runs fine with any of them
``None``, producing null columns instead of raising. Verified against
pre-2016 rows (no NGS) in ``tests/test_features.py``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.features import shared as sh
from src.labels.build import (
    PLAYER_STATS_PATH,
    compute_weekly_points,
    load_scoring_config,
    season_aggregates,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

LABELS_PATH = PROCESSED_DIR / "labels.parquet"
FF_OPPORTUNITY_PATH = RAW_DIR / "ff_opportunity.parquet"
NGS_RECEIVING_PATH = RAW_DIR / "ngs_receiving.parquet"
OUT_PATH = PROCESSED_DIR / "features_wr.parquet"

POSITION = "WR"

# Every base metric gets a "{name}_n1" (season N-1 value) and
# "{name}_yoy_delta" (N-1 minus N-2, null when N-2 is missing) column.
BASE_METRICS = [
    "target_share",
    "air_yards_share",
    "wopr",
    "targets_pg",
    "receptions_pg",
    "rec_yards_pg",
    "adot",
    "ppr_ppg",
    "expected_ppr_ppg",
    "efficiency_residual_pg",
    "yards_per_reception",
    "td_rate",
    "snap_share",
    "avg_separation",
    "avg_cushion",
    "catch_percentage",
]

_OUT_COLUMNS = (
    ["season", "gsis_id", "player_name", "team", "preseason_team_fallback"]
    + ["age", "age_sq", "years_exp", "year_in_league"]
    + ["draft_round", "draft_pick", "log_draft_pick", "undrafted"]
    + ["games_prior", "team_change", "new_hc", "new_oc"]
    + ["team_pass_att_pg_prior", "team_plays_pg_prior", "team_pass_rate_prior"]
    + ["vacated_target_share", "vacated_carry_share", "competition_draft_capital"]
    + ["label_season_2020"]
    + [f"{m}_n1" for m in BASE_METRICS]
    + [f"{m}_yoy_delta" for m in BASE_METRICS]
)


# --------------------------------------------------------------------------
# Raw (unshifted) season-level stat table
# --------------------------------------------------------------------------


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """num / den, null (not NaN) when den is 0 or null.

    A plain ``num / den`` on a genuine 0/0 (e.g. a special-teamer with 0
    targets and 0 air yards) yields float NaN, not null — a real
    correctness bug (NaN survives null-checks, breaks null-rate reporting,
    and most downstream ML libraries treat it differently from a missing
    value). Every ratio-shaped feature in this module goes through this.
    """
    return pl.when(den == 0).then(None).otherwise(num / den)


def _reg_with_points(player_stats: pl.DataFrame, profile: dict) -> pl.DataFrame:
    """REG rows + computed_points, same formula src.labels.build uses (imported, not duplicated)."""
    reg = sh.filter_reg(player_stats)
    return reg.with_columns(compute_weekly_points(reg, profile).alias("computed_points"))


def _rate_stats(reg: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> whole-season (every team) receiving rate stats.

    No team scoping needed here — these are the player's own per-game and
    per-target rates, not shares of a team total.
    """
    agg = reg.group_by(["season", "player_id"]).agg(
        pl.len().alias("games"),
        pl.col("targets").sum().alias("targets"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("receiving_yards").sum().alias("rec_yards"),
        pl.col("receiving_air_yards").sum().alias("air_yards"),
        pl.col("receiving_tds").sum().alias("rec_tds"),
    )
    agg = agg.with_columns(
        _safe_div(pl.col("targets"), pl.col("games")).alias("targets_pg"),
        _safe_div(pl.col("receptions"), pl.col("games")).alias("receptions_pg"),
        _safe_div(pl.col("rec_yards"), pl.col("games")).alias("rec_yards_pg"),
        _safe_div(pl.col("air_yards"), pl.col("targets")).alias("adot"),
        _safe_div(pl.col("rec_yards"), pl.col("receptions")).alias("yards_per_reception"),
        _safe_div(pl.col("rec_tds"), pl.col("targets")).alias("td_rate"),
    )
    return agg.rename({"player_id": "gsis_id"}).select(
        "season", "gsis_id", "targets_pg", "receptions_pg", "rec_yards_pg", "adot", "yards_per_reception", "td_rate"
    )


def _shares(reg: pl.DataFrame, team_assign: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> target_share, air_yards_share, wopr, scoped to the N-1 primary team."""
    usage = sh.player_team_usage(reg)
    totals = sh.team_usage_totals(reg)
    primary = team_assign.select("season", "gsis_id", pl.col("primary_team").alias("team"))
    scoped = primary.join(usage, on=["season", "gsis_id", "team"], how="left")
    scoped = scoped.join(totals, on=["season", "team"], how="left")
    scoped = scoped.with_columns(
        _safe_div(pl.col("targets"), pl.col("team_targets")).alias("target_share"),
        _safe_div(pl.col("air_yards"), pl.col("team_air_yards")).alias("air_yards_share"),
    )
    scoped = scoped.with_columns(
        (1.5 * pl.col("target_share") + 0.7 * pl.col("air_yards_share")).alias("wopr")
    )
    return scoped.select("season", "gsis_id", "target_share", "air_yards_share", "wopr")


def _expected_points(ff_opportunity: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> expected_total: sum of ff_opportunity's total_fantasy_points_exp, REG only.

    ff_opportunity ships no season_type column; REG rows are identified by
    joining game_id against schedules (verified: every ff_opportunity
    game_id resolves to a schedules game_type, no unmatched rows).
    """
    ff = ff_opportunity.with_columns(pl.col("season").cast(pl.Int64))
    reg_games = schedules.filter(pl.col("game_type") == "REG").select("game_id")
    ff_reg = ff.join(reg_games, on="game_id", how="inner").filter(pl.col("player_id").is_not_null())
    agg = ff_reg.group_by(["season", "player_id"]).agg(
        pl.col("total_fantasy_points_exp").sum().alias("expected_total")
    )
    return agg.rename({"player_id": "gsis_id"})


def _snap_share(snap_counts: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> snap_share: mean REG-season offense_pct.

    snap_counts carries no gsis_id, only pfr_player_id; resolved via
    rosters' own (season, pfr_id, gsis_id) crosswalk. Rows that don't
    resolve (older/deeper-roster players with no pfr_id on file) are
    dropped here, not guessed — they simply stay null downstream.
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


def _ngs_features(ngs_receiving: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> avg_separation, avg_cushion, catch_percentage (2016+, REG season totals).

    week == 0 is NGS's own season-aggregate row per player (verified
    against Justin Jefferson 2023: week-0 row matches season_type='REG'
    with no per-week breakdown), used directly rather than averaging the
    weekly rows ourselves.
    """
    reg = ngs_receiving.filter((pl.col("season_type") == "REG") & (pl.col("week") == 0))
    return reg.select(
        "season",
        pl.col("player_gsis_id").alias("gsis_id"),
        "avg_separation",
        "avg_cushion",
        "catch_percentage",
    )


def build_raw_stat_table(
    reg: pl.DataFrame,
    team_assign: pl.DataFrame,
    ff_opportunity: pl.DataFrame,
    schedules: pl.DataFrame,
    snap_counts: pl.DataFrame | None,
    rosters: pl.DataFrame | None,
    ngs_receiving: pl.DataFrame | None,
) -> pl.DataFrame:
    """season, gsis_id -> every BASE_METRICS column, for the season the row is labeled (not shifted).

    ``reg`` must already carry ``computed_points`` (see ``_reg_with_points``).
    """
    totals = season_aggregates(reg).select("season", "gsis_id", "games", "ppr_ppg")
    out = totals.join(_rate_stats(reg), on=["season", "gsis_id"], how="left")
    out = out.join(_shares(reg, team_assign), on=["season", "gsis_id"], how="left")

    exp = _expected_points(ff_opportunity, schedules)
    out = out.join(exp, on=["season", "gsis_id"], how="left")
    out = out.with_columns((pl.col("expected_total") / pl.col("games")).alias("expected_ppr_ppg"))
    out = out.with_columns((pl.col("ppr_ppg") - pl.col("expected_ppr_ppg")).alias("efficiency_residual_pg"))

    if snap_counts is not None and rosters is not None:
        out = out.join(_snap_share(snap_counts, rosters), on=["season", "gsis_id"], how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("snap_share"))

    if ngs_receiving is not None:
        out = out.join(_ngs_features(ngs_receiving), on=["season", "gsis_id"], how="left")
    else:
        out = out.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in ["avg_separation", "avg_cushion", "catch_percentage"]]
        )

    return out.select(["season", "gsis_id"] + BASE_METRICS)


def attach_n1_and_yoy_delta(target: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """Join season-(N-1) value and (N-1 minus N-2) delta of every BASE_METRICS column onto `target`.

    `target` must carry a `season` column meaning label season N.
    """
    n1 = raw.select(["season", "gsis_id"] + BASE_METRICS).with_columns((pl.col("season") + 1).alias("season"))
    n1 = n1.rename({c: f"{c}_n1" for c in BASE_METRICS})
    n2 = raw.select(["season", "gsis_id"] + BASE_METRICS).with_columns((pl.col("season") + 2).alias("season"))
    n2 = n2.rename({c: f"{c}_n2" for c in BASE_METRICS})

    out = target.join(n1, on=["season", "gsis_id"], how="left")
    out = out.join(n2, on=["season", "gsis_id"], how="left")
    for c in BASE_METRICS:
        out = out.with_columns((pl.col(f"{c}_n1") - pl.col(f"{c}_n2")).alias(f"{c}_yoy_delta"))
    return out.drop([f"{c}_n2" for c in BASE_METRICS])


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_features_wr(
    *,
    labels: pl.DataFrame,
    player_stats: pl.DataFrame,
    ff_opportunity: pl.DataFrame,
    rosters: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
    draft_picks: pl.DataFrame,
    schedules: pl.DataFrame,
    snap_counts: pl.DataFrame | None = None,
    ngs_receiving: pl.DataFrame | None = None,
    coaching_changes: pl.DataFrame | None = None,
    scoring_profile: dict | None = None,
) -> pl.DataFrame:
    """Build the WR feature table from already-loaded frames.

    Every argument is a polars DataFrame, not a path — the leakage test
    exploits this directly, passing a `player_stats` frame with a season's
    rows stripped out to prove that season's own feature rows never used
    them.
    """
    if scoring_profile is None:
        scoring_profile = load_scoring_config()

    reg = _reg_with_points(player_stats, scoring_profile)
    team_assign = sh.team_assignments(reg)

    raw = build_raw_stat_table(
        reg, team_assign, ff_opportunity, schedules, snap_counts, rosters, ngs_receiving
    )

    target = labels.filter((pl.col("position") == POSITION) & (~pl.col("is_rookie"))).select(
        "season", "gsis_id", "player_name"
    )
    out = attach_n1_and_yoy_delta(target, raw)

    season_team = sh.season_roster_team(rosters, rosters_weekly)

    out = out.join(sh.age_and_experience(rosters), on=["season", "gsis_id"], how="left")
    out = sh.attach_draft_capital(out, sh.draft_capital(draft_picks))
    out = out.join(sh.games_played_prior(team_assign), on=["season", "gsis_id"], how="left")
    out = out.join(sh.team_change_flag(season_team, team_assign), on=["season", "gsis_id"], how="left")

    out = out.join(season_team, on=["season", "gsis_id"], how="left")

    out = out.join(sh.new_hc_table(schedules), on=["season", "team"], how="left")
    out = sh.attach_new_oc(out, sh.new_oc_table(coaching_changes), season_col="season", team_col="team")
    out = out.join(sh.team_pass_volume_prior(reg), on=["season", "team"], how="left")
    out = out.join(sh.vacated_shares(reg, season_team), on=["season", "team"], how="left")
    out = out.join(sh.competition_draft_capital(draft_picks, POSITION), on=["season", "team"], how="left")

    out = out.with_columns(pl.min_horizontal(pl.col("years_exp"), pl.lit(10)).alias("year_in_league"))
    out = sh.add_covid_flag(out)

    out = out.select(_OUT_COLUMNS).sort(["season", "gsis_id"])

    dupes = out.group_by(["season", "gsis_id"]).len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"duplicate (season, gsis_id) rows in features_wr output: {dupes}"

    return out


def load_and_build(
    *,
    labels_path: Path = LABELS_PATH,
    player_stats_path: Path = PLAYER_STATS_PATH,
    ff_opportunity_path: Path = FF_OPPORTUNITY_PATH,
    schedules_path: Path = sh.PATHS["schedules"],
    rosters_path: Path = sh.PATHS["rosters"],
    rosters_weekly_path: Path = sh.PATHS["rosters_weekly"],
    draft_picks_path: Path = sh.PATHS["draft_picks"],
    snap_counts_path: Path = sh.PATHS["snap_counts"],
    ngs_receiving_path: Path = NGS_RECEIVING_PATH,
    coaching_changes_path: Path = sh.PATHS["coaching_changes"],
    out_path: Path = OUT_PATH,
) -> pl.DataFrame:
    """Load every default source from disk and build + write features_wr.parquet."""
    labels = pl.read_parquet(labels_path)
    player_stats = pl.read_parquet(player_stats_path)
    ff_opportunity = pl.read_parquet(ff_opportunity_path)
    schedules = pl.read_parquet(schedules_path)
    rosters = pl.read_parquet(rosters_path)
    rosters_weekly = pl.read_parquet(rosters_weekly_path)
    draft_picks = pl.read_parquet(draft_picks_path)
    snap_counts = pl.read_parquet(snap_counts_path) if snap_counts_path.exists() else None
    ngs_receiving = pl.read_parquet(ngs_receiving_path) if ngs_receiving_path.exists() else None
    coaching_changes = sh.load_coaching_changes(coaching_changes_path)

    out = build_features_wr(
        labels=labels,
        player_stats=player_stats,
        ff_opportunity=ff_opportunity,
        rosters=rosters,
        rosters_weekly=rosters_weekly,
        draft_picks=draft_picks,
        schedules=schedules,
        snap_counts=snap_counts,
        ngs_receiving=ngs_receiving,
        coaching_changes=coaching_changes,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return out


# --------------------------------------------------------------------------
# Reporting (used by __main__)
# --------------------------------------------------------------------------

_ERA_BUCKETS = [
    ("2014-15", range(2014, 2016)),
    ("2016-21", range(2016, 2022)),
    ("2022-25", range(2022, 2026)),
]

_SANITY_ROWS = [
    (2022, "Justin Jefferson", "should appear, elite N-1 shares"),
    (2023, "Puka Nacua", "should NOT appear (rookie)"),
    (2021, "Cooper Kupp", "should appear, pre-breakout profile"),
    (2024, "Rashee Rice", "should appear, year-2 profile"),
    (2024, "Puka Nacua", "should appear, year-2 profile"),
    (2019, "DK Metcalf", "should NOT appear (rookie)"),
    (2020, "DK Metcalf", "should appear, year-2 profile"),
]


def _null_rate_table(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for label, seasons in _ERA_BUCKETS:
        sub = df.filter(pl.col("season").is_in(list(seasons)))
        n = sub.height
        for col in df.columns:
            if col in ("season", "gsis_id", "player_name", "team"):
                continue
            nulls = sub.get_column(col).null_count()
            rate = nulls / n if n else None
            rows.append({"era": label, "column": col, "n": n, "null_rate": rate})
    return pl.DataFrame(rows)


def main() -> int:
    print("features_wr build | WR only, seasons per labels.parquet")
    out = load_and_build()
    print(f"\nwrote {OUT_PATH} | {out.height:,} rows x {out.width} cols")

    print("\nNull-rate by era bucket (core columns only, era buckets 2014-15 / 2016-21 / 2022-25):")
    nr = _null_rate_table(out)
    core_cols = [
        c
        for c in out.columns
        if c not in ("season", "gsis_id", "player_name", "team")
        and "snap_share" not in c
        and "avg_separation" not in c
        and "avg_cushion" not in c
        and "catch_percentage" not in c
    ]
    with pl.Config(tbl_rows=-1, tbl_width_chars=200):
        print(nr.filter(pl.col("column").is_in(core_cols)).pivot("era", index="column", values="null_rate"))
        print("\nNGS/snap (nullable-by-design) columns:")
        print(nr.filter(~pl.col("column").is_in(core_cols)).pivot("era", index="column", values="null_rate"))

    print("\nSanity rows:")
    for season, name, note in _SANITY_ROWS:
        row = out.filter((pl.col("season") == season) & (pl.col("player_name") == name))
        if row.height == 0:
            print(f"  {season} {name}: NOT in output ({note})")
            continue
        r = row.row(0, named=True)
        print(
            f"  {season} {name} ({note}): team={r['team']} "
            f"target_share_n1={r['target_share_n1']:.3f} wopr_n1={r['wopr_n1']:.3f} "
            f"targets_pg_n1={r['targets_pg_n1']:.2f} ppr_ppg_n1={r['ppr_ppg_n1']:.2f} "
            f"year_in_league={r['year_in_league']} team_change={r['team_change']} "
            f"games_prior={r['games_prior']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
