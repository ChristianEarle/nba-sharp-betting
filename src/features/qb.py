"""QB feature table for breakout modeling (Phase 3).

Builds ``data/processed/features_qb.parquet``: one row per (season N,
gsis_id) for every QB in ``labels.parquet`` with ``is_rookie=False``. Same
population rule, leakage discipline, null handling (``sh.safe_div``), and
output conventions (``*_n1``/``*_yoy_delta``) as ``src.features.wr`` — see
that module's docstring for the full leakage-rule writeup and
season-shifting convention.

Non-negotiable leakage rule
---------------------------
Identical to WR: every feature here is knowable *before Week 1 of season
N*. Player-performance features come from season N-1, scoped to the
player's N-1 *primary team* where a team-relative share is involved
(``rush_yard_share`` — see WR's traded-players note, same simplification).
Team-context features (new_hc, new_oc, team pass/rush volume, vacated
shares, draft capital) describe the player's season-**N** team, using
only pre-Week-1-knowable facts. ``tests/test_features_positions.py``'s
leakage test proves this structurally the same way WR's does.

Column groups in the output
----------------------------
- identity: season, gsis_id, player_name, team, preseason_team_fallback
- shared (``src.features.shared``): age, age_sq, years_exp,
  draft_round/draft_pick/log_draft_pick/undrafted, games_prior,
  team_change, new_hc, new_oc, team_pass_att_pg_prior/team_plays_pg_prior/
  team_pass_rate_prior, vacated_target_share/vacated_carry_share,
  competition_draft_capital (QB picks — succession-risk signal: another QB
  drafted by this team), supporting_cast_capital (WR/TE/RB picks —
  distinct signal: how much the team invested in *this* QB's supporting
  cast; both are kept, they answer different questions), new_oc_interaction,
  label_season_2020
- QB-specific: year_in_league (years_exp capped at 10)
- base metrics, each as ``{name}_n1`` and ``{name}_yoy_delta``:
  pass_attempts_pg, pass_yards_pg, ppr_ppg, rush_attempts_pg,
  rush_yards_pg, rush_yard_share, sack_rate, int_rate, td_rate,
  expected_ppr_ppg, efficiency_residual_pg, avg_time_to_throw (nullable
  NGS), cpoe (nullable NGS)

Formula notes (spec is terse in a few places; documenting the concrete
choice made)
------------------------------------------------------------------------
- ``td_rate`` = passing_tds / attempts — a *passing* TD rate, mirroring
  how WR's td_rate is rec_tds/targets and RB's is rush_tds/carries (TDs
  per the position's primary "opportunity" type). Rushing scoring is
  already captured by rush volume (rush_attempts_pg/rush_yards_pg) rather
  than double-counted into a blended TD-rate denominator.
- ``sack_rate`` = sacks_suffered / (sacks_suffered + attempts) — the
  standard dropback-rate-style denominator (matches
  ``src.features.shared.team_pass_volume_prior``'s own "pass play =
  attempts + sacks" convention), verified against player_stats' own
  ``sacks_suffered`` column (the QB's own sacks taken, not
  ``def_sacks``/``def_sack_yards``, which are defensive-side columns on
  the same wide table).
- ``rush_yard_share`` = player's N-1 primary-team rushing yards / that
  team's N-1 total rushing yards (every rusher, QB scrambles and RB
  carries alike) — a *yards* share per the brief's "rush yard share of
  team" wording, not an attempts share (attempts volume is already a
  separate feature: rush_attempts_pg).
- ``new_oc_interaction`` = ``new_oc * 1``, literally, per the brief.
  Mathematically identical to the shared ``new_oc`` flag (multiplying a
  0/1/null value by 1 changes nothing) — kept as its own named column
  rather than silently dropped, since the brief calls for it by name as
  a column the modeling config can reference independently of the shared
  block's own ``new_oc``.
- ``pass_attempts_pg``, ``pass_yards_pg``, ``rush_attempts_pg``,
  ``rush_yards_pg``, ``sack_rate``, ``int_rate``, ``td_rate`` are whole-N-1-
  season rate stats (every team the player played for that year) — no
  team scoping needed, same as WR's targets_pg/adot/td_rate family.

v2.4 addition (team-change/vacancy, gated -- src.models.vacancy_gate): vacated_td_share
only (share of the team's N-1 offensive TDs belonging to players absent from its
Week-1 season-N roster) -- a team-context fact like vacated_target_share/
vacated_carry_share above. qb_continuity (this feature family's other WR/TE/RB
column) is deliberately NOT wired here: it answers "how much of this team's N-1
passing volume is still on the roster," which is meaningless self-reference for the
QB row itself. max_single_vacated_share/vacated_goal_line_carry_share likewise have
no QB-specific meaning the brief defines.

Skipped by design (documented per the brief, not implemented here):
red-zone/goal-line pass attempt share (needs play-by-play, deferred),
depth chart position (no 2025 depth-chart snapshot available), offensive-
line continuity (no source in data/raw).

Optional inputs (ngs_passing, coaching_changes) are genuinely optional:
every function here runs fine with either ``None``, producing null
columns instead of raising. (QB has no snap_share feature in the brief —
a starting QB's snap share is close to uninformative, unlike WR/TE/RB
usage shares — so, unlike those three modules, this one does not join
snap_counts at all.)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.features import shared as sh
from src.labels.build import PLAYER_STATS_PATH, load_scoring_config, season_aggregates

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

LABELS_PATH = PROCESSED_DIR / "labels.parquet"
FF_OPPORTUNITY_PATH = RAW_DIR / "ff_opportunity.parquet"
NGS_PASSING_PATH = RAW_DIR / "ngs_passing.parquet"
OUT_PATH = PROCESSED_DIR / "features_qb.parquet"

POSITION = "QB"
# Positions whose season-N draft capital feeds this QB's supporting_cast_capital.
SUPPORTING_CAST_POSITIONS = ["WR", "TE", "RB"]

# Every base metric gets a "{name}_n1" (season N-1 value) and
# "{name}_yoy_delta" (N-1 minus N-2, null when N-2 is missing) column.
BASE_METRICS = [
    "pass_attempts_pg",
    "pass_yards_pg",
    "ppr_ppg",
    "rush_attempts_pg",
    "rush_yards_pg",
    "rush_yard_share",
    "sack_rate",
    "int_rate",
    "td_rate",
    "expected_ppr_ppg",
    "efficiency_residual_pg",
    "avg_time_to_throw",
    "cpoe",
]

_OUT_COLUMNS = (
    ["season", "gsis_id", "player_name", "team", "preseason_team_fallback"]
    + ["age", "age_sq", "years_exp", "year_in_league"]
    + ["draft_round", "draft_pick", "log_draft_pick", "undrafted"]
    + ["games_prior", "team_change", "new_hc", "new_oc", "new_oc_interaction"]
    + ["team_pass_att_pg_prior", "team_plays_pg_prior", "team_pass_rate_prior"]
    + ["vacated_target_share", "vacated_carry_share"]
    + ["competition_draft_capital", "supporting_cast_capital"]
    + ["label_season_2020"]
    + ["implied_ppg", "implied_win_prob", "has_vegas"]
    + [f"{m}_n1" for m in BASE_METRICS]
    + [f"{m}_yoy_delta" for m in BASE_METRICS]
    + [
        "returning_starter",
        "depth_rank_derived",
        "depth_rank_derived_n1",
        "depth_moved_up",
        "became_presumptive_starter",
        "young_stayer",
        "moved_into_vacancy",
    ]
    + sh.VACANCY_COLUMNS_QB
)


# --------------------------------------------------------------------------
# Raw (unshifted) season-level stat table
# --------------------------------------------------------------------------


def _rate_stats(reg: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> whole-season (every team) passing+rushing rate stats.

    No team scoping needed — the player's own per-game and per-attempt
    rates, not a share of a team total.
    """
    agg = reg.group_by(["season", "player_id"]).agg(
        pl.len().alias("games"),
        pl.col("attempts").sum().alias("attempts"),
        pl.col("passing_yards").sum().alias("passing_yards"),
        pl.col("passing_tds").sum().alias("passing_tds"),
        pl.col("passing_interceptions").sum().alias("passing_interceptions"),
        pl.col("sacks_suffered").sum().alias("sacks_suffered"),
        pl.col("carries").sum().alias("carries"),
        pl.col("rushing_yards").sum().alias("rushing_yards"),
    )
    agg = agg.with_columns(
        sh.safe_div(pl.col("attempts"), pl.col("games")).alias("pass_attempts_pg"),
        sh.safe_div(pl.col("passing_yards"), pl.col("games")).alias("pass_yards_pg"),
        sh.safe_div(pl.col("carries"), pl.col("games")).alias("rush_attempts_pg"),
        sh.safe_div(pl.col("rushing_yards"), pl.col("games")).alias("rush_yards_pg"),
        sh.safe_div(pl.col("sacks_suffered"), pl.col("sacks_suffered") + pl.col("attempts")).alias("sack_rate"),
        sh.safe_div(pl.col("passing_interceptions"), pl.col("attempts")).alias("int_rate"),
        sh.safe_div(pl.col("passing_tds"), pl.col("attempts")).alias("td_rate"),
    )
    return agg.rename({"player_id": "gsis_id"}).select(
        "season",
        "gsis_id",
        "pass_attempts_pg",
        "pass_yards_pg",
        "rush_attempts_pg",
        "rush_yards_pg",
        "sack_rate",
        "int_rate",
        "td_rate",
    )


def _shares(reg: pl.DataFrame, team_assign: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> rush_yard_share, scoped to the N-1 primary team.

    Player's own rushing yards for that team-season / the team's total
    rushing yards that season (every rusher, not QBs only).
    """
    usage = reg.group_by(["season", "player_id", "team"]).agg(pl.col("rushing_yards").sum().alias("rushing_yards"))
    usage = usage.rename({"player_id": "gsis_id"})
    totals = sh.team_usage_totals(reg)
    primary = team_assign.select("season", "gsis_id", pl.col("primary_team").alias("team"))
    scoped = primary.join(usage, on=["season", "gsis_id", "team"], how="left")
    scoped = scoped.join(totals, on=["season", "team"], how="left")
    scoped = scoped.with_columns(
        sh.safe_div(pl.col("rushing_yards"), pl.col("team_rush_yards")).alias("rush_yard_share")
    )
    return scoped.select("season", "gsis_id", "rush_yard_share")


def _ngs_features(ngs_passing: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> avg_time_to_throw, cpoe (2016+, REG season totals).

    week == 0 is NGS's own season-aggregate row per player, same
    convention as ``src.features.wr._ngs_features`` /
    ``src.features.rb._ngs_features``. ``completion_percentage_above_
    expectation`` is renamed to the shorter ``cpoe`` on the way out, per
    the brief's own naming.
    """
    reg = ngs_passing.filter((pl.col("season_type") == "REG") & (pl.col("week") == 0))
    return reg.select(
        "season",
        pl.col("player_gsis_id").alias("gsis_id"),
        "avg_time_to_throw",
        pl.col("completion_percentage_above_expectation").alias("cpoe"),
    )


def build_raw_stat_table(
    reg: pl.DataFrame,
    team_assign: pl.DataFrame,
    ff_opportunity: pl.DataFrame,
    schedules: pl.DataFrame,
    ngs_passing: pl.DataFrame | None,
) -> pl.DataFrame:
    """season, gsis_id -> every BASE_METRICS column, for the season the row is labeled (not shifted).

    ``reg`` must already carry ``computed_points`` (see ``sh.reg_with_points``).
    """
    totals = season_aggregates(reg).select("season", "gsis_id", "games", "ppr_ppg")
    out = totals.join(_rate_stats(reg), on=["season", "gsis_id"], how="left")
    out = out.join(_shares(reg, team_assign), on=["season", "gsis_id"], how="left")

    exp = sh.expected_points_total(ff_opportunity, schedules)
    out = out.join(exp, on=["season", "gsis_id"], how="left")
    out = out.with_columns((pl.col("expected_total") / pl.col("games")).alias("expected_ppr_ppg"))
    out = out.with_columns((pl.col("ppr_ppg") - pl.col("expected_ppr_ppg")).alias("efficiency_residual_pg"))

    if ngs_passing is not None:
        out = out.join(_ngs_features(ngs_passing), on=["season", "gsis_id"], how="left")
    else:
        out = out.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in ["avg_time_to_throw", "cpoe"]]
        )

    return out.select(["season", "gsis_id"] + BASE_METRICS)


def attach_n1_and_yoy_delta(target: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """Join season-(N-1) value and (N-1 minus N-2) delta of every BASE_METRICS column onto `target`.

    Identical mechanism to ``src.features.wr.attach_n1_and_yoy_delta``.
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


def build_features_qb(
    *,
    labels: pl.DataFrame,
    player_stats: pl.DataFrame,
    ff_opportunity: pl.DataFrame,
    rosters: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
    draft_picks: pl.DataFrame,
    schedules: pl.DataFrame,
    ngs_passing: pl.DataFrame | None = None,
    coaching_changes: pl.DataFrame | None = None,
    vegas_team: pl.DataFrame | None = None,
    scoring_profile: dict | None = None,
) -> pl.DataFrame:
    """Build the QB feature table from already-loaded frames.

    Every argument is a polars DataFrame, not a path — the leakage test
    exploits this directly, same mechanism as ``src.features.wr``.
    """
    if scoring_profile is None:
        scoring_profile = load_scoring_config()

    reg = sh.reg_with_points(player_stats, scoring_profile)
    team_assign = sh.team_assignments(reg)

    raw = build_raw_stat_table(reg, team_assign, ff_opportunity, schedules, ngs_passing)

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
    out = out.with_columns((pl.col("new_oc") * 1).alias("new_oc_interaction"))
    out = out.join(sh.team_pass_volume_prior(reg), on=["season", "team"], how="left")
    out = out.join(sh.vacated_shares(reg, season_team), on=["season", "team"], how="left")
    out = out.join(sh.competition_draft_capital(draft_picks, POSITION), on=["season", "team"], how="left")
    supporting_cast = sh.competition_draft_capital(draft_picks, SUPPORTING_CAST_POSITIONS).rename(
        {"competition_draft_capital": "supporting_cast_capital"}
    )
    out = out.join(supporting_cast, on=["season", "team"], how="left")
    out = sh.attach_vegas_team(out, vegas_team)

    # v2.1 derived depth/competition features -- see src.features.shared's
    # module-level comment block above season_roster_position. QB has no
    # returning_incumbent_share/path_to_volume: those are share-based
    # (target/carry), which has no QB analogue -- the brief gives QB its own
    # "returning starter" flag instead (see returning_qb_starter_flag),
    # which fills that slot here; path_to_volume (vacated share minus
    # returning share) is simply not defined for a starting-job opportunity
    # and is deliberately not added to this table (documented deviation,
    # not an oversight -- see the v2.1 final report).
    out = out.join(sh.returning_qb_starter_flag(reg, season_team), on=["season", "gsis_id"], how="left")
    season_pos = sh.season_roster_position(rosters, rosters_weekly)
    roster_team_pos = season_team.join(season_pos, on=["season", "gsis_id"], how="inner")
    depth_all = sh.depth_rank_table(roster_team_pos, raw, position=POSITION, share_col="pass_attempts_pg")
    out = sh.attach_depth_movement(out, depth_all)
    out = sh.attach_young_stayer(out)
    # QB analogue of moved_into_vacancy: moved teams into a job with no
    # proven returning starter already occupying it (rather than "vacated
    # share > 0.25", which has no QB meaning -- see returning_starter above).
    out = out.with_columns(
        pl.when(pl.col("team_change").is_null() | pl.col("returning_starter").is_null())
        .then(None)
        .otherwise((pl.col("team_change") == 1) & (pl.col("returning_starter") == 0))
        .cast(pl.Int64)
        .alias("moved_into_vacancy")
    )

    # v2.4: vacated_td_share only (see src.features.shared.VACANCY_COLUMNS_QB's docstring --
    # qb_continuity is meaningless for the thrower himself, so it's skipped here by design;
    # vacated_td_share is a team-context fact like vacated_target_share/vacated_carry_share
    # above, gated the same way (src.models.vacancy_gate) before shipping in this model).
    out = out.join(sh.vacated_td_share_table(reg, season_team), on=["season", "team"], how="left")

    out = out.with_columns(pl.min_horizontal(pl.col("years_exp"), pl.lit(10)).alias("year_in_league"))
    out = sh.add_covid_flag(out)

    out = out.select(_OUT_COLUMNS).sort(["season", "gsis_id"])

    dupes = out.group_by(["season", "gsis_id"]).len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"duplicate (season, gsis_id) rows in features_qb output: {dupes}"

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
    ngs_passing_path: Path = NGS_PASSING_PATH,
    coaching_changes_path: Path = sh.PATHS["coaching_changes"],
    vegas_team_path: Path = PROCESSED_DIR / "vegas_team.parquet",
    out_path: Path = OUT_PATH,
) -> pl.DataFrame:
    """Load every default source from disk and build + write features_qb.parquet."""
    labels = pl.read_parquet(labels_path)
    player_stats = pl.read_parquet(player_stats_path)
    ff_opportunity = pl.read_parquet(ff_opportunity_path)
    schedules = pl.read_parquet(schedules_path)
    rosters = pl.read_parquet(rosters_path)
    rosters_weekly = pl.read_parquet(rosters_weekly_path)
    draft_picks = pl.read_parquet(draft_picks_path)
    ngs_passing = pl.read_parquet(ngs_passing_path) if ngs_passing_path.exists() else None
    coaching_changes = sh.load_coaching_changes(coaching_changes_path)
    vegas_team = pl.read_parquet(vegas_team_path) if vegas_team_path.exists() else None

    out = build_features_qb(
        labels=labels,
        player_stats=player_stats,
        ff_opportunity=ff_opportunity,
        rosters=rosters,
        rosters_weekly=rosters_weekly,
        draft_picks=draft_picks,
        schedules=schedules,
        ngs_passing=ngs_passing,
        coaching_changes=coaching_changes,
        vegas_team=vegas_team,
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
    (2019, "Lamar Jackson", "should appear, high N-1 (2018) rush yard share"),
    (2023, "Jalen Hurts", "should appear, high N-1 (2022) rush volume"),
    (2020, "Jalen Hurts", "should NOT appear (rookie)"),
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
    print("features_qb build | QB only, seasons per labels.parquet")
    out = load_and_build()
    print(f"\nwrote {OUT_PATH} | {out.height:,} rows x {out.width} cols")

    print("\nNull-rate by era bucket (core columns only, era buckets 2014-15 / 2016-21 / 2022-25):")
    nr = _null_rate_table(out)
    core_cols = [
        c
        for c in out.columns
        if c not in ("season", "gsis_id", "player_name", "team")
        and "avg_time_to_throw" not in c
        and "cpoe" not in c
    ]
    with pl.Config(tbl_rows=-1, tbl_width_chars=200):
        print(nr.filter(pl.col("column").is_in(core_cols)).pivot("era", index="column", values="null_rate"))
        print("\nNGS (nullable-by-design) columns:")
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
            f"rush_yard_share_n1={r['rush_yard_share_n1']:.3f} rush_attempts_pg_n1={r['rush_attempts_pg_n1']:.2f} "
            f"rush_yards_pg_n1={r['rush_yards_pg_n1']:.2f} pass_attempts_pg_n1={r['pass_attempts_pg_n1']:.2f} "
            f"ppr_ppg_n1={r['ppr_ppg_n1']:.2f} year_in_league={r['year_in_league']} "
            f"team_change={r['team_change']} games_prior={r['games_prior']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
