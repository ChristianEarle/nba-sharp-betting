"""RB feature table for breakout modeling (Phase 3).

Builds ``data/processed/features_rb.parquet``: one row per (season N,
gsis_id) for every RB in ``labels.parquet`` with ``is_rookie=False``.
Same population rule, leakage discipline, null handling (``sh.safe_div``),
and output conventions (``*_n1``/``*_yoy_delta``) as ``src.features.wr`` —
see that module's docstring for the full leakage-rule writeup and
season-shifting convention. This module is a *rushing-plus-receiving*
sibling of WR, not a straight port: RB opportunity is carries AND targets,
so several base metrics are new (carry_share, weighted_opportunity_pg,
yards_per_carry, rush_td_rate, backfield_committee_count) while others are
ported unchanged (target_share, receptions_pg, rec_yards_pg, ppr_ppg,
expected_ppr_ppg, efficiency_residual_pg, snap_share).

Non-negotiable leakage rule
---------------------------
Identical to WR: every feature here is knowable *before Week 1 of season
N*. Player-performance features come from season N-1, scoped to the
player's N-1 *primary team* (``src.features.shared.team_assignments`` —
same simplification WR documents for traded players: shares computed
against the team he produced the bulk of his N-1 stats for). Team-context
features (new_hc, new_oc, vacated shares, team rush/pass volume,
competition draft capital) describe the player's season-**N** team, using
only pre-Week-1-knowable facts. ``tests/test_features_positions.py``'s
leakage test proves this structurally the same way WR's does: rebuild a
recent season's RB rows from a ``player_stats`` frame with that season's
rows deleted and assert identical output.

Column groups in the output
----------------------------
- identity: season, gsis_id, player_name, team, preseason_team_fallback
- shared (``src.features.shared``): age, age_sq, years_exp,
  draft_round/draft_pick/log_draft_pick/undrafted, games_prior,
  team_change, new_hc, new_oc, team_pass_att_pg_prior/team_plays_pg_prior/
  team_pass_rate_prior, team_rush_att_pg_prior, vacated_target_share/
  vacated_carry_share, competition_draft_capital (RB picks only),
  label_season_2020
- RB-specific: year_in_league (years_exp capped at 10)
- base metrics, each as ``{name}_n1`` and ``{name}_yoy_delta``:
  carry_share, target_share, weighted_opportunity_pg, receptions_pg,
  rush_yards_pg, rec_yards_pg, yards_per_carry, ppr_ppg,
  expected_ppr_ppg, efficiency_residual_pg, rush_td_rate,
  backfield_committee_count, snap_share (nullable), NGS rushing
  (nullable, 2016+): rush_yards_over_expected_per_att, ngs_efficiency
  (renamed from ngs_rushing's own ``efficiency`` column — kept distinct
  from ``efficiency_residual_pg``, which means something different: the
  ff_opportunity-based PPR-points residual, not NGS's rushing-yards
  efficiency metric)

``weighted_opportunity_pg`` = (carries + 2*targets) / games, the standard
RB workload proxy that weights a target roughly 2x a carry (targets are
worth more in PPR and carry a higher per-touch expected-points value).

``backfield_committee_count``: how many RBs on the player's own N-1
primary team posted a carry share (of that team's *total* rush attempts,
same denominator as ``carry_share`` above — includes non-RB rushers, e.g.
a QB's scrambles, in the denominator but not the count itself) above 20%
that season. A high count describes a crowded backfield the player was
himself part of; it is scoped and shifted exactly like every other
``BASE_METRICS`` column (N-1 value via primary-team join, N-2 for the
delta) even though it is a team fact, because "how committee was my own
team's backfield" is precisely the N-1 context a breakout call needs.

Skipped by design (documented per the brief, not implemented here):
red-zone/goal-line carry share (needs play-by-play, deferred), depth
chart position (no 2025 depth-chart snapshot available), offensive-line
continuity (no source in data/raw).

Optional inputs (snap_counts, ngs_rushing, coaching_changes) are
genuinely optional: every function here runs fine with any of them
``None``, producing null columns instead of raising.
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
NGS_RUSHING_PATH = RAW_DIR / "ngs_rushing.parquet"
OUT_PATH = PROCESSED_DIR / "features_rb.parquet"

POSITION = "RB"

# RBs on a team with more than this share of the team's total rush
# attempts count toward that team's backfield_committee_count.
COMMITTEE_CARRY_SHARE_THRESHOLD = 0.20

# Every base metric gets a "{name}_n1" (season N-1 value) and
# "{name}_yoy_delta" (N-1 minus N-2, null when N-2 is missing) column.
BASE_METRICS = [
    "carry_share",
    "target_share",
    "weighted_opportunity_pg",
    "receptions_pg",
    "rush_yards_pg",
    "rec_yards_pg",
    "yards_per_carry",
    "ppr_ppg",
    "expected_ppr_ppg",
    "efficiency_residual_pg",
    "rush_td_rate",
    "backfield_committee_count",
    "snap_share",
    "rush_yards_over_expected_per_att",
    "ngs_efficiency",
]

_OUT_COLUMNS = (
    ["season", "gsis_id", "player_name", "team", "preseason_team_fallback"]
    + ["age", "age_sq", "years_exp", "year_in_league"]
    + ["draft_round", "draft_pick", "log_draft_pick", "undrafted"]
    + ["games_prior", "team_change", "new_hc", "new_oc"]
    + ["team_pass_att_pg_prior", "team_plays_pg_prior", "team_pass_rate_prior", "team_rush_att_pg_prior"]
    + ["vacated_target_share", "vacated_carry_share", "competition_draft_capital"]
    + ["label_season_2020"]
    + [f"{m}_n1" for m in BASE_METRICS]
    + [f"{m}_yoy_delta" for m in BASE_METRICS]
)


# --------------------------------------------------------------------------
# Raw (unshifted) season-level stat table
# --------------------------------------------------------------------------


def _rate_stats(reg: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> whole-season (every team) rushing+receiving rate stats.

    No team scoping needed — the player's own per-game and per-touch
    rates, not a share of a team total.
    """
    agg = reg.group_by(["season", "player_id"]).agg(
        pl.len().alias("games"),
        pl.col("carries").sum().alias("carries"),
        pl.col("targets").sum().alias("targets"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("rushing_yards").sum().alias("rush_yards"),
        pl.col("receiving_yards").sum().alias("rec_yards"),
        pl.col("rushing_tds").sum().alias("rush_tds"),
    )
    agg = agg.with_columns(
        sh.safe_div(pl.col("carries") + 2 * pl.col("targets"), pl.col("games")).alias("weighted_opportunity_pg"),
        sh.safe_div(pl.col("receptions"), pl.col("games")).alias("receptions_pg"),
        sh.safe_div(pl.col("rush_yards"), pl.col("games")).alias("rush_yards_pg"),
        sh.safe_div(pl.col("rec_yards"), pl.col("games")).alias("rec_yards_pg"),
        sh.safe_div(pl.col("rush_yards"), pl.col("carries")).alias("yards_per_carry"),
        sh.safe_div(pl.col("rush_tds"), pl.col("carries")).alias("rush_td_rate"),
    )
    return agg.rename({"player_id": "gsis_id"}).select(
        "season",
        "gsis_id",
        "weighted_opportunity_pg",
        "receptions_pg",
        "rush_yards_pg",
        "rec_yards_pg",
        "yards_per_carry",
        "rush_td_rate",
    )


def _shares(reg: pl.DataFrame, team_assign: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> carry_share, target_share, scoped to the N-1 primary team."""
    usage = sh.player_team_usage(reg)
    totals = sh.team_usage_totals(reg)
    primary = team_assign.select("season", "gsis_id", pl.col("primary_team").alias("team"))
    scoped = primary.join(usage, on=["season", "gsis_id", "team"], how="left")
    scoped = scoped.join(totals, on=["season", "team"], how="left")
    scoped = scoped.with_columns(
        sh.safe_div(pl.col("carries"), pl.col("team_carries")).alias("carry_share"),
        sh.safe_div(pl.col("targets"), pl.col("team_targets")).alias("target_share"),
    )
    return scoped.select("season", "gsis_id", "carry_share", "target_share")


def _backfield_committee_counts(reg: pl.DataFrame) -> pl.DataFrame:
    """season, team -> backfield_committee_count: # of RBs with carry_share > threshold that season.

    ``reg`` here is the *full* (all-positions) REG frame — carry_share's
    denominator (team_carries) must include every rusher on the team, not
    just RBs, to match the same ``carry_share`` definition used above.
    Only the numerator (which players get counted) is RB-filtered.
    """
    rb_usage = sh.player_team_usage(reg.filter(pl.col("position") == "RB"))
    totals = sh.team_usage_totals(reg)
    j = rb_usage.join(totals, on=["season", "team"], how="left")
    j = j.with_columns(sh.safe_div(pl.col("carries"), pl.col("team_carries")).alias("carry_share"))
    committee = j.filter(pl.col("carry_share") > COMMITTEE_CARRY_SHARE_THRESHOLD)
    return committee.group_by(["season", "team"]).agg(pl.len().alias("committee_n"))


def _committee_count_for_primary_team(reg: pl.DataFrame, team_assign: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> backfield_committee_count, looked up via the player's own N-1 primary team.

    Left join + fill 0: a team-season with no RB crossing the 20% carry-
    share threshold (e.g. an even 3-way committee where nobody clears it)
    is a real zero, not a missing value.
    """
    committee = _backfield_committee_counts(reg)
    primary = team_assign.select("season", "gsis_id", pl.col("primary_team").alias("team"))
    out = primary.join(committee, on=["season", "team"], how="left")
    out = out.with_columns(pl.col("committee_n").fill_null(0).alias("backfield_committee_count"))
    return out.select("season", "gsis_id", "backfield_committee_count")


def _ngs_features(ngs_rushing: pl.DataFrame) -> pl.DataFrame:
    """season, gsis_id -> rush_yards_over_expected_per_att, ngs_efficiency (2016+, REG season totals).

    week == 0 is NGS's own season-aggregate row per player (same
    convention as ``src.features.wr._ngs_features``), used directly rather
    than averaging weekly rows ourselves. ``efficiency`` is renamed to
    ``ngs_efficiency`` on the way out — see module docstring.
    """
    reg = ngs_rushing.filter((pl.col("season_type") == "REG") & (pl.col("week") == 0))
    return reg.select(
        "season",
        pl.col("player_gsis_id").alias("gsis_id"),
        "rush_yards_over_expected_per_att",
        pl.col("efficiency").alias("ngs_efficiency"),
    )


def build_raw_stat_table(
    reg: pl.DataFrame,
    team_assign: pl.DataFrame,
    ff_opportunity: pl.DataFrame,
    schedules: pl.DataFrame,
    snap_counts: pl.DataFrame | None,
    rosters: pl.DataFrame | None,
    ngs_rushing: pl.DataFrame | None,
) -> pl.DataFrame:
    """season, gsis_id -> every BASE_METRICS column, for the season the row is labeled (not shifted).

    ``reg`` must already carry ``computed_points`` (see ``sh.reg_with_points``).
    """
    totals = season_aggregates(reg).select("season", "gsis_id", "games", "ppr_ppg")
    out = totals.join(_rate_stats(reg), on=["season", "gsis_id"], how="left")
    out = out.join(_shares(reg, team_assign), on=["season", "gsis_id"], how="left")
    out = out.join(_committee_count_for_primary_team(reg, team_assign), on=["season", "gsis_id"], how="left")

    exp = sh.expected_points_total(ff_opportunity, schedules)
    out = out.join(exp, on=["season", "gsis_id"], how="left")
    out = out.with_columns((pl.col("expected_total") / pl.col("games")).alias("expected_ppr_ppg"))
    out = out.with_columns((pl.col("ppr_ppg") - pl.col("expected_ppr_ppg")).alias("efficiency_residual_pg"))

    if snap_counts is not None and rosters is not None:
        out = out.join(sh.snap_share_from_counts(snap_counts, rosters), on=["season", "gsis_id"], how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("snap_share"))

    if ngs_rushing is not None:
        out = out.join(_ngs_features(ngs_rushing), on=["season", "gsis_id"], how="left")
    else:
        out = out.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in ["rush_yards_over_expected_per_att", "ngs_efficiency"]]
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


def build_features_rb(
    *,
    labels: pl.DataFrame,
    player_stats: pl.DataFrame,
    ff_opportunity: pl.DataFrame,
    rosters: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
    draft_picks: pl.DataFrame,
    schedules: pl.DataFrame,
    snap_counts: pl.DataFrame | None = None,
    ngs_rushing: pl.DataFrame | None = None,
    coaching_changes: pl.DataFrame | None = None,
    scoring_profile: dict | None = None,
) -> pl.DataFrame:
    """Build the RB feature table from already-loaded frames.

    Every argument is a polars DataFrame, not a path — the leakage test
    exploits this directly, same mechanism as ``src.features.wr``.
    """
    if scoring_profile is None:
        scoring_profile = load_scoring_config()

    reg = sh.reg_with_points(player_stats, scoring_profile)
    team_assign = sh.team_assignments(reg)

    raw = build_raw_stat_table(
        reg, team_assign, ff_opportunity, schedules, snap_counts, rosters, ngs_rushing
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
    out = out.join(sh.team_rush_volume_prior(reg), on=["season", "team"], how="left")
    out = out.join(sh.vacated_shares(reg, season_team), on=["season", "team"], how="left")
    out = out.join(sh.competition_draft_capital(draft_picks, POSITION), on=["season", "team"], how="left")

    out = out.with_columns(pl.min_horizontal(pl.col("years_exp"), pl.lit(10)).alias("year_in_league"))
    out = sh.add_covid_flag(out)

    out = out.select(_OUT_COLUMNS).sort(["season", "gsis_id"])

    dupes = out.group_by(["season", "gsis_id"]).len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"duplicate (season, gsis_id) rows in features_rb output: {dupes}"

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
    ngs_rushing_path: Path = NGS_RUSHING_PATH,
    coaching_changes_path: Path = sh.PATHS["coaching_changes"],
    out_path: Path = OUT_PATH,
) -> pl.DataFrame:
    """Load every default source from disk and build + write features_rb.parquet."""
    labels = pl.read_parquet(labels_path)
    player_stats = pl.read_parquet(player_stats_path)
    ff_opportunity = pl.read_parquet(ff_opportunity_path)
    schedules = pl.read_parquet(schedules_path)
    rosters = pl.read_parquet(rosters_path)
    rosters_weekly = pl.read_parquet(rosters_weekly_path)
    draft_picks = pl.read_parquet(draft_picks_path)
    snap_counts = pl.read_parquet(snap_counts_path) if snap_counts_path.exists() else None
    ngs_rushing = pl.read_parquet(ngs_rushing_path) if ngs_rushing_path.exists() else None
    coaching_changes = sh.load_coaching_changes(coaching_changes_path)

    out = build_features_rb(
        labels=labels,
        player_stats=player_stats,
        ff_opportunity=ff_opportunity,
        rosters=rosters,
        rosters_weekly=rosters_weekly,
        draft_picks=draft_picks,
        schedules=schedules,
        snap_counts=snap_counts,
        ngs_rushing=ngs_rushing,
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
    (2017, "Alvin Kamara", "should NOT appear (rookie)"),
    (2018, "Alvin Kamara", "should appear, year-2 profile"),
    (2019, "Austin Ekeler", "should appear, strong receiving usage"),
    (2020, "Austin Ekeler", "should appear, elite receiving-back profile"),
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
    print("features_rb build | RB only, seasons per labels.parquet")
    out = load_and_build()
    print(f"\nwrote {OUT_PATH} | {out.height:,} rows x {out.width} cols")

    print("\nNull-rate by era bucket (core columns only, era buckets 2014-15 / 2016-21 / 2022-25):")
    nr = _null_rate_table(out)
    core_cols = [
        c
        for c in out.columns
        if c not in ("season", "gsis_id", "player_name", "team")
        and "snap_share" not in c
        and "rush_yards_over_expected_per_att" not in c
        and "ngs_efficiency" not in c
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
            f"carry_share_n1={r['carry_share_n1']:.3f} target_share_n1={r['target_share_n1']:.3f} "
            f"weighted_opportunity_pg_n1={r['weighted_opportunity_pg_n1']:.2f} ppr_ppg_n1={r['ppr_ppg_n1']:.2f} "
            f"backfield_committee_count_n1={r['backfield_committee_count_n1']} "
            f"year_in_league={r['year_in_league']} team_change={r['team_change']} "
            f"games_prior={r['games_prior']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
