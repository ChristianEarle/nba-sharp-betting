"""TE feature table for breakout modeling (Phase 3).

Builds ``data/processed/features_te.parquet``: one row per (season N,
gsis_id) for every TE in ``labels.parquet`` with ``is_rookie=False`` —
same population rule, same leakage discipline, and (almost) the same
receiving-family feature set as ``src.features.wr``, whose module
docstring has the full leakage-rule writeup and season-shifting
convention this module follows identically. Differences from WR are
called out inline; everything else is a direct port.

Non-negotiable leakage rule
---------------------------
Identical to WR: every feature here is knowable *before Week 1 of season
N* — season N-1 (and earlier) production, plus offseason facts. Player-
performance features come from season N-1 (scoped to the player's N-1
primary team for target_share et al — see WR's "traded players" note,
which applies here unchanged); team-context features describe the
player's season-N team using only pre-Week-1-knowable facts.
``tests/test_features_positions.py``'s leakage test proves this
structurally the same way WR's does: rebuild season-2023 TE rows from a
``player_stats`` frame with every 2023 row deleted and assert identical
output.

Column groups in the output
----------------------------
- identity: season, gsis_id, player_name, team, preseason_team_fallback
- shared (``src.features.shared``): age, age_sq, years_exp,
  draft_round/draft_pick/log_draft_pick/undrafted, games_prior,
  team_change, new_hc, new_oc, team_pass_att_pg_prior/team_plays_pg_prior/
  team_pass_rate_prior, vacated_target_share/vacated_carry_share,
  competition_draft_capital (TE picks only), label_season_2020
- TE-specific: year_in_league (years_exp capped at 10)
- base metrics, each as ``{name}_n1`` and ``{name}_yoy_delta``:
  target_share, air_yards_share, wopr, targets_pg, receptions_pg,
  rec_yards_pg, adot, ppr_ppg, expected_ppr_ppg, efficiency_residual_pg,
  yards_per_reception, td_rate, snap_share (nullable), avg_separation/
  avg_cushion/catch_percentage (nullable, NGS 2016+), rz_target_share/
  ez_target_share (nullable, v1.5 pbp-derived -- see WR's docstring)

This is the *identical* BASE_METRICS list WR uses — the brief specs "same
receiving family as WR" for TE, and nothing about tight end usage needs a
different formula for any of these (target_share/wopr/adot etc. are all
already position-agnostic ratios). The only thing that changes between
the two modules is the population filter (``position == "TE"``) and the
competition-draft-capital position argument (TE picks, not WR picks).

v1.5 addition (Phase C): rz_target_share/ez_target_share, inherited
unchanged from ``src.features.wr`` (TE reuses WR's ``BASE_METRICS`` and
``build_raw_stat_table`` verbatim -- see the module-level import above),
so no TE-specific pbp wiring was needed beyond passing ``pbp`` through.

Skipped by design (documented per the brief, not implemented here): depth
chart position (no 2025 depth-chart snapshot available), offensive-line
continuity (no source in data/raw).

Optional inputs (snap_counts, ngs_receiving, coaching_changes) are
genuinely optional: every function here runs fine with any of them
``None``, producing null columns instead of raising. Verified against
pre-2016 rows (no NGS) in ``tests/test_features_positions.py``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.features import shared as sh
from src.features.wr import (
    BASE_METRICS,
    attach_n1_and_yoy_delta,
    build_raw_stat_table,
)
from src.labels.build import PLAYER_STATS_PATH, load_scoring_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

LABELS_PATH = PROCESSED_DIR / "labels.parquet"
FF_OPPORTUNITY_PATH = RAW_DIR / "ff_opportunity.parquet"
NGS_RECEIVING_PATH = RAW_DIR / "ngs_receiving.parquet"
OUT_PATH = PROCESSED_DIR / "features_te.parquet"

POSITION = "TE"

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
# Orchestration
# --------------------------------------------------------------------------


def build_features_te(
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
    pbp: pl.DataFrame | None = None,
    scoring_profile: dict | None = None,
) -> pl.DataFrame:
    """Build the TE feature table from already-loaded frames.

    Same builders as ``src.features.wr.build_features_wr`` (receiving
    family is identical, v1.5 pbp-derived rz_target_share/ez_target_share
    included), swapped to the TE population and TE-scoped competition
    draft capital. Every argument is a polars DataFrame, not a path — see
    WR's docstring for why (the leakage test exploits this directly).
    """
    if scoring_profile is None:
        scoring_profile = load_scoring_config()

    reg = sh.reg_with_points(player_stats, scoring_profile)
    team_assign = sh.team_assignments(reg)

    raw = build_raw_stat_table(
        reg, team_assign, ff_opportunity, schedules, snap_counts, rosters, ngs_receiving, pbp
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
    assert dupes.is_empty(), f"duplicate (season, gsis_id) rows in features_te output: {dupes}"

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
    pbp_path: Path = RAW_DIR / "pbp.parquet",
    out_path: Path = OUT_PATH,
) -> pl.DataFrame:
    """Load every default source from disk and build + write features_te.parquet."""
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
    pbp = pl.read_parquet(pbp_path) if pbp_path.exists() else None

    out = build_features_te(
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
        pbp=pbp,
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
    (2020, "Darren Waller", "should appear, elite N-1 shares (2019 breakout year)"),
    (2021, "Mark Andrews", "should appear, elite N-1 shares (2020 season)"),
    (2018, "Mark Andrews", "should NOT appear (rookie)"),
    (2019, "Mark Andrews", "should appear, year-2 profile"),
    (2016, "Darren Waller", "should appear, year-2 profile"),
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
    print("features_te build | TE only, seasons per labels.parquet")
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
        and "rz_target_share" not in c
        and "ez_target_share" not in c
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
