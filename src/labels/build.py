"""Breakout target labels (Phase 2).

Builds ``data/processed/labels.parquet``: one row per (season, gsis_id) for
every QB/RB/WR/TE who appeared in a **regular-season** game 2014-2025,
carrying a computed PPR points-per-game rate, a within-position finish
rank, a preseason market-expectation rank, and the derived breakout label.

Everything here is REG-season only. ``player_stats.parquet`` ships REG and
POST rows together; a finish rank built without filtering silently credits
playoff production (Puka Nacua's 2023 reads 18 games unfiltered vs. 17
REG) — see ``test_labels.py::test_reg_only_...`` for the lock.

Pipeline
--------
1. **Computed points** (``compute_weekly_points``): every weekly REG row's
   fantasy points from ``configs/scoring.yaml``'s ``standard_ppr`` profile,
   *not* the shipped ``fantasy_points_ppr`` column — see that config's
   header comment and ``scoring_cross_check`` for why they differ (return
   TDs) and by how much (tiny).
2. **Season aggregates** (``season_aggregates``): games and ppr_ppg per
   (season, gsis_id), computed from *every* REG row regardless of recorded
   position (a player's weekly ``position`` value can wobble — a WR
   occasionally lines up and gets stamped RB, etc.). The season's
   *position* is then assigned separately as the most common QB/RB/WR/TE
   value across that player's weeks; a player with no such week that season
   (kickers, long-snappers, purely defensive players who picked up a stat)
   is dropped entirely — they're not a fantasy-relevant position that year.
3. **Finish ranks** (``add_finish_ranks``): within (season, position),
   1 = best ppr_ppg, restricted to players with >= ``min_games`` REG games
   (``configs/labels.yaml``). Below that, ``finish_pos_rank`` is null —
   there is no such thing as a "top finish" on a 2-game sample.
4. **Market expectation** (``add_expectation``): season's preseason rank.
   2020-2025 join ``market_expectation.parquet`` directly (``adp_source=
   'ecr'``). 2014-2019 have no real market data (Phase 1b), so expectation
   is the proxy: the player's *own* season N-1 ``finish_pos_rank``
   (``adp_source='proxy'``). Either way, a player with no expectation
   (undrafted/unranked and no qualifying prior finish) is never dropped —
   they get capped one slot behind the deepest rank the era's expectation
   signal reaches for that season+position (``adp_source='capped'``): the
   market_expectation table's own deepest ``pos_rank`` for ECR seasons, or
   the deepest prior-season ``finish_pos_rank`` for proxy seasons. Capped
   players are exactly where an unscouted breakout hides, so they must
   count as "expected late," not fall out of the table.
5. **Labels** (``add_labels``): breakout = 1 iff
   ``finish_pos_rank <= finish_top`` AND
   ``expectation_pos_rank >= adp_worse_than`` (both from
   ``configs/labels.yaml``, per position). A null finish rank forces
   breakout = 0. ``finish_rank_delta = expectation_pos_rank -
   finish_pos_rank`` (positive = outperformed expectation), null when
   finish rank is null.
6. **Flags** (``add_rookie_flag``, ``add_training_pool_flag``): ``is_rookie``
   compares the season to the player's rookie season (draft_picks season
   when drafted, else their first season anywhere in player_stats).
   ``in_training_pool`` applies ``configs/labels.yaml``'s
   ``training_pool_filter`` (drop rows with a low prior-season ppg *and* a
   deep preseason rank — thin, low-signal samples) and additionally
   excludes every rookie season, per the brief: rookies keep their label for
   a separate rookie heuristic later but never enter the main model's
   training pool.

Nothing is ever dropped from the *output table* for having no market data,
no prior season, or being a rookie — only ``in_training_pool`` reflects
those exclusions, exactly so the full population stays available for
auditing and for the rookie-specific model later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CONFIG_DIR = REPO_ROOT / "configs"

PLAYER_STATS_PATH = RAW_DIR / "player_stats.parquet"
DRAFT_PICKS_PATH = RAW_DIR / "draft_picks.parquet"
MARKET_EXPECTATION_PATH = PROCESSED_DIR / "market_expectation.parquet"
SCORING_CONFIG_PATH = CONFIG_DIR / "scoring.yaml"
LABELS_CONFIG_PATH = CONFIG_DIR / "labels.yaml"
OUT_PATH = PROCESSED_DIR / "labels.parquet"

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

# market_expectation.parquet only reaches back to 2020 (Phase 1b); years
# before that have no real market snapshot to join.
ECR_FIRST_SEASON = 2020

_OUT_COLUMNS = [
    "season",
    "gsis_id",
    "player_name",
    "position",
    "games",
    "ppr_ppg",
    "finish_pos_rank",
    "expectation_pos_rank",
    "adp_source",
    "breakout",
    "finish_rank_delta",
    "is_rookie",
    "in_training_pool",
]


@dataclass
class ScoringCrossCheck:
    mean_abs_diff: float
    n_rows: int
    n_nonzero_diff: int
    top5: pl.DataFrame


def load_scoring_config(path: Path = SCORING_CONFIG_PATH, profile: str = "standard_ppr") -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg["profiles"][profile]


def load_labels_config(path: Path = LABELS_CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def compute_weekly_points(df: pl.DataFrame, profile: dict) -> pl.Expr:
    """Build the computed-points expression from a scoring profile.

    Each rule's ``column`` is either a single player_stats column name or a
    list of columns summed together (the three 2pt-conversion columns,
    which all score identically) before multiplying by ``points``.
    """
    terms = []
    for rule in profile.values():
        cols = rule["column"]
        if isinstance(cols, str):
            stat = pl.col(cols)
        else:
            stat = pl.sum_horizontal([pl.col(c) for c in cols])
        terms.append(stat.cast(pl.Float64) * float(rule["points"]))
    total = terms[0]
    for t in terms[1:]:
        total = total + t
    return total


def scoring_cross_check(reg: pl.DataFrame) -> ScoringCrossCheck:
    """Compare computed_points against the shipped fantasy_points_ppr column.

    Diagnostic only — never used to build labels. A small, systematic gap
    is expected and documented in configs/scoring.yaml: the shipped column
    credits +6 per special-teams (punt/kickoff return) TD, which standard
    redraft PPR scoring does not count.
    """
    diff = (pl.col("computed_points") - pl.col("fantasy_points_ppr")).abs()
    checked = reg.with_columns(diff.alias("abs_diff"))
    mean_abs_diff = checked.get_column("abs_diff").mean()
    n_nonzero = checked.filter(pl.col("abs_diff") > 1e-6).height
    top5 = checked.sort("abs_diff", descending=True).head(5).select(
        "player_name",
        "season",
        "week",
        "abs_diff",
        "computed_points",
        "fantasy_points_ppr",
        "special_teams_tds",
        "def_tds",
        "fumble_recovery_tds",
        "misc_yards",
    )
    return ScoringCrossCheck(
        mean_abs_diff=mean_abs_diff, n_rows=checked.height, n_nonzero_diff=n_nonzero, top5=top5
    )


def load_reg_stats(path: Path = PLAYER_STATS_PATH, scoring_profile: dict | None = None) -> pl.DataFrame:
    """Load player_stats, restrict to REG rows with a real player_id, attach computed_points."""
    if scoring_profile is None:
        scoring_profile = load_scoring_config()
    df = pl.read_parquet(path)
    reg = df.filter((pl.col("season_type") == "REG") & pl.col("player_id").is_not_null())
    reg = reg.with_columns(compute_weekly_points(reg, scoring_profile).alias("computed_points"))
    return reg


def season_aggregates(reg: pl.DataFrame) -> pl.DataFrame:
    """Per (season, gsis_id): games, ppr_ppg, player_name, and skill position.

    games/ppr_ppg come from *every* REG row for that player-season,
    regardless of the position recorded on any individual week. The season
    position is then derived separately as the most common QB/RB/WR/TE
    value across their weeks (an inner join below drops player-seasons with
    no such week — not a fantasy-relevant position that year).
    """
    totals = reg.group_by(["season", "player_id"]).agg(
        pl.len().alias("games"),
        pl.col("computed_points").sum().alias("total_points"),
        pl.col("player_display_name").drop_nulls().first().alias("player_name"),
    )
    totals = totals.with_columns((pl.col("total_points") / pl.col("games")).alias("ppr_ppg"))

    skill_weeks = reg.filter(pl.col("position").is_in(SKILL_POSITIONS))
    pos_counts = skill_weeks.group_by(["season", "player_id", "position"]).agg(pl.len().alias("n"))
    # Most-frequent skill position per player-season; ties break on
    # position's sort order, which is deterministic (not meaningful) but
    # stable across runs.
    pos_mode = (
        pos_counts.sort(["season", "player_id", "n"], descending=[False, False, True])
        .group_by(["season", "player_id"], maintain_order=True)
        .first()
        .select("season", "player_id", "position")
    )

    agg = totals.join(pos_mode, on=["season", "player_id"], how="inner")
    return agg.rename({"player_id": "gsis_id"}).select(
        "season", "gsis_id", "player_name", "position", "games", "ppr_ppg", "total_points"
    )


def add_finish_ranks(agg: pl.DataFrame, min_games: int) -> pl.DataFrame:
    """finish_pos_rank: 1 = best ppr_ppg within (season, position), games >= min_games only."""
    eligible = agg.filter(pl.col("games") >= min_games)
    ranked = eligible.with_columns(
        pl.col("ppr_ppg")
        .rank(method="ordinal", descending=True)
        .over(["season", "position"])
        .cast(pl.Int64)
        .alias("finish_pos_rank")
    )
    return agg.join(
        ranked.select("season", "gsis_id", "finish_pos_rank"), on=["season", "gsis_id"], how="left"
    )


def _deepest_rank_by_season_position(
    agg: pl.DataFrame, market_expectation: pl.DataFrame, ecr_first_season: int, proxy_seasons: range
) -> pl.DataFrame:
    """The deepest expectation rank each season+position's signal reaches.

    ECR seasons: the deepest pos_rank market_expectation.parquet carries for
    that season+position (the full ECR sheet, not just player_stats
    matches — that's genuinely how deep the market ranked players). Proxy
    seasons: the deepest finish_pos_rank the *prior* season produced for
    that position — the proxy has no ranking depth beyond what real
    production gave it the year before.
    """
    deepest_ecr = (
        market_expectation.filter(pl.col("season") >= ecr_first_season)
        .group_by(["season", "position"])
        .agg(pl.col("pos_rank").max().alias("deepest"))
    )
    deepest_proxy = (
        agg.filter(pl.col("finish_pos_rank").is_not_null())
        .group_by(["season", "position"])
        .agg(pl.col("finish_pos_rank").max().alias("deepest"))
        .with_columns((pl.col("season") + 1).alias("season"))
        .filter(pl.col("season").is_in(list(proxy_seasons)))
    )
    return pl.concat([deepest_ecr, deepest_proxy], how="vertical_relaxed")


def add_expectation(
    agg: pl.DataFrame,
    market_expectation: pl.DataFrame,
    label_seasons: range,
    ecr_first_season: int = ECR_FIRST_SEASON,
) -> pl.DataFrame:
    """Attach expectation_pos_rank + adp_source: ecr (2020+), proxy (pre-2020), capped (neither)."""
    proxy_seasons = range(label_seasons.start, ecr_first_season)

    me = (
        market_expectation.filter(pl.col("gsis_id").is_not_null())
        .select("season", "gsis_id", "pos_rank")
        .sort("pos_rank")
        .unique(subset=["season", "gsis_id"], keep="first")
    )
    ecr_expect = me.filter(pl.col("season") >= ecr_first_season).rename(
        {"pos_rank": "expectation_pos_rank"}
    ).with_columns(pl.lit("ecr").alias("adp_source"))

    proxy_expect = (
        agg.filter(pl.col("finish_pos_rank").is_not_null())
        .select("season", "gsis_id", "finish_pos_rank")
        .with_columns((pl.col("season") + 1).alias("season"))
        .filter(pl.col("season").is_in(list(proxy_seasons)))
        .rename({"finish_pos_rank": "expectation_pos_rank"})
        .with_columns(pl.lit("proxy").alias("adp_source"))
    )

    expect = pl.concat([ecr_expect, proxy_expect], how="vertical_relaxed")

    out = agg.filter(pl.col("season").is_in(list(label_seasons))).join(
        expect, on=["season", "gsis_id"], how="left"
    )

    deepest = _deepest_rank_by_season_position(agg, market_expectation, ecr_first_season, proxy_seasons)
    out = out.join(deepest, on=["season", "position"], how="left")
    out = out.with_columns(
        pl.when(pl.col("expectation_pos_rank").is_null())
        .then(pl.col("deepest").fill_null(0) + 1)
        .otherwise(pl.col("expectation_pos_rank"))
        .alias("expectation_pos_rank"),
        pl.when(pl.col("adp_source").is_null()).then(pl.lit("capped")).otherwise(pl.col("adp_source")).alias(
            "adp_source"
        ),
    )
    return out.drop("deepest")


def add_labels(df: pl.DataFrame, thresholds: dict) -> pl.DataFrame:
    """breakout + finish_rank_delta from configs/labels.yaml's per-position thresholds."""
    thresh_rows = [
        {"position": pos, "finish_top": t["finish_top"], "adp_worse_than": t["adp_worse_than"]}
        for pos, t in thresholds.items()
    ]
    thresh_df = pl.DataFrame(thresh_rows)
    out = df.join(thresh_df, on="position", how="left")
    out = out.with_columns(
        pl.when(pl.col("finish_pos_rank").is_null())
        .then(0)
        .when(
            (pl.col("finish_pos_rank") <= pl.col("finish_top"))
            & (pl.col("expectation_pos_rank") >= pl.col("adp_worse_than"))
        )
        .then(1)
        .otherwise(0)
        .cast(pl.Int64)
        .alias("breakout"),
        pl.when(pl.col("finish_pos_rank").is_not_null())
        .then(pl.col("expectation_pos_rank") - pl.col("finish_pos_rank"))
        .otherwise(None)
        .alias("finish_rank_delta"),
    )
    return out.drop("finish_top", "adp_worse_than")


def _rookie_seasons(reg: pl.DataFrame, draft_picks_path: Path = DRAFT_PICKS_PATH) -> pl.DataFrame:
    """gsis_id -> rookie season: draft_picks season when drafted (proper gsis format only),

    else the player's first season anywhere in player_stats REG (undrafted).
    """
    draft = pl.read_parquet(draft_picks_path)
    draft = draft.filter(pl.col("gsis_id").is_not_null() & pl.col("gsis_id").str.starts_with("00-"))
    draft_rookie = draft.group_by("gsis_id").agg(pl.col("season").min().alias("draft_rookie_season"))

    first_appear = reg.group_by("player_id").agg(pl.col("season").min().alias("first_season")).rename(
        {"player_id": "gsis_id"}
    )

    merged = first_appear.join(draft_rookie, on="gsis_id", how="left")
    merged = merged.with_columns(
        pl.coalesce(["draft_rookie_season", "first_season"]).alias("rookie_season")
    )
    return merged.select("gsis_id", "rookie_season")


def add_rookie_flag(df: pl.DataFrame, reg: pl.DataFrame) -> pl.DataFrame:
    rookie = _rookie_seasons(reg)
    out = df.join(rookie, on="gsis_id", how="left")
    out = out.with_columns((pl.col("season") == pl.col("rookie_season")).fill_null(False).alias("is_rookie"))
    return out.drop("rookie_season")


def add_training_pool_flag(df: pl.DataFrame, agg: pl.DataFrame, labels_cfg: dict) -> pl.DataFrame:
    """in_training_pool: passes configs/labels.yaml's training_pool_filter AND not a rookie season.

    A player-season with no prior-season row at all (didn't appear in
    player_stats the year before) is treated as prior_ppr_ppg = 0 — no
    established role is at least as disqualifying as a measured low one.
    """
    filt = labels_cfg["training_pool_filter"]
    prior_ppg = (
        agg.select("season", "gsis_id", "ppr_ppg")
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({"ppr_ppg": "prior_ppr_ppg"})
    )
    out = df.join(prior_ppg, on=["season", "gsis_id"], how="left")
    out = out.with_columns(pl.col("prior_ppr_ppg").fill_null(0.0))

    if filt.get("enabled", True):
        low_quality = (pl.col("prior_ppr_ppg") < filt["min_prior_ppg"]) & (
            pl.col("expectation_pos_rank") > filt["max_pos_rank_when_low_ppg"]
        )
    else:
        low_quality = pl.lit(False)

    out = out.with_columns((~low_quality & ~pl.col("is_rookie")).alias("in_training_pool"))
    return out.drop("prior_ppr_ppg")


def build_labels(
    *,
    player_stats_path: Path = PLAYER_STATS_PATH,
    market_expectation_path: Path = MARKET_EXPECTATION_PATH,
    scoring_config_path: Path = SCORING_CONFIG_PATH,
    labels_config_path: Path = LABELS_CONFIG_PATH,
    out_path: Path = OUT_PATH,
) -> tuple[pl.DataFrame, ScoringCrossCheck]:
    labels_cfg = load_labels_config(labels_config_path)
    scoring_profile = load_scoring_config(scoring_config_path, labels_cfg["scoring_profile"])

    reg = load_reg_stats(player_stats_path, scoring_profile)
    cross_check = scoring_cross_check(reg)

    agg = season_aggregates(reg)
    agg = add_finish_ranks(agg, labels_cfg["min_games"])

    label_seasons = range(labels_cfg["label_seasons"]["start"], labels_cfg["label_seasons"]["end"] + 1)
    market_expectation = pl.read_parquet(market_expectation_path)
    out = add_expectation(agg, market_expectation, label_seasons)
    out = add_labels(out, labels_cfg["thresholds"])
    out = add_rookie_flag(out, reg)
    out = add_training_pool_flag(out, agg, labels_cfg)

    out = out.select(_OUT_COLUMNS).sort(["season", "position", "finish_pos_rank"])

    dupes = out.group_by(["season", "gsis_id"]).len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"duplicate (season, gsis_id) rows in labels output: {dupes}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return out, cross_check


# --------------------------------------------------------------------------
# Reporting (used by __main__)
# --------------------------------------------------------------------------

_NAMED_CANARIES = [
    (2023, "Puka Nacua"),
    (2022, "Justin Jefferson"),
    (2021, "Cordarrelle Patterson"),
    (2017, "Alvin Kamara"),
    (2019, "Lamar Jackson"),
]


def positive_rate_table(df: pl.DataFrame, *, training_pool_only: bool = False) -> pl.DataFrame:
    sub = df.filter(pl.col("games") >= 1)
    if training_pool_only:
        sub = sub.filter(pl.col("in_training_pool"))
    rates = sub.group_by(["season", "position"]).agg(
        pl.col("breakout").mean().alias("rate"), pl.len().alias("n")
    )
    return rates.sort(["season", "position"])


def _print_rate_table(rates: pl.DataFrame, title: str) -> None:
    print(f"\n{title}")
    positions = SKILL_POSITIONS
    seasons = sorted(rates.get_column("season").unique().to_list())
    header = f"  {'season':>6} " + " ".join(f"{p:>14}" for p in positions)
    print(header)
    for season in seasons:
        cells = []
        for pos in positions:
            row = rates.filter((pl.col("season") == season) & (pl.col("position") == pos))
            if row.height == 0:
                cells.append(f"{'n/a':>14}")
            else:
                rate = row.get_column("rate").item()
                n = row.get_column("n").item()
                cells.append(f"{rate * 100:5.1f}% (n={n:<4})")
        print(f"  {season:>6} " + " ".join(cells))


def main() -> int:
    print("labels build | seasons 2014-2025")
    df, cross_check = build_labels()
    print(f"\nwrote {OUT_PATH} | {df.height:,} rows x {df.width} cols")

    _print_rate_table(positive_rate_table(df, training_pool_only=False), "Breakout positive rate (games>=1):")
    _print_rate_table(
        positive_rate_table(df, training_pool_only=True), "Breakout positive rate (in_training_pool only):"
    )

    print("\nNamed canaries:")
    for season, name in _NAMED_CANARIES:
        row = df.filter((pl.col("season") == season) & (pl.col("player_name") == name))
        if row.height == 0:
            print(f"  {season} {name}: NOT FOUND")
            continue
        r = row.row(0, named=True)
        print(
            f"  {season} {name}: pos={r['position']} games={r['games']} ppr_ppg={r['ppr_ppg']:.2f} "
            f"finish_pos_rank={r['finish_pos_rank']} expectation_pos_rank={r['expectation_pos_rank']} "
            f"adp_source={r['adp_source']} breakout={r['breakout']} finish_rank_delta={r['finish_rank_delta']} "
            f"is_rookie={r['is_rookie']} in_training_pool={r['in_training_pool']}"
        )

    print("\nScoring cross-check (computed_points vs shipped fantasy_points_ppr, REG rows):")
    print(f"  mean |diff| = {cross_check.mean_abs_diff:.4f} over {cross_check.n_rows:,} rows")
    print(f"  rows with nonzero diff: {cross_check.n_nonzero_diff:,}")
    print("  top 5 discrepancies:")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(cross_check.top5)

    print("\nadp_source counts by season:")
    counts = df.group_by(["season", "adp_source"]).agg(pl.len().alias("n")).sort(["season", "adp_source"])
    with pl.Config(tbl_rows=-1):
        print(counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
