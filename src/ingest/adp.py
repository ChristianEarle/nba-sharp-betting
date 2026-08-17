"""Preseason market-expectation table (Phase 1b).

Builds ``data/processed/market_expectation.parquet``: one row per
(season, player) carrying the *preseason* FantasyPros PPR consensus rank —
the market's prior on a player before a single regular-season snap is
played. This is the baseline Phase 3 breakout labels get compared against;
a "breakout" is a player who outproduces this expectation, not merely one
who produces a lot.

Source: ``data/raw/ff_ecr.parquet`` (dynastyprocess's archived FantasyPros
ECR scrapes, Phase 1a/1d). Restricted to the PPR overall cheatsheet pages:

    (page_type == 'redraft-overall' & fp_page == '/nfl/rankings/ppr-cheatsheets.php')
    OR (page_type == 'redraft-offense' & fp_page contains 'ppr-cheatsheets')

with ``ecr_type == 'ro'`` (rest-of-offense... in practice the only ecr_type
present on these pages throughout). The archive has nothing usable before
2019-12-27, so labeled seasons start at 2020.

For each season S, the eligible snapshot window is [June 1 of S, first REG
game of S) — free-agency/draft dust has settled but no games have been
played yet — and we take the **latest** snapshot in that window as the
"final preseason consensus". A season with no snapshot in that window (no
ECR coverage, or the schedule for S isn't cached) is left out entirely
rather than backfilling with an adjacent season's numbers.

Within the chosen snapshot: restrict to QB/RB/WR/TE (the ECR pages also
carry K/DST, irrelevant to this model), dedupe by FantasyPros player id
keeping the row with the best (lowest) ecr, then rank within position
(``pos_rank``, 1 = best) off that overall ecr (``ecr_overall``).

Rows are resolved to gsis_id via ``src.ingest.id_map.match_to_gsis`` on the
FantasyPros id (rung 1 of the ladder — nearly everything resolves there).
Unmatched rows are **kept** with a null gsis_id rather than dropped, so they
surface as a coverage gap instead of silently vanishing; see
``data/id_map_review.csv`` for every fuzzy/failed resolution.

Manual escape hatch: a file at ``data/external/adp/<season>.csv`` (see that
directory's README for the schema) *replaces* the ECR snapshot for that
season entirely, with ``adp_source='manual'``.

Seasons 2014-2019 are intentionally absent from this table. The ECR archive
doesn't reach that far back, and there is no equivalent-quality market
signal for it in this pipeline. Phase 2 labeling fills those seasons with a
prior-season-finish proxy (``adp_source='proxy'``, not implemented here).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.ingest.id_map import match_to_gsis, normalize_name, reset_review

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
EXTERNAL_ADP_DIR = REPO_ROOT / "data" / "external" / "adp"

ECR_PATH = RAW_DIR / "ff_ecr.parquet"
SCHEDULES_PATH = RAW_DIR / "schedules.parquet"
OUT_PATH = PROCESSED_DIR / "market_expectation.parquet"

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

# ECR archive coverage begins here; seasons before this have no usable
# preseason snapshot regardless of the June-1 window.
FIRST_SEASON = 2020
LAST_SEASON = 2026

_OUT_COLUMNS = [
    "season",
    "gsis_id",
    "player_name",
    "normalized_name",
    "position",
    "ecr_overall",
    "pos_rank",
    "scrape_date",
    "adp_source",
    "match_method",
]


@dataclass
class SeasonCoverage:
    season: int
    adp_source: str
    snapshot_date: str | None
    n_players: int
    n_by_pos: dict[str, int]
    top200_match_rate: float | None


def _load_ecr_preseason(path: Path = ECR_PATH) -> pl.DataFrame:
    df = pl.read_parquet(path)
    df = df.filter(
        (
            (pl.col("page_type") == "redraft-overall")
            & (pl.col("fp_page") == "/nfl/rankings/ppr-cheatsheets.php")
        )
        | (
            (pl.col("page_type") == "redraft-offense")
            & pl.col("fp_page").str.contains("ppr-cheatsheets")
        )
    ).filter(pl.col("ecr_type") == "ro")
    return df.with_columns(pl.col("scrape_date").str.to_date())


def _first_reg_dates(path: Path = SCHEDULES_PATH) -> dict[int, object]:
    df = pl.read_parquet(path).filter(pl.col("game_type") == "REG")
    df = df.with_columns(pl.col("gameday").str.to_date())
    first = df.group_by("season").agg(pl.col("gameday").min().alias("first_reg"))
    return dict(zip(first.get_column("season").to_list(), first.get_column("first_reg").to_list()))


def _snapshot_for_season(ecr: pl.DataFrame, season: int, first_reg) -> pl.DataFrame | None:
    """Latest ECR snapshot in [June 1 of `season`, `first_reg`), or None."""
    import datetime as dt

    window_start = dt.date(season, 6, 1)
    candidates = ecr.filter(
        (pl.col("scrape_date") >= window_start) & (pl.col("scrape_date") < first_reg)
    )
    if candidates.is_empty():
        return None
    snap_date = candidates.get_column("scrape_date").max()
    return candidates.filter(pl.col("scrape_date") == snap_date)


def _rank_snapshot(snap: pl.DataFrame, *, ecr_col: str, id_col: str) -> pl.DataFrame:
    """Restrict to skill positions, dedupe by id (best ecr wins), assign pos_rank."""
    snap = snap.filter(pl.col("pos").is_in(SKILL_POSITIONS))
    snap = snap.sort(ecr_col).unique(subset=[id_col], keep="first")
    snap = snap.with_columns(
        pl.col(ecr_col).rank(method="ordinal").over("pos").cast(pl.Int64).alias("pos_rank")
    )
    return snap


def _load_manual(season: int) -> pl.DataFrame | None:
    path = EXTERNAL_ADP_DIR / f"{season}.csv"
    if not path.exists():
        return None
    df = pl.read_csv(path)
    missing = {"player", "position", "rank"} - set(df.columns)
    if missing:
        raise ValueError(f"data/external/adp/{season}.csv missing columns: {missing}")
    df = df.with_columns(pl.col("rank").cast(pl.Float64))
    df = df.sort("rank").with_columns(
        pl.col("rank").rank(method="ordinal").over("position").cast(pl.Int64).alias("pos_rank")
    )
    return df.rename({"rank": "ecr_overall", "position": "pos"})


def build_season(season: int, ecr: pl.DataFrame, first_reg_dates: dict) -> pl.DataFrame | None:
    manual = _load_manual(season)
    if manual is not None:
        ranked = manual.with_columns(pl.lit(None, dtype=pl.Date).alias("scrape_date"))
        adp_source = "manual"
        fp_id_col = None
    else:
        first_reg = first_reg_dates.get(season)
        if first_reg is None:
            print(f"  season {season}: no schedules coverage, skipping")
            return None
        snap = _snapshot_for_season(ecr, season, first_reg)
        if snap is None:
            print(f"  season {season}: no ECR snapshot in [{season}-06-01, {first_reg}), skipping")
            return None
        ranked = _rank_snapshot(snap, ecr_col="ecr", id_col="id")
        ranked = ranked.rename({"ecr": "ecr_overall"})
        adp_source = "ecr"
        fp_id_col = "id"

    ranked = ranked.rename({"player": "player_name", "pos": "position"})
    ranked = ranked.with_columns(
        pl.col("player_name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("normalized_name"),
        pl.lit(season).alias("season"),
        pl.lit(adp_source).alias("adp_source"),
    )

    matched = match_to_gsis(
        ranked, name_col="player_name", pos_col="position", fp_id_col=fp_id_col, season=season
    )

    return matched.select(
        "season",
        "gsis_id",
        "player_name",
        "normalized_name",
        "position",
        pl.col("ecr_overall").cast(pl.Float64),
        "pos_rank",
        "scrape_date",
        "adp_source",
        "match_method",
    )


def build_market_expectation(
    *,
    ecr_path: Path = ECR_PATH,
    schedules_path: Path = SCHEDULES_PATH,
    out_path: Path = OUT_PATH,
) -> pl.DataFrame:
    reset_review()
    ecr = _load_ecr_preseason(ecr_path)
    first_reg_dates = _first_reg_dates(schedules_path)

    parts = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        part = build_season(season, ecr, first_reg_dates)
        if part is not None:
            parts.append(part)

    if not parts:
        raise RuntimeError("no seasons produced market-expectation rows")

    out = pl.concat(parts, how="vertical_relaxed").select(_OUT_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return out


def coverage_report(df: pl.DataFrame) -> list[SeasonCoverage]:
    rows = []
    for season in sorted(df.get_column("season").unique().to_list()):
        sub = df.filter(pl.col("season") == season)
        adp_source = sub.get_column("adp_source").unique().to_list()[0]
        snap_dates = sub.get_column("scrape_date").unique().to_list()
        snapshot_date = str(snap_dates[0]) if snap_dates and snap_dates[0] is not None else None
        n_by_pos = {
            p: sub.filter(pl.col("position") == p).height for p in SKILL_POSITIONS
        }
        top200 = sub.sort("ecr_overall").head(200)
        rate = (
            top200.filter(pl.col("gsis_id").is_not_null()).height / top200.height
            if top200.height
            else None
        )
        rows.append(
            SeasonCoverage(
                season=season,
                adp_source=adp_source,
                snapshot_date=snapshot_date,
                n_players=sub.height,
                n_by_pos=n_by_pos,
                top200_match_rate=rate,
            )
        )
    return rows


def main() -> int:
    print(f"market_expectation build | seasons {FIRST_SEASON}-{LAST_SEASON}")
    df = build_market_expectation()
    print(f"\nwrote {OUT_PATH} | {df.height:,} rows x {df.width} cols")

    print("\nPer-season coverage:")
    print(
        f"  {'season':>6} {'source':>7} {'snapshot':>10} {'n':>5} "
        f"{'QB':>4} {'RB':>4} {'WR':>4} {'TE':>4} {'top200 match%':>14}"
    )
    for row in coverage_report(df):
        rate = f"{row.top200_match_rate * 100:.1f}%" if row.top200_match_rate is not None else "n/a"
        print(
            f"  {row.season:>6} {row.adp_source:>7} {str(row.snapshot_date):>10} {row.n_players:>5} "
            f"{row.n_by_pos.get('QB', 0):>4} {row.n_by_pos.get('RB', 0):>4} "
            f"{row.n_by_pos.get('WR', 0):>4} {row.n_by_pos.get('TE', 0):>4} {rate:>14}"
        )

    overall_top200 = df.sort(["season", "ecr_overall"]).group_by("season", maintain_order=True).head(200)
    overall_rate = overall_top200.filter(pl.col("gsis_id").is_not_null()).height / overall_top200.height
    print(f"\noverall top-200 match rate across seasons: {overall_rate * 100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
