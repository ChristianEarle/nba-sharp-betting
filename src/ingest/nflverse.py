"""nflverse data acquisition and parquet caching (Phase 1a).

Pulls pre-aggregated nflverse tables via nflreadpy and caches them to
``data/raw/``. Play-by-play is deliberately not pulled here: 13 seasons of pbp
is multiple GB, and everything Phase 3 needs is derivable from the weekly,
NGS and ff_opportunity tables. Only pull pbp later, column-restricted, for
features that genuinely have no pre-aggregated source (red zone / end zone
shares).

All loaders return polars DataFrames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

import nflreadpy as nfl

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"

# nflverse seasonal loaders that carry the season on a differently-named column.
_SEASON_COL_CANDIDATES = ("season", "draft_year", "year")


@dataclass
class PullResult:
    """Outcome of a single dataset pull."""

    name: str
    path: Path | None
    rows: int
    cols: int
    seasons_present: list[int]
    cached: bool
    error: str | None = None
    fallback_used: bool = False
    missing_seasons: list[int] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _read_csv_url(url: str) -> pl.DataFrame:
    """Fetch a CSV over HTTP into polars, honouring the environment's proxy.

    Read every column as Utf8: this path serves ID crosswalks, where columns are
    identifiers rather than quantities. Numeric inference would strip leading
    zeros and choke on the R-style ``NA`` token these files use.
    """
    import io

    import requests

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return pl.read_csv(
        io.BytesIO(resp.content),
        infer_schema_length=0,
        null_values=["NA", ""],
    )


def _read_parquet_url(url: str) -> pl.DataFrame:
    """Fetch a parquet file over HTTP into polars, honouring the environment's proxy."""
    import io

    import requests

    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return pl.read_parquet(io.BytesIO(resp.content))


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def season_range(cfg: dict[str, Any]) -> list[int]:
    s = cfg["seasons"]
    return list(range(s["start"], s["end"] + 1))


def _season_column(df: pl.DataFrame) -> str | None:
    for col in _SEASON_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _seasons_present(df: pl.DataFrame) -> list[int]:
    col = _season_column(df)
    if col is None:
        return []
    vals = df.get_column(col).drop_nulls().unique().to_list()
    return sorted(int(v) for v in vals)


def pull_dataset(
    name: str,
    spec: dict[str, Any],
    seasons: list[int],
    *,
    force: bool = False,
    raw_dir: Path = RAW_DIR,
) -> PullResult:
    """Pull one nflverse dataset and cache it to parquet.

    Cached files are reused unless ``force`` is set, so re-running is cheap and
    does not hammer the nflverse release endpoints.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{name}.parquet"

    if out.exists() and not force:
        df = pl.read_parquet(out)
        return PullResult(name, out, df.height, df.width, _seasons_present(df), cached=True)

    loader = getattr(nfl, spec["loader"])
    kwargs = dict(spec.get("kwargs") or {})

    if spec.get("seasonal", False):
        first = spec.get("first_available")
        want = [s for s in seasons if first is None or s >= first]
        if not want:
            return PullResult(name, None, 0, 0, [], cached=False, error="no seasons in range")
        kwargs["seasons"] = want

    fallback_used = False
    try:
        df = loader(**kwargs)
    except Exception as exc:  # noqa: BLE001 - report, never silently substitute
        csv_url = spec.get("fallback_csv_url")
        parquet_url = spec.get("fallback_parquet_url")
        if not csv_url and not parquet_url:
            return PullResult(
                name, None, 0, 0, [], cached=False, error=f"{type(exc).__name__}: {exc}"
            )
        try:
            df = _read_parquet_url(parquet_url) if parquet_url else _read_csv_url(csv_url)
            fallback_used = True
        except Exception as exc2:  # noqa: BLE001
            return PullResult(
                name,
                None,
                0,
                0,
                [],
                cached=False,
                error=f"loader {type(exc).__name__}: {exc} | fallback {type(exc2).__name__}: {exc2}",
            )

    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)

    # Non-seasonal tables (draft picks, combine) come back full-history; clip
    # them so downstream row counts mean what they say.
    if not spec.get("seasonal", False):
        col = _season_column(df)
        if col is not None and spec.get("first_available") is not None:
            df = df.filter(pl.col(col) <= max(seasons))

    df.write_parquet(out)
    present = _seasons_present(df)

    # A seasonal loader that returns zero rows for a requested season is a
    # silent gap, not an error. Surface it rather than discovering it in Phase 3.
    missing: list[int] = []
    if spec.get("seasonal", False):
        want = kwargs.get("seasons") or []
        missing = [s for s in want if s not in present]

    return PullResult(
        name,
        out,
        df.height,
        df.width,
        present,
        cached=False,
        fallback_used=fallback_used,
        missing_seasons=missing,
    )


def pull_all(*, force: bool = False, only: list[str] | None = None) -> list[PullResult]:
    cfg = load_config()
    seasons = season_range(cfg)
    results: list[PullResult] = []

    for name, spec in cfg["datasets"].items():
        if only and name not in only:
            continue
        t0 = time.time()
        res = pull_dataset(name, spec, seasons, force=force)
        elapsed = time.time() - t0
        status = "cached" if res.cached else ("FAIL" if not res.ok else "pulled")
        detail = res.error if not res.ok else f"{res.rows:,} rows x {res.cols} cols"
        suffix = "  [via fallback URL]" if res.fallback_used else ""
        print(f"  [{status:>6}] {name:<16} {detail}{suffix}  ({elapsed:.1f}s)")

        if not res.ok and spec.get("required", False):
            print(f"           ^ REQUIRED dataset failed: {name}")
        if res.missing_seasons:
            gap = ", ".join(str(s) for s in res.missing_seasons)
            print(f"           ^ no rows returned for season(s): {gap}")
        results.append(res)

    return results


def season_row_counts(results: list[PullResult], seasons: list[int]) -> pl.DataFrame:
    """Per-season row counts for every seasonal dataset, for eyeball verification."""
    rows = []
    for res in results:
        if not res.ok or res.path is None:
            continue
        df = pl.read_parquet(res.path)
        col = _season_column(df)
        if col is None:
            continue
        counts = (
            df.group_by(col)
            .len()
            .rename({col: "season", "len": "n"})
            .with_columns(pl.col("season").cast(pl.Int64))
        )
        mapping = dict(zip(counts["season"].to_list(), counts["n"].to_list()))
        rows.append({"dataset": res.name, **{str(s): mapping.get(s, 0) for s in seasons}})

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Pull nflverse data and cache to parquet")
    ap.add_argument("--force", action="store_true", help="re-pull even if cached")
    ap.add_argument("--only", nargs="*", help="limit to named datasets")
    args = ap.parse_args()

    cfg = load_config()
    seasons = season_range(cfg)
    print(f"nflverse pull | seasons {seasons[0]}-{seasons[-1]} | cache -> {RAW_DIR}")

    results = pull_all(force=args.force, only=args.only)

    failed = [r for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} datasets ok")
    if failed:
        print("failed: " + ", ".join(f"{r.name} ({r.error})" for r in failed))

    counts = season_row_counts(results, seasons)
    if not counts.is_empty():
        print("\nPer-season row counts:")
        with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200):
            print(counts)

    return 1 if any(not r.ok for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
