"""Player ID crosswalk and name-resolution ladder (Phase 1d).

The dynastyprocess ``ff_playerids`` map (``data/raw/ff_playerids.parquet``) is
the backbone crosswalk: it carries gsis_id (nflverse's key), fantasypros_id,
sleeper_id and a handful of others for the same player row. Every other
source in this pipeline (market consensus, ADP, projections) ships its own
id or a free-text player name, so getting everything onto gsis_id is the one
join every downstream phase depends on.

``build_id_map`` applies the documented manual corrections in
``configs/id_overrides.csv`` on top of the raw crosswalk (see that file's
header for the override actions). It does **not** filter by position: the
upstream ``ff_playerids`` table stamps a retired player's position as "XX"
(their current-day non-roster status, not their playing position), so a
skill-position filter applied before matching silently drops every retired
QB/RB/WR/TE from the crosswalk — including an exact fantasypros_id match,
which needs no position at all. ``SKILL_POSITIONS`` is applied downstream,
at the point where a source table's own rows are restricted (e.g. the ECR
snapshot in ``src.ingest.adp``), never inside the crosswalk itself.

``match_to_gsis`` resolves an arbitrary source dataframe to gsis_id through a
three-rung ladder, each rung only attempting rows the previous rung couldn't
place:

  1. fantasypros_id exact join (when the source carries one — cheap and
     unambiguous, FantasyPros ids are stable). No position constraint: an
     exact id match is an exact id match regardless of what position either
     side lists. A crosswalk row that owns the id but has no gsis_id (never
     backfilled upstream, or an override cleared it) is not a match — it
     falls through to rungs 2-3 like any other miss.
  2. normalized-name exact join, constrained to the same position, accepted
     only when it resolves to a single crosswalk row (a same-name collision
     at this position falls through rather than guessing). A crosswalk row
     with position XX (or null) satisfies any source position — it carries
     no real positional information to constrain against.
  3. rapidfuzz token_sort_ratio fuzzy match (>=93), constrained to position
     (XX/null crosswalk rows pass any position, as in rung 2) and, when the
     candidate's draft_year is known, to within 15 years of the target
     season (guards against two same-named players a decade apart).
     Accepted only when the best candidate clears the threshold with a
     >=2-point margin over the runner-up — anything closer is genuinely
     ambiguous and left unmatched rather than guessed.

Every fuzzy acceptance and every failure (unmatched or ambiguous) is
appended to ``data/id_map_review.csv`` for manual audit — silent fuzzy
guesses are exactly the failure mode this ladder exists to avoid.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import polars as pl
from rapidfuzz import fuzz, process

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_DIR = REPO_ROOT / "configs"
OVERRIDES_PATH = CONFIG_DIR / "id_overrides.csv"
PLAYERIDS_PATH = RAW_DIR / "ff_playerids.parquet"
REVIEW_PATH = REPO_ROOT / "data" / "id_map_review.csv"

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

# ff_playerids stamps a retired player's *current* status as "XX", not their
# playing position. Such a row carries no real positional signal, so it is
# treated as a wildcard: it satisfies a position constraint against any
# source position rather than being excluded or mismatched.
_WILDCARD_POSITIONS = {"XX", None}

FUZZY_THRESHOLD = 93.0
FUZZY_MARGIN = 2.0
DRAFT_YEAR_WINDOW = 15

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCT_RE = re.compile(r"[.’']")
_WS_RE = re.compile(r"\s+")

_ID_COLUMNS = ["mfl_id", "gsis_id", "fantasypros_id", "sleeper_id"]

_REVIEW_COLUMNS = [
    "source_name",
    "normalized_name",
    "position",
    "season",
    "method",
    "matched_gsis_id",
    "matched_name",
    "score",
    "runner_up_score",
]


def normalize_name(name: str | None) -> str:
    """Collapse a display name to a join-safe key.

    Periods and apostrophes are deleted outright ("D.J. Moore" -> "dj moore",
    "De'Von Achane" -> "devon achane") since they sit inside a token rather
    than between name parts. Hyphens are treated as separators and become
    spaces ("Ray-Ray" -> "ray ray"). A single trailing generational suffix
    (jr/sr/ii/iii/iv/v) is dropped so "Odell Beckham Jr." and a hypothetical
    unsuffixed "Odell Beckham" resolve to the same key.
    """
    if name is None:
        return ""
    s = name.lower().strip()
    s = _PUNCT_RE.sub("", s)
    s = s.replace("-", " ")
    tokens = _WS_RE.split(s.strip())
    tokens = [t for t in tokens if t]
    if tokens and tokens[-1] in _SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def _normalize_id(value: Any) -> str | None:
    """Canonicalize an id value to a plain digit string regardless of source dtype.

    ``nfl.load_ff_playerids()`` schema-infers id columns as Int64 on a normal
    (non-proxied) machine; this environment's CSV fallback reads everything
    as Utf8. A source table's own id column can arrive as either, or as
    Float64 if it round-tripped through pandas/Excel (12459 -> "12459.0").
    Every id comparison in this module goes through this function so all of
    those converge on one representation ("12459") before comparing.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        f = float(s)
    except ValueError:
        return s
    if f.is_integer():
        return str(int(f))
    return s


def _apply_overrides(df: pl.DataFrame, overrides_path: Path = OVERRIDES_PATH) -> pl.DataFrame:
    if not overrides_path.exists():
        return df
    ov = pl.read_csv(overrides_path, comment_prefix="#", infer_schema_length=0)
    if ov.is_empty():
        return df

    clears = ov.filter(pl.col("action") == "clear_gsis").get_column("mfl_id").to_list()
    sets = ov.filter(pl.col("action") == "set_gsis")

    # Compare mfl_id as strings on both sides. The CSV side is always Utf8
    # (infer_schema_length=0 above), but the parquet side's dtype depends on
    # which source produced it: nflreadpy's own loader yields Int64, the
    # fallback_parquet_url's file yields Utf8. An Int64-vs-Utf8 is_in matches
    # nothing, silently skipping every documented override -- caught when a
    # local (non-proxied) environment pulled via nflreadpy for the first time
    # and tests/test_ingest.py's duplicate-gsis guard tripped.
    mfl_as_str = pl.col("mfl_id").cast(pl.Utf8)
    if clears:
        df = df.with_columns(
            pl.when(mfl_as_str.is_in([str(c) for c in clears]))
            .then(None)
            .otherwise(pl.col("gsis_id"))
            .alias("gsis_id")
        )
    for mfl_id, value in zip(sets.get_column("mfl_id").to_list(), sets.get_column("value").to_list()):
        df = df.with_columns(
            pl.when(mfl_as_str == str(mfl_id))
            .then(pl.lit(value))
            .otherwise(pl.col("gsis_id"))
            .alias("gsis_id")
        )
    return df


def build_id_map(
    playerids_path: Path = PLAYERIDS_PATH,
    overrides_path: Path = OVERRIDES_PATH,
) -> pl.DataFrame:
    """Build the full gsis crosswalk (all positions) with manual overrides applied.

    Deliberately unfiltered by position — see module docstring. Callers that
    want QB/RB/WR/TE-only rows filter ``position`` on the result themselves.
    """
    df = pl.read_parquet(playerids_path)
    # Converge both loader paths (nflreadpy's Int64-inferred schema and this
    # environment's all-Utf8 CSV fallback) onto one dtype before anything
    # else touches these columns — an is_in()/== against a mismatched dtype
    # raises rather than silently returning false.
    df = df.with_columns(pl.col(c).cast(pl.Utf8) for c in _ID_COLUMNS if c in df.columns)
    df = _apply_overrides(df, overrides_path)

    df = df.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("merge_name"),
        pl.col("draft_year").cast(pl.Int64, strict=False).alias("draft_year"),
    )

    return df.select(
        "mfl_id",
        "gsis_id",
        "fantasypros_id",
        "sleeper_id",
        "name",
        "merge_name",
        "position",
        "draft_year",
        "birthdate",
    )


def _append_review(rows: list[dict[str, Any]], review_path: Path = REVIEW_PATH) -> None:
    if not rows:
        return
    review_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not review_path.exists()
    with open(review_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_REVIEW_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def reset_review(review_path: Path = REVIEW_PATH) -> None:
    """Delete the review log so a fresh build run starts clean."""
    if review_path.exists():
        review_path.unlink()


def match_to_gsis(
    df: pl.DataFrame,
    *,
    name_col: str,
    pos_col: str,
    fp_id_col: str | None = None,
    season: int | None = None,
    crosswalk: pl.DataFrame | None = None,
    review_path: Path = REVIEW_PATH,
) -> pl.DataFrame:
    """Resolve every row of ``df`` to a gsis_id via the fp_id -> name -> fuzzy ladder.

    Returns ``df`` with ``gsis_id`` and ``match_method`` columns attached
    (values: fantasypros_id, exact_name, fuzzy, ambiguous, unmatched).
    """
    cw = crosswalk if crosswalk is not None else build_id_map()
    n = df.height
    gsis: list[str | None] = [None] * n
    method: list[str] = ["unmatched"] * n

    norm_names = [normalize_name(v) for v in df.get_column(name_col).to_list()]
    positions = df.get_column(pos_col).to_list()
    source_names = df.get_column(name_col).to_list()

    remaining = set(range(n))

    # Rung 1: fantasypros_id exact join. Both sides go through _normalize_id
    # so an Int64-inferred crosswalk column (a normal machine, no CSV
    # fallback) still matches a Utf8 or Float64 probe column.
    if fp_id_col is not None and fp_id_col in df.columns:
        fp_lookup: dict[str, tuple[str | None, str]] = {}
        for fp_id, gid, cname in zip(
            cw.get_column("fantasypros_id").to_list(),
            cw.get_column("gsis_id").to_list(),
            cw.get_column("name").to_list(),
        ):
            key = _normalize_id(fp_id)
            if key is not None:
                fp_lookup.setdefault(key, (gid, cname))

        fp_ids = df.get_column(fp_id_col).to_list()
        for i in list(remaining):
            key = _normalize_id(fp_ids[i])
            if key is None:
                continue
            hit = fp_lookup.get(key)
            # A crosswalk row can carry a fantasypros_id with no gsis_id
            # (dynastyprocess never backfilled one, or an override cleared
            # it) — that's not a match, it's a dead end. Leave the row in
            # `remaining` so rungs 2-3 get a shot, and if they also miss it
            # is logged as unmatched below rather than silently accepted
            # with a null gsis_id.
            if hit is not None and hit[0] is not None:
                gsis[i] = hit[0]
                method[i] = "fantasypros_id"
                remaining.discard(i)

    # Rung 2: normalized-name exact join, constrained to position (XX/null
    # crosswalk rows are wildcards, see module docstring), unique only.
    name_exact: dict[str, list[tuple[str | None, str, str | None]]] = {}
    for mname, pos, gid, cname in zip(
        cw.get_column("merge_name").to_list(),
        cw.get_column("position").to_list(),
        cw.get_column("gsis_id").to_list(),
        cw.get_column("name").to_list(),
    ):
        name_exact.setdefault(mname, []).append((gid, cname, pos))

    for i in list(remaining):
        pos = positions[i]
        pool = name_exact.get(norm_names[i], [])
        candidates = [c for c in pool if c[2] == pos or c[2] in _WILDCARD_POSITIONS]
        if len(candidates) == 1:
            gsis[i] = candidates[0][0]
            method[i] = "exact_name"
            remaining.discard(i)

    # Rung 3: rapidfuzz fuzzy match, constrained to position (XX/null
    # crosswalk rows are wildcards, + draft-year window).
    review_rows: list[dict[str, Any]] = []
    if remaining:
        by_pos: dict[str, list[tuple[str, str | None, str, int | None]]] = {}
        wildcard: list[tuple[str, str | None, str, int | None]] = []
        for mname, pos, gid, cname, dyear in zip(
            cw.get_column("merge_name").to_list(),
            cw.get_column("position").to_list(),
            cw.get_column("gsis_id").to_list(),
            cw.get_column("name").to_list(),
            cw.get_column("draft_year").to_list(),
        ):
            if pos in _WILDCARD_POSITIONS:
                wildcard.append((mname, gid, cname, dyear))
            else:
                by_pos.setdefault(pos, []).append((mname, gid, cname, dyear))

        for i in remaining:
            pos = positions[i]
            candidates = by_pos.get(pos, []) + wildcard
            if season is not None:
                candidates = [
                    c for c in candidates if c[3] is None or abs(c[3] - season) <= DRAFT_YEAR_WINDOW
                ]
            query = norm_names[i]
            if not candidates or not query:
                review_rows.append(
                    {
                        "source_name": source_names[i],
                        "normalized_name": query,
                        "position": pos,
                        "season": season,
                        "method": "unmatched",
                        "matched_gsis_id": None,
                        "matched_name": None,
                        "score": None,
                        "runner_up_score": None,
                    }
                )
                continue

            choices = [c[0] for c in candidates]
            hits = process.extract(query, choices, scorer=fuzz.token_sort_ratio, limit=2)
            best_score = hits[0][1] if hits else 0.0
            best_idx = hits[0][2] if hits else None
            runner_score = hits[1][1] if len(hits) > 1 else 0.0

            if (
                best_idx is not None
                and best_score >= FUZZY_THRESHOLD
                and (best_score - runner_score) >= FUZZY_MARGIN
            ):
                gid, cname = candidates[best_idx][1], candidates[best_idx][2]
                gsis[i] = gid
                method[i] = "fuzzy"
                review_rows.append(
                    {
                        "source_name": source_names[i],
                        "normalized_name": query,
                        "position": pos,
                        "season": season,
                        "method": "fuzzy",
                        "matched_gsis_id": gid,
                        "matched_name": cname,
                        "score": best_score,
                        "runner_up_score": runner_score,
                    }
                )
            elif best_idx is not None and best_score >= FUZZY_THRESHOLD:
                cname = candidates[best_idx][2]
                method[i] = "ambiguous"
                review_rows.append(
                    {
                        "source_name": source_names[i],
                        "normalized_name": query,
                        "position": pos,
                        "season": season,
                        "method": "ambiguous",
                        "matched_gsis_id": candidates[best_idx][1],
                        "matched_name": cname,
                        "score": best_score,
                        "runner_up_score": runner_score,
                    }
                )
            else:
                method[i] = "unmatched"
                review_rows.append(
                    {
                        "source_name": source_names[i],
                        "normalized_name": query,
                        "position": pos,
                        "season": season,
                        "method": "unmatched",
                        "matched_gsis_id": candidates[best_idx][1] if best_idx is not None else None,
                        "matched_name": candidates[best_idx][2] if best_idx is not None else None,
                        "score": best_score if best_idx is not None else None,
                        "runner_up_score": runner_score if len(hits) > 1 else None,
                    }
                )

    _append_review(review_rows, review_path)

    return df.with_columns(
        pl.Series("gsis_id", gsis, dtype=pl.Utf8),
        pl.Series("match_method", method, dtype=pl.Utf8),
    )
