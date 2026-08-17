"""Tests for the Phase 3 RB/TE/QB feature builders (src/features/{rb,te,qb}.py).

Parameterized over the three non-WR positions — WR keeps its own
tests/test_features.py untouched (same mechanism, same test bodies, just
not re-run three more times here). Build-dependent tests read the
gitignored raw/processed parquet files and skip when they're absent,
matching tests/test_features.py's pattern. Run, in order (after the WR
build's prerequisites): ``python -m src.features.te``,
``python -m src.features.rb``, ``python -m src.features.qb``.

The leakage test is the important one — see its docstring below for the
two independent mechanisms it uses (identical to WR's).
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl
import pytest

from src.features import shared as sh
from src.features.qb import NGS_PASSING_PATH, OUT_PATH as QB_OUT_PATH, build_features_qb
from src.features.rb import NGS_RUSHING_PATH, OUT_PATH as RB_OUT_PATH, build_features_rb
from src.features.te import NGS_RECEIVING_PATH, OUT_PATH as TE_OUT_PATH, build_features_te
from src.labels.build import OUT_PATH as LABELS_PATH

_REQUIRED_PATHS = [
    LABELS_PATH,
    sh.PATHS["player_stats"],
    sh.PATHS["rosters"],
    sh.PATHS["rosters_weekly"],
    sh.PATHS["draft_picks"],
    sh.PATHS["schedules"],
]

# Per-position wiring: build function, output path, optional-NGS path, and
# canary rows verified directly against data before being encoded here (see
# the brief) -- (season, player_name, must_appear, note).
_POSITIONS = {
    "rb": {
        "build_fn": build_features_rb,
        "out_path": RB_OUT_PATH,
        "ngs_path": NGS_RUSHING_PATH,
        "ngs_kwarg": "ngs_rushing",
        "core_columns": [
            "age",
            "age_sq",
            "years_exp",
            "year_in_league",
            "draft_round",
            "draft_pick",
            "log_draft_pick",
            "undrafted",
            "team_pass_att_pg_prior",
            "team_rush_att_pg_prior",
            "vacated_target_share",
            "vacated_carry_share",
            "label_season_2020",
            "carry_share_n1",
            "target_share_n1",
            "weighted_opportunity_pg_n1",
            "receptions_pg_n1",
            "rush_yards_pg_n1",
            "ppr_ppg_n1",
            "backfield_committee_count_n1",
        ],
        "canaries": [
            (2017, "Alvin Kamara", False, "rookie season, must be absent"),
            (2019, "Austin Ekeler", True, "strong receiving usage"),
        ],
    },
    "te": {
        "build_fn": build_features_te,
        "out_path": TE_OUT_PATH,
        "ngs_path": NGS_RECEIVING_PATH,
        "ngs_kwarg": "ngs_receiving",
        "core_columns": [
            "age",
            "age_sq",
            "years_exp",
            "year_in_league",
            "draft_round",
            "draft_pick",
            "log_draft_pick",
            "undrafted",
            "team_pass_att_pg_prior",
            "vacated_target_share",
            "vacated_carry_share",
            "label_season_2020",
            "target_share_n1",
            "wopr_n1",
            "targets_pg_n1",
            "ppr_ppg_n1",
            "td_rate_n1",
        ],
        "canaries": [
            (2020, "Darren Waller", True, "elite shares"),
            (2021, "Mark Andrews", True, "elite shares"),
        ],
    },
    "qb": {
        "build_fn": build_features_qb,
        "out_path": QB_OUT_PATH,
        "ngs_path": NGS_PASSING_PATH,
        "ngs_kwarg": "ngs_passing",
        "core_columns": [
            "age",
            "age_sq",
            "years_exp",
            "year_in_league",
            "draft_round",
            "draft_pick",
            "log_draft_pick",
            "undrafted",
            "team_pass_att_pg_prior",
            "vacated_target_share",
            "vacated_carry_share",
            "supporting_cast_capital",
            "label_season_2020",
            "pass_attempts_pg_n1",
            "pass_yards_pg_n1",
            "ppr_ppg_n1",
            "rush_attempts_pg_n1",
            "rush_yards_pg_n1",
        ],
        "canaries": [
            (2019, "Lamar Jackson", True, "high N-1 rush yard share"),
            (2023, "Jalen Hurts", True, "high N-1 rush volume"),
        ],
    },
}


def _raw_available() -> bool:
    return all(p.exists() for p in _REQUIRED_PATHS)


def _skip_if_raw_missing() -> None:
    if not _raw_available():
        pytest.skip(
            "raw data not cached; run `python -m src.ingest.nflverse` then `python -m src.labels.build`"
        )


@lru_cache(maxsize=1)
def _default_frames() -> dict[str, pl.DataFrame | None]:
    """Every default input frame shared by rb/te/qb builders, loaded once and cached."""
    ff_opportunity_path = sh.RAW_DIR / "ff_opportunity.parquet"
    return {
        "labels": pl.read_parquet(LABELS_PATH),
        "player_stats": pl.read_parquet(sh.PATHS["player_stats"]),
        "ff_opportunity": pl.read_parquet(ff_opportunity_path),
        "rosters": pl.read_parquet(sh.PATHS["rosters"]),
        "rosters_weekly": pl.read_parquet(sh.PATHS["rosters_weekly"]),
        "draft_picks": pl.read_parquet(sh.PATHS["draft_picks"]),
        "schedules": pl.read_parquet(sh.PATHS["schedules"]),
        "snap_counts": pl.read_parquet(sh.PATHS["snap_counts"]) if sh.PATHS["snap_counts"].exists() else None,
        "ngs_rushing": pl.read_parquet(NGS_RUSHING_PATH) if NGS_RUSHING_PATH.exists() else None,
        "ngs_receiving": pl.read_parquet(NGS_RECEIVING_PATH) if NGS_RECEIVING_PATH.exists() else None,
        "ngs_passing": pl.read_parquet(NGS_PASSING_PATH) if NGS_PASSING_PATH.exists() else None,
        "coaching_changes": sh.load_coaching_changes(),
    }


def _build_kwargs(pos: str) -> dict:
    """The subset of _default_frames() a given position's build_features_* accepts."""
    frames = dict(_default_frames())
    spec = _POSITIONS[pos]
    common = {
        "labels": frames["labels"],
        "player_stats": frames["player_stats"],
        "ff_opportunity": frames["ff_opportunity"],
        "rosters": frames["rosters"],
        "rosters_weekly": frames["rosters_weekly"],
        "draft_picks": frames["draft_picks"],
        "schedules": frames["schedules"],
        "coaching_changes": frames["coaching_changes"],
    }
    if pos in ("rb", "te"):
        common["snap_counts"] = frames["snap_counts"]
    common[spec["ngs_kwarg"]] = frames[spec["ngs_kwarg"]]
    return common


def _build(pos: str, **overrides) -> pl.DataFrame:
    kwargs = _build_kwargs(pos)
    kwargs.update(overrides)
    return _POSITIONS[pos]["build_fn"](**kwargs)


def _df(pos: str) -> pl.DataFrame:
    out_path = _POSITIONS[pos]["out_path"]
    if not out_path.exists():
        pytest.skip(f"{out_path.name} not built; run `python -m src.features.{pos}`")
    return pl.read_parquet(out_path)


_ALL_POS = sorted(_POSITIONS)


# --------------------------------------------------------------------------
# Structural leakage test — the important one. Same two-mechanism approach
# as tests/test_features.py::test_leakage_structural_2023_stripped_matches_full_build:
# build season-2023 rows twice, once from the full player_stats frame and
# once from a copy with every 2023 row deleted, and assert byte-identical
# output. player_stats is the only season-N-shaped input a season-N-labeled
# feature row could possibly read game-level 2023 data from.
# --------------------------------------------------------------------------


def _assert_frames_close(a: pl.DataFrame, b: pl.DataFrame, *, context: str) -> None:
    """Column-wise equality, tolerant of float64 summation-order noise (~1e-6).

    Identical tolerance mechanism to tests/test_features.py's helper —
    polars aggregation order is not guaranteed stable across two
    structurally different input frames.
    """
    assert a.columns == b.columns, f"{context}: column sets differ"
    assert a.height == b.height, f"{context}: row counts differ ({a.height} vs {b.height})"
    for col in a.columns:
        ca, cb = a.get_column(col), b.get_column(col)
        if ca.dtype in (pl.Float32, pl.Float64):
            diff = (ca - cb).abs()
            both_null = ca.is_null() & cb.is_null()
            bad = (~both_null) & ((diff > 1e-6) | (ca.is_null() != cb.is_null()))
            assert bad.sum() == 0, f"{context}: column {col!r} differs beyond float tolerance ({bad.sum()} rows)"
        else:
            assert ca.equals(cb), f"{context}: column {col!r} differs"


@pytest.mark.parametrize("pos", _ALL_POS)
def test_leakage_structural_2023_stripped_matches_full_build(pos: str) -> None:
    _skip_if_raw_missing()

    full = _build(pos).filter(pl.col("season") == 2023).sort(["season", "gsis_id"])
    if full.height == 0:
        pytest.skip(f"no season-2023 {pos.upper()} rows to check")

    stripped_stats = _default_frames()["player_stats"].filter(pl.col("season") != 2023)
    stripped = _build(pos, player_stats=stripped_stats).filter(pl.col("season") == 2023).sort(["season", "gsis_id"])

    _assert_frames_close(
        full, stripped, context=f"{pos}: season-2023 feature rows changed when 2023 player_stats rows were removed"
    )


# --------------------------------------------------------------------------
# Uniqueness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", _ALL_POS)
def test_no_duplicate_season_gsis_id(pos: str) -> None:
    df = _df(pos)
    dupes = df.group_by(["season", "gsis_id"]).len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"{pos}: duplicate (season, gsis_id) rows: {dupes}"


# --------------------------------------------------------------------------
# Population correctness: rookies excluded, non-rookies at this position
# match labels.parquet exactly.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", _ALL_POS)
def test_rookie_seasons_excluded(pos: str) -> None:
    df = _df(pos)
    labels = pl.read_parquet(LABELS_PATH)
    label_pos = pos.upper()
    rookie_rows = labels.filter((pl.col("position") == label_pos) & pl.col("is_rookie")).select("season", "gsis_id")
    overlap = df.join(rookie_rows, on=["season", "gsis_id"], how="inner")
    assert overlap.is_empty(), f"{pos}: rookie seasons leaked into features_{pos}: {overlap.select('season', 'gsis_id')}"


@pytest.mark.parametrize("pos", _ALL_POS)
def test_non_rookie_population_matches_labels(pos: str) -> None:
    df = _df(pos)
    labels = pl.read_parquet(LABELS_PATH)
    label_pos = pos.upper()
    expected = labels.filter((pl.col("position") == label_pos) & (~pl.col("is_rookie"))).select("season", "gsis_id")
    assert df.height == expected.height
    missing = expected.join(df, on=["season", "gsis_id"], how="anti")
    assert missing.is_empty(), f"{pos}: non-rookie {label_pos} labels rows missing from features_{pos}: {missing.height}"


# --------------------------------------------------------------------------
# Nullability: zero all-null core columns.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pos", _ALL_POS)
def test_no_all_null_core_columns(pos: str) -> None:
    df = _df(pos)
    for col in _POSITIONS[pos]["core_columns"]:
        assert df.get_column(col).null_count() < df.height, f"{pos}: {col} is entirely null across the whole output"


# --------------------------------------------------------------------------
# Named sanity canaries — verified against data first (see the brief), then
# encoded here as a presence/absence check.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pos,season,name,should_appear,note",
    [(pos, season, name, should_appear, note) for pos, spec in _POSITIONS.items() for season, name, should_appear, note in spec["canaries"]],
)
def test_named_canaries(pos: str, season: int, name: str, should_appear: bool, note: str) -> None:
    df = _df(pos)
    row = df.filter((pl.col("season") == season) & (pl.col("player_name") == name))
    if should_appear:
        assert row.height == 1, f"{pos} {season} {name} ({note}) should appear in features_{pos} but does not"
    else:
        assert row.height == 0, f"{pos} {season} {name} ({note}) should NOT appear in features_{pos} but does"


def test_kamara_2018_present_year_two() -> None:
    """Complements the 2017-absent canary above: Kamara's very next season must be present."""
    df = _df("rb")
    row = df.filter((pl.col("season") == 2018) & (pl.col("player_name") == "Alvin Kamara"))
    assert row.height == 1


def test_hurts_2020_absent_rookie() -> None:
    """Complements the 2023-present canary above: Hurts' rookie season must be absent."""
    df = _df("qb")
    row = df.filter((pl.col("season") == 2020) & (pl.col("player_name") == "Jalen Hurts"))
    assert row.height == 0
