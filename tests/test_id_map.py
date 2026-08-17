"""Unit + coverage guards for the Phase 1d ID crosswalk.

The crosswalk build reads the gitignored raw cache, so the build-dependent
tests skip when it is absent (fresh clone / CI), matching the pattern in
test_ingest.py.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.ingest.id_map import PLAYERIDS_PATH, SKILL_POSITIONS, build_id_map, match_to_gsis, normalize_name


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Odell Beckham Jr.", "odell beckham"),
        ("D.J. Moore", "dj moore"),
        ("Kenneth Walker III", "kenneth walker"),
        ("De'Von Achane", "devon achane"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_normalize_name_collapses_whitespace_and_case() -> None:
    assert normalize_name("  Mike   WILLIAMS  ") == "mike williams"


def test_normalize_name_none_is_empty() -> None:
    assert normalize_name(None) == ""


def _crosswalk() -> pl.DataFrame:
    if not PLAYERIDS_PATH.exists():
        pytest.skip(f"{PLAYERIDS_PATH.name} not cached; run `python -m src.ingest.nflverse`")
    return build_id_map()


def test_override_clears_gsis_for_mfl_12459() -> None:
    """mfl_id 12459 (Fred Williams) is documented in configs/id_overrides.csv

    as colliding with a different player's gsis_id; the build must null it
    rather than carry the wrong player's id forward.
    """
    cw = _crosswalk()
    row = cw.filter(pl.col("mfl_id") == "12459")
    assert row.height == 1, "mfl_id 12459 missing from crosswalk entirely"
    assert row.get_column("gsis_id").to_list() == [None]


def test_no_duplicate_gsis_id_within_matchable_scope() -> None:
    """No duplicate gsis_id among rows ``match_to_gsis`` can actually select.

    build_id_map() is unfiltered by position (retired players carry position
    "XX", not their playing position — see module docstring), so the raw
    crosswalk now surfaces ~9 duplicate-gsis_id pairs that were previously
    invisible because they were filtered out before this test ever saw them
    (e.g. a CB and an RB both named Tony Carter sharing one gsis_id). Those
    are real upstream data quality issues, but they are inert here: neither
    side of a CB/DT/S/LB/DE/DL/PN pair is ever a match_to_gsis candidate for
    a QB/RB/WR/TE query (rungs 2-3 constrain by position, and XX/null is the
    only wildcard). What must stay unique is the set match_to_gsis actually
    draws from: SKILL_POSITIONS rows plus XX/null wildcard rows. The Fred
    Williams override (configs/id_overrides.csv, mfl_id 12459) is the one
    documented collision inside that scope; this asserts nothing else has
    joined it.
    """
    cw = _crosswalk()
    matchable = cw.filter(
        pl.col("gsis_id").is_not_null()
        & (pl.col("position").is_in(SKILL_POSITIONS) | pl.col("position").is_in(["XX"]) | pl.col("position").is_null())
    )
    dupes = matchable.group_by("gsis_id").len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), (
        f"duplicate gsis_id within the skill+XX matchable scope: {dupes.get_column('gsis_id').to_list()}. "
        "Verify against nflverse and add to configs/id_overrides.csv."
    )


def test_crosswalk_is_not_filtered_by_position() -> None:
    """build_id_map must keep every position, including retired players' "XX".

    A skill-position filter applied before matching silently drops every
    retired QB/RB/WR/TE from the crosswalk (ff_playerids stamps a retired
    player's *current* roster status as position "XX", not their playing
    position) — including rows an exact fantasypros_id join would otherwise
    resolve with no position information needed at all.
    """
    cw = _crosswalk()
    positions = set(cw.get_column("position").unique().to_list())
    assert set(SKILL_POSITIONS) <= positions, "skill positions missing from crosswalk"
    assert "XX" in positions, "crosswalk no longer carries retired-player XX rows"


def test_xx_position_rows_have_resolvable_gsis_ids() -> None:
    """Locks the no-position-filter behavior: XX rows must reach the output

    with a real gsis_id, not just exist as an empty stub row.
    """
    cw = _crosswalk()
    xx_with_gsis = cw.filter((pl.col("position") == "XX") & pl.col("gsis_id").is_not_null())
    assert xx_with_gsis.height > 0, "no XX-position row has a non-null gsis_id"


def test_build_id_map_survives_int64_id_columns(tmp_path) -> None:
    """On a normal machine (no CSV-fallback proxy), ``nfl.load_ff_playerids()``

    schema-infers id columns as Int64. Everything downstream (overrides'
    ``is_in``/``==`` comparisons, rung-1 lookups) must still work against
    that dtype exactly as it does against this environment's all-Utf8 CSV
    fallback — both paths are expected to converge to the same crosswalk.
    """
    raw = pl.DataFrame(
        {
            "mfl_id": [12459, 99999],
            "gsis_id": ["00-0031320", None],
            "fantasypros_id": [None, 5555],
            "sleeper_id": [111, None],
            "name": ["Fred Williams", "Some Player"],
            "position": ["WR", "RB"],
            "draft_year": ["2014", "2020"],
            "birthdate": ["1988-04-15", None],
        }
    ).with_columns(
        pl.col("mfl_id").cast(pl.Int64),
        pl.col("fantasypros_id").cast(pl.Int64),
        pl.col("sleeper_id").cast(pl.Int64),
    )
    playerids_path = tmp_path / "ff_playerids.parquet"
    raw.write_parquet(playerids_path)

    overrides_path = tmp_path / "id_overrides.csv"
    overrides_path.write_text(
        "mfl_id,action,value,name,reason\n12459,clear_gsis,,Fred Williams,test override\n"
    )

    cw = build_id_map(playerids_path=playerids_path, overrides_path=overrides_path)

    assert cw.schema["mfl_id"] == pl.Utf8
    assert cw.schema["gsis_id"] == pl.Utf8
    assert cw.schema["fantasypros_id"] == pl.Utf8
    assert cw.schema["sleeper_id"] == pl.Utf8

    row = cw.filter(pl.col("mfl_id") == "12459")
    assert row.height == 1
    assert row.get_column("gsis_id").to_list() == [None], "Int64 mfl_id override did not apply"


def test_match_to_gsis_rung1_hits_across_int64_and_string_id_dtypes(tmp_path) -> None:
    """Rung 1 must match regardless of which side (crosswalk or source) has

    an integer-typed fantasypros_id and which has a string/float one — the
    _normalize_id canonicalization has to make all of them converge.
    """
    cw = pl.DataFrame(
        {
            "mfl_id": ["1"],
            "gsis_id": ["00-0012345"],
            "fantasypros_id": [4242],
            "sleeper_id": ["1"],
            "name": ["Test Player"],
            "merge_name": ["test player"],
            "position": ["WR"],
            "draft_year": [2018],
            "birthdate": [None],
        }
    ).with_columns(pl.col("fantasypros_id").cast(pl.Int64))

    src = pl.DataFrame(
        {"player": ["Test Player", "Other Player"], "pos": ["WR", "WR"], "fp_id": ["4242", "4242.0"]}
    )

    out = match_to_gsis(
        src,
        name_col="player",
        pos_col="pos",
        fp_id_col="fp_id",
        crosswalk=cw,
        review_path=tmp_path / "review.csv",
    )
    assert out.get_column("gsis_id").to_list() == ["00-0012345", "00-0012345"]
    assert out.get_column("match_method").to_list() == ["fantasypros_id", "fantasypros_id"]


def test_match_to_gsis_rung1_falls_through_on_null_gsis(tmp_path) -> None:
    """A crosswalk row that owns the fantasypros_id but has no gsis_id is not

    a match — it must fall through to rungs 2-3 rather than being accepted
    with a null gsis_id under match_method='fantasypros_id'.
    """
    cw = pl.DataFrame(
        {
            "mfl_id": ["1"],
            "gsis_id": [None],
            "fantasypros_id": ["4242"],
            "sleeper_id": ["1"],
            "name": ["Test Player"],
            "merge_name": ["test player"],
            "position": ["WR"],
            "draft_year": [2018],
            "birthdate": [None],
        },
        schema_overrides={"gsis_id": pl.Utf8},
    )
    src = pl.DataFrame({"player": ["Test Player"], "pos": ["WR"], "fp_id": ["4242"]})

    out = match_to_gsis(
        src,
        name_col="player",
        pos_col="pos",
        fp_id_col="fp_id",
        crosswalk=cw,
        review_path=tmp_path / "review.csv",
    )
    assert out.get_column("gsis_id").to_list() == [None]
    # Falls through past rung 1 to rung 2 (unique normalized-name match on
    # the same crosswalk row) — still no gsis_id to offer, but the method
    # correctly reflects that rung 1's id hit was not accepted as a match.
    assert out.get_column("match_method").to_list() == ["exact_name"]
