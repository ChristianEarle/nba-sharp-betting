"""Unit + coverage guards for the Phase 1d ID crosswalk.

The crosswalk build reads the gitignored raw cache, so the build-dependent
tests skip when it is absent (fresh clone / CI), matching the pattern in
test_ingest.py.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.ingest.id_map import PLAYERIDS_PATH, SKILL_POSITIONS, build_id_map, normalize_name


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
