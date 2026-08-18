"""Expanding-window CV folds for Phase 4 modeling (WR only).

Two disjoint entry points, deliberately kept apart:

- ``validation_folds()`` — the only thing the tuning path may see. Four
  expanding-window folds, training season always ``2014..val_season-1``:
  val 2020, 2021, 2022, 2023.
- ``holdout_split()`` — 2024 and 2025, evaluated exactly once after every
  hyperparameter/blend/calibration choice is frozen. It is a *separate*
  function with a *separate* return type (``HoldoutSplit``, not ``Fold``)
  so nothing that consumes ``validation_folds()`` output can accidentally
  be pointed at a holdout year — there is no shared "give me any season"
  parameter connecting the two. ``HOLDOUT_SEASONS`` is a module-level
  constant, not a caller-supplied argument, anywhere in this file.

Every ``Fold`` asserts ``max(train_seasons) < val_season`` at construction
(the brief's fold-boundary invariant) — a fold that violated it could never
be built, so the invariant holds structurally, not just by test.
"""

from __future__ import annotations

from dataclasses import dataclass

# First season with usable N-1 features (features_wr.parquet starts 2014 —
# see src/features/wr.py). Kept as a module constant, not hardcoded inline,
# so both entry points below agree on where "all of history" starts.
TRAIN_START_SEASON = 2014

# The four expanding-window validation seasons (brief, Deliverable 1).
VALIDATION_SEASONS: tuple[int, ...] = (2020, 2021, 2022, 2023)

# Holdout years. Structurally unreachable from validation_folds(): no
# function in this module accepts a season list that could include these.
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

# v1.7 (proxy-era sensitivity, configs/model_{pos}.yaml's train_season_start):
# a fold needs at least this many training seasons to be usable at all --
# below it a single training year is too thin to fit anything meaningful.
# Only bites when train_start_season is pushed later than the default 2014
# (e.g. 2020, the "real-market-only" pool) -- with the default start every
# fold already has 6+ training seasons, so this is a no-op for every
# position that keeps the full 2014-2025 pool.
MIN_FOLD_TRAIN_YEARS = 2


@dataclass(frozen=True)
class Fold:
    """One expanding-window validation fold: train on every season strictly before val_season."""

    train_seasons: tuple[int, ...]
    val_season: int

    def __post_init__(self) -> None:
        assert self.train_seasons, "a fold must have at least one training season"
        assert max(self.train_seasons) < self.val_season, (
            f"fold-boundary invariant violated: max(train_seasons)={max(self.train_seasons)} "
            f"is not < val_season={self.val_season}"
        )


@dataclass(frozen=True)
class HoldoutSplit:
    """Train on every season through 2023; evaluate on 2024 and 2025, once."""

    train_seasons: tuple[int, ...]
    holdout_seasons: tuple[int, ...]

    def __post_init__(self) -> None:
        assert self.train_seasons, "holdout split must have at least one training season"
        assert max(self.train_seasons) < min(self.holdout_seasons), (
            f"fold-boundary invariant violated: max(train_seasons)={max(self.train_seasons)} "
            f"is not < min(holdout_seasons)={min(self.holdout_seasons)}"
        )


def validation_folds(train_start_season: int = TRAIN_START_SEASON) -> list[Fold]:
    """The tuning-path folds: (train <=2019 -> val 2020), ..., (train <=2022 -> val 2023)
    at the default ``train_start_season`` (2014) -- all four ``VALIDATION_SEASONS``.

    ``train_start_season`` only controls how far back training seasons go;
    it cannot push a fold's ``val_season`` into ``HOLDOUT_SEASONS`` because
    ``VALIDATION_SEASONS`` is fixed, not a parameter. A ``val_season`` whose
    training window (``train_start_season..val_season-1``) would have fewer
    than ``MIN_FOLD_TRAIN_YEARS`` seasons is skipped rather than raising --
    a later ``train_start_season`` (e.g. 2020, the proxy-era-sensitivity
    real-market-only pool) legitimately yields fewer, later-starting folds
    (val 2022, 2023 only) instead of crashing on an empty/one-year window.
    """
    folds = []
    for val_season in VALIDATION_SEASONS:
        train_seasons = tuple(range(train_start_season, val_season))
        if len(train_seasons) < MIN_FOLD_TRAIN_YEARS:
            continue
        folds.append(Fold(train_seasons=train_seasons, val_season=val_season))
    return folds


def holdout_split(train_start_season: int = TRAIN_START_SEASON) -> HoldoutSplit:
    """The one-shot holdout split: train <=2023, evaluate on 2024 and 2025."""
    train_end = min(HOLDOUT_SEASONS) - 1
    return HoldoutSplit(
        train_seasons=tuple(range(train_start_season, train_end + 1)),
        holdout_seasons=HOLDOUT_SEASONS,
    )
