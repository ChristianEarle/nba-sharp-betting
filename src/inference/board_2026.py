"""2026 breakout board: veteran scoring + rookie heuristic + market overlay (Phase 6).

Builds ``outputs/breakout_board_2026.csv`` / ``.md`` — the v1 deliverable —
from three independent pieces, kept structurally separate throughout this
module so it stays obvious which numbers are model output and which are
presentation:

1. **Veteran scoring** (``build_veteran_feature_matrix`` / ``score_veterans_batch``):
   reuses the *exact* Phase-3 feature builders (``src.features.{wr,rb,te,qb}``)
   on a synthetic 2026 "labels" population instead of the real
   ``labels.parquet`` — every builder only ever reads its ``labels`` argument
   for the target population (season, gsis_id, player_name, position,
   is_rookie), so handing it a 2026 population built from ``rosters.parquet``
   (years_exp >= 1, i.e. non-rookie) drives the identical N-1-shifted,
   leakage-safe feature logic Phase 3/4 already tested, with zero new
   feature code. Scored with the saved holdout-retrained bundles
   (``data/models/{pos}_model_bundle.joblib``): the classifier trio blended
   and calibrated exactly as ``src.models.train.holdout_predictions`` does,
   and the regression head (``expected_rank_delta``) as the unweighted mean
   of ``lgbm_reg``/``xgb_reg`` (see ``src.explain.shap_report.regression_prediction``'s
   docstring for why no blend weight exists for that head).
2. **Rookie heuristic** (``src.models.rookie_heuristic``): a separate, much
   simpler model for 2026's drafted QB/RB/WR/TE class, kept in its own
   board section and its own probability column — never merged onto the
   veteran calibration scale, which it was not fit to match.
3. **Overlay** (this module, presentation layer only — see "Overlay formula"
   below): consensus ADP (2026 ECR positional rank, already computed by
   ``src.ingest.adp``), optional manual Sleeper ADP / Vegas implied-points
   CSVs, and an availability screen. None of this ever feeds back into
   either model; it exists purely to help a reader sanity-check and act on
   the board.

2026-specific data wrinkles (verified directly against the cached data,
not assumed)
-------------------------------------------------------------------------
- ``rosters.parquet``'s 2026 snapshot spells Arizona "AZ" (every other
  season, and every other 2026 source, uses "ARI") — added to
  ``src.features.shared.TEAM_ALIASES`` rather than special-cased here.
- ``rosters_weekly`` has zero 2026 rows (the loader raises for season >
  2025 — no Week 1 has been played), so every 2026 row falls through
  ``season_roster_team``'s existing fallback path to the season-level
  ``rosters.parquet`` table with ``preseason_team_fallback=1``. For every
  *historical* season that flag signals "we used a less current-at-the-time
  proxy, treat with a little more caution" (a rare late-season-only
  signee). For 2026 it fires on *every* row for a structural reason that
  has nothing to do with data quality — there is no in-season trade this
  team snapshot could possibly be behind, because no season has been
  played — so this module overrides it to 0 right after building the
  feature matrix (see ``build_veteran_feature_matrix``), per the brief's
  "preseason_team_fallback=0 semantics equivalent (no leak possible
  preseason)."
- ``configs/coaching_changes.csv`` has no 2026 rows, so ``new_oc`` is null
  for every 2026 row — correct behavior of the existing null-when-season-
  absent rule in ``src.features.shared.attach_new_oc``, not a bug here.
  Left as a known limitation (see README) rather than guessed at.
- 2026 ``draft_picks`` ships each rookie's *own* temporary, non-nflverse
  gsis_id — see ``src.models.rookie_heuristic``'s docstring for the
  two-rung id resolution this pipeline uses instead.

Overlay formula (documented, not tuned)
------------------------------------------
``edge = probability * log1p(consensus_pos_rank)`` — a simple, monotone
ADP-discount weighting: two players at the same calibrated probability
get more "edge" credit the deeper the market already has them ranked
(log1p keeps a WR1-ranked player's near-zero discount from swamping a
WR60's). This is a stated design choice for sorting/flagging sleepers on
the board, not a fitted quantity — nothing here was tuned against
holdout data, and ``edge`` never feeds back into either model.
``adp_gap = sleeper_adp_pos_rank - consensus_pos_rank`` (positive = Sleeper's
market has him going later than ECR's preseason consensus, i.e. a
possible market inefficiency) is null whenever ``data/external/sleeper_adp_2026.csv``
is absent.

Displayed-probability clamp (v1.5: fixed at the source, clamp now a no-op safety net)
---------------------------------------------------------------------------------------
v1 scored off each bundle's plain isotonic/Platt ``calibrator``. Isotonic
regression fits a step function over pooled OOF validation predictions;
at these sample sizes its terminal buckets (the highest/lowest raw-score
bucket) can be entirely one class, which isotonic maps to an exact 0.0 or
1.0 — not evidence of certainty, just a small-bucket artifact. Verified
directly on the v1 build: hundreds of veteran rows across all four
positions landed on an exact 0.0 or 1.0 calibrated probability (mostly at
the low end, where "will not break out" is the overwhelming majority
class), which (a) reads as false certainty and (b) collapses board
ordering to ties within a saturated cluster.

v1.5 fixes this at the source: ``score_veterans_batch`` now calibrates
through each bundle's ``smoothed_calibrator``
(``src.models.train.SmoothedIsotonic`` — Laplace/rule-of-succession-
smoothed isotonic blocks, re-monotonized; see that class's docstring),
which is constructed so no block can ever output exactly 0.0 or 1.0. The
old plain ``calibration_method``/``calibrator`` are left in every bundle
untouched (never deleted) for anything else that still wants them; the
board just no longer reads them. Two presentation-layer mechanisms from
v1 are kept as belt-and-suspenders, now expected to be no-ops:

- ``probability`` is still clamped to ``[PROB_DISPLAY_LO, PROB_DISPLAY_HI]``
  = ``[0.01, 0.95]`` for display — a safety net, not the fix.
  ``probability_saturated`` (bool) still flags any row whose *pre-clamp*
  calibrated value was exactly 0.0 or 1.0; ``main()`` prints (and
  ``tests/test_inference.py`` asserts) that this is 0 rows on a v1.5 board.
- ``raw_score`` (the pre-calibration blended classifier score) is still
  kept as the secondary sort key, in case a future recalibration
  reintroduces ties.

v2.0: quantile projections + two lenses
------------------------------------------
Alongside the v1.7 classifier scoring above, every position with a saved
``{pos}_quantile_bundle.joblib`` (``src.models.quantile``) is also scored
for its full ppg distribution; ``src.inference.projections`` turns that into
``floor_ppg``/``expected_ppg``/``ceiling_ppg`` (q10/q50/q90),
``predicted_finish_rank``, the outcome ladder (``p_elite``/``p_startable``/
``p_useful``, honesty-renormalized the same way ``probability`` already is),
``value_gap``, and ``breakout_eligible``. Neither model is replaced by the
other -- ``src.inference.projections.load_engine_decision`` (Deliverable 3's
pre-stated comparison gate) says, per position, whether the "Breakout hunt"
lens ranks by the classifier's ``probability`` or the quantile head's
``p_startable``; the "Full projections" lens (ALL veterans, ranked by
``expected_ppg``) is the same regardless of that decision, since it isn't a
breakout-odds ranking at all. A position with no quantile bundle yet
degrades gracefully: its quantile-derived columns are null and its
"Breakout hunt" lens keeps ranking by the classifier's probability, exactly
the v1.7 board.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.explain import shap_report as shp
from src.features import qb as feat_qb
from src.features import rb as feat_rb
from src.features import shared as sh
from src.features import te as feat_te
from src.features import wr as feat_wr
from src.inference import projections as pj
from src.ingest.id_map import build_id_map, match_to_gsis, normalize_name
from src.labels.build import PLAYER_STATS_PATH, add_expectation, load_labels_config, load_scoring_config
from src.models import quantile as qmod
from src.models import rookie_heuristic as rh
from src.models import train

REPO_ROOT = train.REPO_ROOT
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = train.PROCESSED_DIR
OUTPUTS_DIR = train.OUTPUTS_DIR
CONFIG_DIR = train.CONFIG_DIR
EXTERNAL_DIR = REPO_ROOT / "data" / "external"

SEASON = 2026
MARKET_EXPECTATION_PATH = PROCESSED_DIR / "market_expectation.parquet"
AVAILABILITY_PATH = CONFIG_DIR / "availability_2026.csv"
SLEEPER_ADP_PATH = EXTERNAL_DIR / "sleeper_adp_2026.csv"
VEGAS_IMPLIED_PATH = EXTERNAL_DIR / "vegas_implied_2026.csv"

BOARD_CSV_PATH = OUTPUTS_DIR / "breakout_board_2026.csv"
BOARD_MD_PATH = OUTPUTS_DIR / "breakout_board_2026.md"

TOP_DRIVERS_ON_BOARD = 3

# Displayed-probability clamp (presentation only) -- see module docstring's
# "Displayed-probability clamp" section for why.
PROB_DISPLAY_LO = 0.01
PROB_DISPLAY_HI = 0.95

_POSITION_BUILD = {
    "wr": {
        "build_fn": feat_wr.build_features_wr,
        "ngs_kwarg": "ngs_receiving",
        "ngs_path": feat_wr.NGS_RECEIVING_PATH,
        "needs_snap": True,
        "needs_pbp": True,
        "needs_ftn": True,  # v2.2: catchable_target_rate (src.features.shared.ftn_catchable_target_rate_table)
    },
    "rb": {
        "build_fn": feat_rb.build_features_rb,
        "ngs_kwarg": "ngs_rushing",
        "ngs_path": feat_rb.NGS_RUSHING_PATH,
        "needs_snap": True,
        "needs_pbp": True,
        "needs_ftn": True,
    },
    "te": {
        "build_fn": feat_te.build_features_te,
        "ngs_kwarg": "ngs_receiving",
        "ngs_path": feat_te.NGS_RECEIVING_PATH,
        "needs_snap": True,
        "needs_pbp": True,
        "needs_ftn": True,
    },
    "qb": {
        "build_fn": feat_qb.build_features_qb,
        "ngs_kwarg": "ngs_passing",
        "ngs_path": feat_qb.NGS_PASSING_PATH,
        "needs_snap": False,
        "needs_pbp": False,  # QB gets no pbp-derived features, per the brief (skip QB for Phase C)
        "needs_ftn": False,  # QB gets no v2.2 efficiency-proxy columns either -- see src.models.efficiency_gate
    },
}


def veteran_features_path(pos: str) -> Path:
    return PROCESSED_DIR / f"features_2026_{pos}.parquet"


ROOKIE_FEATURES_PATH = PROCESSED_DIR / "features_2026_rookies.parquet"


# --------------------------------------------------------------------------
# Raw frames (loaded once, shared across all four positions)
# --------------------------------------------------------------------------


def load_raw_frames() -> dict:
    ff_opportunity_path = RAW_DIR / "ff_opportunity.parquet"
    pbp_path = RAW_DIR / "pbp.parquet"
    ftn_charting_path = RAW_DIR / "ftn_charting.parquet"
    return {
        "player_stats": pl.read_parquet(PLAYER_STATS_PATH),
        "ff_opportunity": pl.read_parquet(ff_opportunity_path),
        "rosters": pl.read_parquet(sh.PATHS["rosters"]),
        "rosters_weekly": pl.read_parquet(sh.PATHS["rosters_weekly"]),
        "draft_picks": pl.read_parquet(sh.PATHS["draft_picks"]),
        "schedules": pl.read_parquet(sh.PATHS["schedules"]),
        "snap_counts": pl.read_parquet(sh.PATHS["snap_counts"]) if sh.PATHS["snap_counts"].exists() else None,
        "ngs_receiving": pl.read_parquet(RAW_DIR / "ngs_receiving.parquet") if (RAW_DIR / "ngs_receiving.parquet").exists() else None,
        "ngs_rushing": pl.read_parquet(RAW_DIR / "ngs_rushing.parquet") if (RAW_DIR / "ngs_rushing.parquet").exists() else None,
        "ngs_passing": pl.read_parquet(RAW_DIR / "ngs_passing.parquet") if (RAW_DIR / "ngs_passing.parquet").exists() else None,
        "coaching_changes": sh.load_coaching_changes(),
        "combine": pl.read_parquet(RAW_DIR / "combine.parquet"),
        "market_expectation": pl.read_parquet(MARKET_EXPECTATION_PATH),
        # v1.5 (Phase C): pbp is required:false and has zero *season-2026* rows by
        # construction (no games played yet -- see configs/data.yaml's last_available
        # cap). When pbp.parquet exists, the 2026 board's rz/ez/goal-line share
        # features are populated from real season-2025 pbp via the same N-1 shift
        # every other feature uses (identical to player_stats-derived features,
        # which are likewise "absent for 2026 itself, populated via N-1"). If
        # pbp.parquet is entirely absent (never pulled, or pulled and then
        # removed), every rz/ez/goal-line feature -- historical AND 2026 -- is
        # null: None here exercises that exact pbp-absent code path
        # src.features.{wr,rb,te} were built against (see their build_raw_stat_table
        # docstrings), the same "nullable by design" contract as snap_counts/ngs_*.
        "pbp": pl.read_parquet(pbp_path) if pbp_path.exists() else None,
        # v2.2: FTN charting data (catchable_target_rate) -- 2022+ coverage only, joined
        # onto pbp by play id (src.features.shared.ftn_catchable_target_rate_table).
        # Optional, same "nullable, never a raised error" contract as pbp/snap_counts/ngs_*.
        "ftn_charting": pl.read_parquet(ftn_charting_path) if ftn_charting_path.exists() else None,
    }


# --------------------------------------------------------------------------
# Veteran population + feature matrix
# --------------------------------------------------------------------------


def veteran_population(rosters: pl.DataFrame, position: str) -> pl.DataFrame:
    """season=2026 non-rookie population for one position: on the 2026 roster,

    years_exp >= 1 (the preseason-knowable rookie signal — see
    src.features.shared.age_and_experience's docstring for why years_exp,
    not a derived label, is the source of truth). Shaped like the `labels`
    argument every build_features_{pos} expects: season, gsis_id,
    player_name, position, is_rookie.
    """
    pop = rosters.filter(
        (pl.col("season") == SEASON) & (pl.col("position") == position.upper()) & (pl.col("years_exp") >= 1)
    )
    return pop.select(
        pl.col("season"),
        pl.col("gsis_id"),
        pl.col("full_name").alias("player_name"),
        pl.lit(position.upper()).alias("position"),
        pl.lit(False).alias("is_rookie"),
    ).unique(subset=["season", "gsis_id"])


def attach_2026_expectation(df: pl.DataFrame, position: str, market_expectation: pl.DataFrame) -> pl.DataFrame:
    """expectation_pos_rank + adp_source for season 2026, reusing src.labels.build.add_expectation

    verbatim (ecr for anyone the ECR snapshot ranked, capped one slot behind
    the deepest ECR rank at that position for anyone it didn't) — the
    identical rule label-building applies historically, just for a season
    with no finish yet to build a proxy from (irrelevant here: 2026 >=
    ECR_FIRST_SEASON, so the proxy branch never activates). ``build_features_{pos}``'s
    output carries no "position" column of its own (it's a single-position table), so
    it's supplied by the caller rather than read off df.
    """
    agg = df.select("season", "gsis_id").with_columns(
        pl.lit(position.upper()).alias("position"), pl.lit(None, dtype=pl.Int64).alias("finish_pos_rank")
    )
    expect = add_expectation(agg, market_expectation, label_seasons=range(SEASON, SEASON + 1))
    return df.join(expect.select("season", "gsis_id", "expectation_pos_rank", "adp_source"), on=["season", "gsis_id"], how="left")


def _attach_adp_source_dummies(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Reproduces src.models.train.load_modeling_frame's adp_source one-hot columns exactly

    (same category list, same dropped baseline) so the bundle's tree_feature_cols line
    up 1:1 against this frame's columns.
    """
    categories = cfg["adp_source_categories"]
    baseline = cfg["adp_source_baseline"]
    for cat in categories:
        if cat == baseline:
            continue
        df = df.with_columns((pl.col("adp_source") == cat).cast(pl.Float64).alias(f"adp_source_{cat}"))
    return df


def build_veteran_feature_matrix(pos: str, raw: dict, bundle: dict) -> pl.DataFrame:
    """The full 2026 feature matrix for one position: population -> Phase-3 builder ->

    expectation/adp_source -> adp_source dummies -> preseason_team_fallback override.
    Saved to veteran_features_path(pos) so src.explain.shap_report's --season 2026
    lookups can reuse it without rebuilding.
    """
    spec = _POSITION_BUILD[pos]
    labels_2026 = veteran_population(raw["rosters"], pos)

    kwargs = dict(
        labels=labels_2026,
        player_stats=raw["player_stats"],
        ff_opportunity=raw["ff_opportunity"],
        rosters=raw["rosters"],
        rosters_weekly=raw["rosters_weekly"],
        draft_picks=raw["draft_picks"],
        schedules=raw["schedules"],
        coaching_changes=raw["coaching_changes"],
    )
    if spec["needs_snap"]:
        kwargs["snap_counts"] = raw["snap_counts"]
    if spec["needs_pbp"]:
        kwargs["pbp"] = raw["pbp"]
    if spec.get("needs_ftn"):
        kwargs["ftn_charting"] = raw["ftn_charting"]
    kwargs[spec["ngs_kwarg"]] = raw[spec["ngs_kwarg"]]

    out = spec["build_fn"](**kwargs)
    out = attach_2026_expectation(out, pos, raw["market_expectation"])
    out = _attach_adp_source_dummies(out, bundle["cfg"])

    # See module docstring: rosters_weekly has no 2026 rows at all, so every
    # row landed on season_roster_team's season-level fallback -- for a
    # season with literally no games played yet that is not a data-quality
    # signal the way it is historically, so it is normalized to 0 here.
    out = out.with_columns(pl.lit(0).alias("preseason_team_fallback"))

    veteran_features_path(pos).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(veteran_features_path(pos))
    return out


# --------------------------------------------------------------------------
# Scoring (vectorized batch — see src.explain.shap_report for the identical
# per-row versions the --why CLI uses)
# --------------------------------------------------------------------------


def score_veterans_batch(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Attaches `probability_calibrated_raw` (Platt-calibrated, un-renormalized), `raw_score`

    (the pre-calibration ENSEMBLE blended classifier score -- see below), `probability`
    (the v1.7 base-rate-renormalized display probability, attached later by
    ``attach_renormalized_probability`` once every position's frame is on hand),
    `probability_saturated` (bool safety-net flag, expected to stay 0 -- see "Honest
    probabilities" below), and `expected_rank_delta` to every row.

    v1.7 seed ensemble + Platt (replaces v1.5's SmoothedIsotonic as the active method):
    ``bundle["per_seed_models"]`` carries one LGBM+XGB pair per Optuna seed
    (``src.models.train``'s ``optuna.seeds``), each blended with that seed's own frozen
    ``bundle["per_seed"][seed]["blend_weights"]`` against the ONE shared logistic head
    (``bundle["logistic"]``); ``raw_score`` is the mean of those per-seed blended scores
    (mirrors ``src.models.train.holdout_predictions`` exactly). ``bundle["active_calibrator"]``
    (Platt, sigmoid, strictly monotone) turns that into ``probability_calibrated_raw`` --
    monotone means the calibrated ranking is identical to the raw ranking by construction,
    unlike v1.5's isotonic-family calibrators (piecewise-constant, ties within a block).
    The final displayed ``probability`` is a separate presentation-layer renormalization
    (see ``attach_renormalized_probability``'s docstring) applied after every position is
    scored, not part of this function.
    """
    tree_cols = bundle["tree_feature_cols"]
    # Backward compatibility: a pre-v1.7 (single-model) bundle, or one built by a
    # module out of this task's scope (src.models.overlay_backtest, which loads a saved
    # bundle and calls this function directly), may not carry "per_seed_models" /
    # "active_calibrator" at all -- fall back to a synthetic single-seed ensemble (the
    # legacy bundle["lgbm"]/["xgb"]/["blend_weights"]) and the legacy calibrator as
    # "active", exactly the same fallback src.models.train.holdout_predictions applies.
    per_seed_models = bundle.get("per_seed_models") or {bundle.get("seed", 0): {"lgbm": bundle["lgbm"], "xgb": bundle["xgb"]}}
    per_seed_weights = bundle.get("per_seed") or {bundle.get("seed", 0): {"blend_weights": bundle["blend_weights"]}}
    active_calibration_method = bundle.get("active_calibration_method", bundle.get("calibration_method"))
    active_calibrator = bundle.get("active_calibrator", bundle.get("calibrator"))

    # The logistic head is SHARED across every seed -- fit once, at the base seed, and
    # never varies with it (see the docstring above / train.holdout_predictions' own
    # docstring: "Shared logistic head: fit once, at the base seed -- unchanged from
    # pre-ensemble"). Its prediction is therefore seed-INVARIANT: computing it inside the
    # per-seed loop below would recompute the identical array once per seed for no
    # reason (train.holdout_predictions already hoists this the same way). Computed at
    # most once, up front, and reused by every seed's blend.
    logit_pred = None
    if any(per_seed_weights[seed].get("blend_weights", {}).get("logistic", 0) > 0 for seed in per_seed_models):
        model, imputer, scaler = bundle["logistic"]
        logit_cols = bundle["logistic_feature_cols"]
        X = scaler.transform(imputer.transform(df[logit_cols]))
        logit_pred = model.predict_proba(X)[:, 1]

    seed_scores = []
    for seed, models in per_seed_models.items():
        weights = per_seed_weights[seed]["blend_weights"]
        preds = {}
        if weights.get("lgbm", 0) > 0:
            preds["lgbm"] = models["lgbm"].predict_proba(df[tree_cols])[:, 1]
        if weights.get("xgb", 0) > 0:
            preds["xgb"] = models["xgb"].predict_proba(df[tree_cols])[:, 1]
        if weights.get("logistic", 0) > 0:
            preds["logistic"] = logit_pred
        nonzero_weights = {k: w for k, w in weights.items() if k in preds}
        seed_scores.append(train.apply_blend(nonzero_weights, **preds))
    blended = np.mean(np.vstack(seed_scores), axis=0)
    calibrated_raw = train.apply_calibration(active_calibration_method, active_calibrator, blended)

    lgbm_reg_pred = bundle["lgbm_reg"].predict(df[tree_cols])
    xgb_reg_pred = bundle["xgb_reg"].predict(df[tree_cols])

    out = df.copy()
    out["raw_score"] = blended
    out["probability_saturated"] = (np.isclose(calibrated_raw, 0.0) | np.isclose(calibrated_raw, 1.0)).astype(int)
    out["probability_calibrated_raw"] = np.clip(calibrated_raw, PROB_DISPLAY_LO, PROB_DISPLAY_HI)
    # Legacy alias: callers that use this function standalone, without the board-level
    # renormalization pass (src.models.overlay_backtest -- out of this task's scope to
    # rewrite) read `probability` and expect the pre-renormalization calibrated value,
    # exactly the v1.5 contract. build_veteran_board's real board pipeline overwrites
    # this column with the renormalized display value via attach_renormalized_probability
    # once every position's frame is on hand.
    out["probability"] = out["probability_calibrated_raw"]
    out["expected_rank_delta"] = (lgbm_reg_pred + xgb_reg_pred) / 2.0
    return out


# --------------------------------------------------------------------------
# v1.7: base-rate renormalization (board/inference time; replaces v1.5's
# "displayed-probability clamp" / SmoothedIsotonic write-up -- see this
# module's docstring's "Honest probabilities" section for why).
#
# A calibrated probability (even Platt's, even SmoothedIsotonic's) is only
# ever calibrated against the historical OOF validation pool it was fit on
# -- nothing forces the SUM of a position's 2026 calibrated probabilities to
# equal how many breakouts that position typically produces in a season.
# Verified directly: pre-renormalization, several positions' veteran pools
# summed to 2-3x their historical per-season breakout count, which reads as
# systematically overconfident even though every individual probability is
# a legitimate calibrated (and, since v1.7, strictly rank-preserving) output.
#
# The fix is a per-position linear rescale at *board* time, never touching
# either model: scale = historical_mean_breakouts / sum(calibrated p over
# that position's scored 2026 veterans); displayed_probability =
# min(0.97, p * scale). This is presentation-layer only (same spirit as the
# old clamp it replaces) -- ``probability_calibrated_raw`` (pre-rescale) and
# ``raw_score`` (pre-calibration) are both still stored, so nothing is lost,
# and because the rescale is a single positive per-position constant it
# cannot change within-position ranking (it can only shrink or grow every
# probability in that position by the same factor, then the 0.97 cap can
# only ever bind for the top of the pool).
# --------------------------------------------------------------------------

# Displayed-probability cap (post-renormalization safety net -- a rescale
# could in principle push an already-near-1.0 probability slightly past a
# believable ceiling; capped, not clamped from below, since renormalization
# is expected to shrink most positions, not inflate them).
PROB_DISPLAY_CAP = 0.97

# Tier multiples (v1.7): base rate for a position = historical_mean_breakouts
# / n_scored_2026_veterans at that position (a probability, not a count --
# "if you drafted every scored veteran at this position, what fraction would
# break out"). A player's tier is their displayed probability's multiple of
# that number -- "N times a normal late pick's odds" -- not a fixed absolute
# cutoff, so it stays meaningful across positions with very different base
# rates (QB's is much higher than TE's, historically).
TIER_THRESHOLDS = (
    (8.0, "Elite target"),
    (4.0, "Strong swing"),
    (2.0, "Worth a flier"),
)
TIER_FALLBACK = "Long shot"


def historical_mean_breakouts_by_position() -> dict[str, float]:
    """{position: mean per-season breakout count, 2014-2025} from labels.parquet --

    computed fresh every call, never hardcoded (see module docstring's
    "Honest probabilities" section). Every (position, season) pair in
    labels.parquet's season range counts toward the mean, including a
    season with zero breakouts at that position -- dropping zero-breakout
    seasons would inflate the mean.
    """
    labels = pl.read_parquet(train.LABELS_PATH)
    counts = (
        labels.filter(pl.col("breakout") == 1)
        .group_by(["position", "season"])
        .agg(pl.len().alias("n"))
    )
    grid = labels.select("position").unique().join(labels.select("season").unique(), how="cross")
    full = grid.join(counts, on=["position", "season"], how="left").with_columns(pl.col("n").fill_null(0))
    means = full.group_by("position").agg(pl.col("n").mean().alias("mean_breakouts"))
    return {row["position"]: float(row["mean_breakouts"]) for row in means.to_dicts()}


def _tier_for_multiple(multiple: float) -> str:
    if pd.isna(multiple):
        return TIER_FALLBACK
    for threshold, label in TIER_THRESHOLDS:
        if multiple >= threshold:
            return label
    return TIER_FALLBACK


def attach_renormalized_probability(board: pd.DataFrame, mean_breakouts: dict[str, float]) -> pd.DataFrame:
    """Per position: scale so displayed probabilities sum to that position's historical

    mean per-season breakout count, cap at ``PROB_DISPLAY_CAP``, and attach a base-rate-
    multiple tier -- see the module-level "v1.7: base-rate renormalization" comment above
    for the full rationale. Adds ``probability`` (displayed), ``renorm_scale``,
    ``base_rate`` (position's implied per-player breakout probability = mean_breakouts /
    n_scored), ``base_rate_multiple`` (``probability / base_rate``), and ``tier``.
    """
    board = board.copy()
    for col in ("probability", "renorm_scale", "base_rate", "base_rate_multiple"):
        board[col] = np.nan
    board["tier"] = TIER_FALLBACK

    for pos in board["pos"].unique():
        mask = (board["pos"] == pos).to_numpy()
        raw_p = board.loc[mask, "probability_calibrated_raw"]
        total_raw = float(raw_p.sum())
        n_scored = int(mask.sum())
        target = mean_breakouts.get(pos, np.nan)
        scale = (target / total_raw) if (total_raw > 0 and pd.notna(target)) else 1.0
        displayed = np.minimum(PROB_DISPLAY_CAP, raw_p * scale)
        base_rate = (target / n_scored) if (n_scored > 0 and pd.notna(target)) else np.nan

        board.loc[mask, "renorm_scale"] = scale
        board.loc[mask, "probability"] = displayed
        board.loc[mask, "base_rate"] = base_rate
        multiple = displayed / base_rate if base_rate and base_rate > 0 else pd.Series(np.nan, index=displayed.index)
        board.loc[mask, "base_rate_multiple"] = multiple
        board.loc[mask, "tier"] = multiple.apply(_tier_for_multiple)
    return board


# --------------------------------------------------------------------------
# v2.1 Deliverable 3: driver-chip honesty fix. The v2.0 chips showed a
# feature name + SHAP direction but never the feature's own VALUE -- "New
# team +" rendered identically whether the player actually changed teams
# (team_change=1) or not (team_change=0, positive SHAP for some other
# reason entirely). Every chip below is resolved to a state-aware label
# BEFORE it leaves this module -- the compact ``shap_top3`` string, the
# ``rationale`` sentence, and the dashboard's driver chips (which just
# render whatever plain-language label this module already produced; see
# ``src/dashboard/template.py``'s ``driverChips`` and its comment) all read
# the identical honest text, so there is exactly one place this can drift.
# --------------------------------------------------------------------------

# Binary/flag features: (label when 0, label when 1). Covers every 0/1
# feature that can appear as a top SHAP driver, including the v2.1
# depth/competition composites (src.features.shared).
BINARY_FEATURE_STATES: dict[str, tuple[str, str]] = {
    "team_change": ("Stayed put", "New team"),
    "new_hc": ("Coach continuity", "New coach"),
    "new_oc": ("OC continuity", "New OC"),
    "new_oc_interaction": ("OC continuity", "New OC"),
    "undrafted": ("Drafted", "Undrafted"),
    "label_season_2020": ("Normal season", "COVID-disrupted season"),
    "preseason_team_fallback": ("Live Week-1 roster", "Roster-data fallback"),
    "young_stayer": ("Not a young stayer", "Young + stayed put"),
    "moved_into_vacancy": ("No clear vacancy move", "Moved into a vacancy"),
    "became_presumptive_starter": ("Not a new QB1", "Became the presumptive starter"),
    "returning_starter": ("No proven returning starter", "Proven starter returns"),
    "adp_source_proxy": ("Real-market preseason ADP", "Proxy-era (pre-2020) expectation signal"),
    "adp_source_capped": ("Ranked in the era's expectation signal", "Capped -- no expectation signal reached him"),
}

# Feature bases where a LOWER raw value is the "good"/"elite" direction --
# everything else defaults to higher=elite. Age and draft capital read
# backwards from a plain magnitude comparison; depth_rank_derived (1 = top
# of the depth chart) and competition_draft_capital/sack_rate/int_rate are
# the same "lower is better" shape.
LOWER_IS_BETTER = {
    "age",
    "age_sq",
    "draft_pick",
    "log_draft_pick",
    "depth_rank_derived",
    "depth_rank_derived_n1",
    "competition_draft_capital",
    "sack_rate",
    "int_rate",
    "returning_incumbent_share",
    "backfield_committee_count",
}

# Level features where a plain "Elite <label>"/"Thin <label>" chip reads backwards --
# "Elite backfield competition" sounds like heavy competition is a GOOD thing, when the
# opposite is true (a crowded returning backfield/receiving corps is bad for a breakout
# candidate's opportunity). These get bespoke high/low state text instead of the generic
# Elite/Thin prefix. Keyed by the base feature (after stripping "_n1"/"_yoy_delta");
# first element = the LOW-raw-value ("good") state, second = HIGH-raw-value ("bad") state
# -- i.e. matched against LOWER_IS_BETTER's `is_elite` the same way Elite/Thin would be.
CUSTOM_LEVEL_FEATURE_STATES: dict[str, tuple[str, str]] = {
    "returning_incumbent_share": ("Light returning competition ▲", "Heavy returning competition ▼"),
    "backfield_committee_count": ("Light returning competition ▲", "Heavy returning competition ▼"),
}


def honest_state_label(feature: str, raw_value, median) -> str:
    """The value-aware chip/rationale text for `feature` at `raw_value` -- a binary flag's

    named state ("Stayed put" / "New team"), or a continuous feature's high/low descriptor
    relative to `median` ("Elite target share" / "Thin target share"; "Rising"/"Falling" for
    a ``_yoy_delta`` trend column, since "Elite ... trend" reads oddly). Falls back to the
    plain humanized feature name when the value or median is missing -- never invents a
    state the data doesn't support.
    """
    base = re.sub(r"_(n1|yoy_delta)$", "", feature)
    label = _humanize_feature(feature)
    missing = raw_value is None or (isinstance(raw_value, (int, float, np.floating, np.integer)) and pd.isna(raw_value))

    if base in BINARY_FEATURE_STATES:
        if missing:
            return label
        lo, hi = BINARY_FEATURE_STATES[base]
        return hi if float(raw_value) >= 0.5 else lo

    # _FEATURE_LABELS' phrases were written to read naturally after "boosted
    # by"/"held back by" (e.g. "a deep preseason ranking"); strip a leading
    # article before prefixing Elite/Thin/Rising/Falling so the chip doesn't
    # read "Elite a deep preseason ranking".
    bare = re.sub(r"^(a|an|the)\s+", "", label, flags=re.IGNORECASE)

    if feature.endswith("_yoy_delta"):
        # Trend features compare the player's OWN year-over-year delta to ZERO, never to
        # the position median: "Rising" iff raw_value > 0, "Falling" otherwise. Comparing
        # to the median mislabels decliners whenever the population's median delta is
        # itself negative (a below-zero player scoring above that negative median would
        # read as "Rising" despite having actually declined). No median is needed here.
        if missing:
            return label
        return f"{'Rising' if raw_value > 0 else 'Falling'} {bare}"

    # Elite/Thin (vs the position median) applies ONLY to level features.
    if missing or median is None or pd.isna(median):
        return label
    higher_is_elite = base not in LOWER_IS_BETTER
    is_elite = (raw_value >= median) if higher_is_elite else (raw_value <= median)
    if base in CUSTOM_LEVEL_FEATURE_STATES:
        lo_state, hi_state = CUSTOM_LEVEL_FEATURE_STATES[base]
        return lo_state if is_elite else hi_state
    return f"{'Elite' if is_elite else 'Thin'} {bare}"


def attach_shap_drivers(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Attach shap_top3 (honest, value-aware chip text) + _shap_top_features (feature, signed

    SHAP value, honest state label -- consumed by build_rationale below). Medians are
    computed from `df` itself: this position's own currently-scored 2026 population, i.e.
    "the position median for that season" the brief calls for.
    """
    tree_cols = bundle["tree_feature_cols"]
    medians = {c: df[c].median() for c in tree_cols}
    sv = shp.blend_weighted_shap(bundle, df[tree_cols])
    compact, top_features = [], []
    for i in range(len(df)):
        order = np.argsort(-np.abs(sv[i]))[:TOP_DRIVERS_ON_BOARD]
        parts, feats_row = [], []
        for j in order:
            feat = tree_cols[j]
            raw_val = df[feat].iloc[i]
            state_label = honest_state_label(feat, raw_val, medians.get(feat))
            sign = "+" if sv[i][j] >= 0 else "-"
            parts.append(f"{state_label}:{sign}")
            feats_row.append((feat, float(sv[i][j]), state_label))
        compact.append(", ".join(parts))
        top_features.append(feats_row)
    out = df.copy()
    out["shap_top3"] = compact
    out["_shap_top_features"] = top_features
    return out


# --------------------------------------------------------------------------
# Rationale template
# --------------------------------------------------------------------------

_FEATURE_LABELS = {
    "expectation_pos_rank": "a deep preseason ranking",
    "target_share_n1": "target share",
    "wopr_n1": "weighted opportunity",
    "targets_pg_n1": "target volume",
    "receptions_pg_n1": "reception volume",
    "rec_yards_pg_n1": "receiving yardage",
    "adot_n1": "downfield target depth",
    "ppr_ppg_n1": "per-game scoring",
    "expected_ppr_ppg_n1": "expected per-game scoring",
    "efficiency_residual_pg_n1": "efficiency residual on volume",
    "yards_per_reception_n1": "yards per catch",
    "td_rate_n1": "touchdown rate",
    "snap_share_n1": "snap share",
    "carry_share_n1": "carry share",
    "weighted_opportunity_pg_n1": "weighted opportunity",
    "rush_yards_pg_n1": "rushing volume",
    "yards_per_carry_n1": "yards per carry",
    "rush_td_rate_n1": "rushing touchdown rate",
    "backfield_committee_count_n1": "backfield competition",
    "returning_incumbent_share_n1": "returning competition",
    "pass_attempts_pg_n1": "pass-attempt volume",
    "pass_yards_pg_n1": "passing yardage",
    "rush_attempts_pg_n1": "rushing volume",
    "rush_yard_share_n1": "rush yard share",
    "sack_rate_n1": "sack rate",
    "int_rate_n1": "interception rate",
    "vacated_target_share": "vacated targets ahead",
    "vacated_carry_share": "vacated carries ahead",
    "competition_draft_capital": "positional draft competition",
    "supporting_cast_capital": "investment in his supporting cast",
    "new_hc": "a new head coach",
    "new_oc": "a new offensive coordinator",
    "new_oc_interaction": "a new offensive play-caller",
    "team_change": "a team change",
    "team_pass_att_pg_prior": "team pass volume",
    "team_plays_pg_prior": "team play volume",
    "team_pass_rate_prior": "team pass rate",
    "team_rush_att_pg_prior": "team rush volume",
    "age": "age",
    "age_sq": "age",
    "games_prior": "games played last season",
    "log_draft_pick": "draft capital",
    "draft_pick": "draft position",
    "draft_round": "draft position",
    "undrafted": "undrafted status",
    "year_in_league": "career stage",
    "avg_separation_n1": "receiving separation",
    "avg_cushion_n1": "defensive cushion faced",
    "catch_percentage_n1": "catch rate",
    "avg_time_to_throw_n1": "time to throw",
    "cpoe_n1": "completion rate over expected",
    "label_season_2020": "the 2020 COVID-era flag",
    "rz_target_share_n1": "red-zone target share",
    "ez_target_share_n1": "end-zone-adjacent target share",
    "rz_carry_share_n1": "red-zone carry share",
    "goal_line_carry_share_n1": "goal-line carry share",
    "years_exp": "years of experience",
    "air_yards_share_n1": "air yards share",
    "ngs_efficiency_n1": "Next Gen Stats rushing efficiency",
    "rush_yards_over_expected_per_att_n1": "rush yards over expected per attempt",
    # Bare (unsuffixed) fallbacks for every "_n1" concept above -- reached only via
    # `_humanize_feature`'s stripped-base lookup, i.e. for that concept's "_yoy_delta"
    # trend column (there is no bare/unsuffixed feature column in any bundle).
    "target_share": "target share",
    "wopr": "weighted opportunity",
    "targets_pg": "target volume",
    "receptions_pg": "reception volume",
    "rec_yards_pg": "receiving yardage",
    "adot": "downfield target depth",
    "ppr_ppg": "per-game scoring",
    "expected_ppr_ppg": "expected per-game scoring",
    "efficiency_residual_pg": "efficiency residual on volume",
    "yards_per_reception": "yards per catch",
    "td_rate": "touchdown rate",
    "snap_share": "snap share",
    "carry_share": "carry share",
    "weighted_opportunity_pg": "weighted opportunity",
    "rush_yards_pg": "rushing volume",
    "yards_per_carry": "yards per carry",
    "rush_td_rate": "rushing touchdown rate",
    "backfield_committee_count": "backfield competition",
    "returning_incumbent_share": "returning competition",
    "pass_attempts_pg": "pass-attempt volume",
    "pass_yards_pg": "passing yardage",
    "rush_attempts_pg": "rushing volume",
    "rush_yard_share": "rush yard share",
    "sack_rate": "sack rate",
    "int_rate": "interception rate",
    "avg_separation": "receiving separation",
    "avg_cushion": "defensive cushion faced",
    "catch_percentage": "catch rate",
    "avg_time_to_throw": "time to throw",
    "cpoe": "completion rate over expected",
    "rz_target_share": "red-zone target share",
    "ez_target_share": "end-zone-adjacent target share",
    "rz_carry_share": "red-zone carry share",
    "goal_line_carry_share": "goal-line carry share",
    "air_yards_share": "air yards share",
    "ngs_efficiency": "Next Gen Stats rushing efficiency",
    "rush_yards_over_expected_per_att": "rush yards over expected per attempt",
}


def _humanize_feature(name: str) -> str:
    # _FEATURE_LABELS is keyed by the FULL feature name as it appears in the tree/logistic
    # column lists (almost always the "_n1" form, e.g. "target_share_n1") -- try that exact
    # key first. Only fall back to the suffix-stripped base (and, failing that, the raw
    # code with underscores swapped for spaces) for keys the map doesn't carry verbatim,
    # e.g. a "_yoy_delta" trend column, which reuses its base feature's phrase + " trend".
    base = re.sub(r"_(n1|yoy_delta)$", "", name)
    label = _FEATURE_LABELS.get(name, _FEATURE_LABELS.get(base, base.replace("_", " ")))
    if name.endswith("_yoy_delta"):
        label = f"{label} trend"
    return label


def build_rationale(pos: str, row: pd.Series, top_features: list[tuple[str, float, str]]) -> str:
    """Template sentence from the top-3 SHAP drivers, e.g. "Year-2 WR: boosted by elite

    target share; boosted by vacated targets ahead; held back by thin age." -- a
    plain-language echo of ``shap_top3``'s honest, value-aware state labels (v2.1
    Deliverable 3), not a new signal and not the pre-v2.1 value-blind feature name alone.
    ``top_features`` entries are (feature, signed shap value, honest state label) -- see
    ``attach_shap_drivers``.
    """
    year = row.get("year_in_league")
    year_str = f"Year-{int(year) + 1}" if pd.notna(year) else "Veteran"
    phrases = []
    for feature, value, state_label in top_features:
        verb = "boosted by" if value >= 0 else "held back by"
        phrases.append(f"{verb} {state_label.lower()}")
    return f"{year_str} {pos.upper()}: " + "; ".join(phrases) + "."


# --------------------------------------------------------------------------
# Availability screen (manual, template-only — see configs/availability_2026.csv)
# --------------------------------------------------------------------------


def load_availability(path: Path = AVAILABILITY_PATH) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema={"norm_name": pl.Utf8, "position": pl.Utf8, "availability": pl.Utf8, "note": pl.Utf8})
    df = pl.read_csv(path, comment_prefix="#")
    if df.is_empty():
        return pl.DataFrame(schema={"norm_name": pl.Utf8, "position": pl.Utf8, "availability": pl.Utf8, "note": pl.Utf8})
    df = df.with_columns(
        pl.col("player").map_elements(normalize_name, return_dtype=pl.Utf8).alias("norm_name"),
        pl.col("position").str.strip_chars().str.to_uppercase(),
    )
    return df.select("norm_name", "position", pl.col("status").alias("availability"), "note")


def attach_availability(board: pd.DataFrame, availability: pl.DataFrame) -> pd.DataFrame:
    if availability.is_empty():
        board["availability"] = None
        return board
    board = board.copy()
    board["_norm_name"] = board["player_name"].map(normalize_name)
    avail_pd = availability.to_pandas()
    merged = board.merge(
        avail_pd[["norm_name", "position", "availability", "note"]],
        left_on=["_norm_name", "pos"],
        right_on=["norm_name", "position"],
        how="left",
    )
    merged = merged.drop(columns=["_norm_name", "norm_name", "position"], errors="ignore")
    return merged


# --------------------------------------------------------------------------
# Overlay: Sleeper ADP + Vegas implied points (optional manual CSVs)
# --------------------------------------------------------------------------


def _load_overlay_csv(path: Path, value_col: str, crosswalk: pl.DataFrame) -> pl.DataFrame:
    """player,position,<value_col> -> gsis_id, position, <value_col> via id_map.match_to_gsis.

    Returns an empty (but correctly typed) frame if the file doesn't exist —
    every downstream join against this is a left join, so absence just
    means null columns on the board (per the brief).
    """
    if not path.exists():
        return pl.DataFrame(schema={"gsis_id": pl.Utf8, "position": pl.Utf8, value_col: pl.Float64})
    df = pl.read_csv(path)
    df = df.with_columns(pl.col("position").str.strip_chars().str.to_uppercase())
    matched = match_to_gsis(df, name_col="player", pos_col="position", season=SEASON, crosswalk=crosswalk)
    return matched.select("gsis_id", "position", pl.col(value_col).cast(pl.Float64))


def attach_overlay(board: pd.DataFrame, crosswalk: pl.DataFrame) -> pd.DataFrame:
    sleeper = _load_overlay_csv(SLEEPER_ADP_PATH, "adp", crosswalk)
    vegas = _load_overlay_csv(VEGAS_IMPLIED_PATH, "implied_pts", crosswalk)

    board = board.copy()
    if not sleeper.is_empty():
        sleeper = sleeper.with_columns(
            pl.col("adp").rank(method="ordinal").over("position").cast(pl.Int64).alias("sleeper_adp_pos_rank")
        )
        sleeper_pd = sleeper.to_pandas().rename(columns={"position": "pos"})
        board = board.merge(sleeper_pd[["gsis_id", "pos", "adp", "sleeper_adp_pos_rank"]], on=["gsis_id", "pos"], how="left")
    else:
        board["adp"] = np.nan
        board["sleeper_adp_pos_rank"] = np.nan

    if not vegas.is_empty():
        vegas_pd = vegas.to_pandas().rename(columns={"position": "pos"})
        board = board.merge(vegas_pd[["gsis_id", "pos", "implied_pts"]], on=["gsis_id", "pos"], how="left")
    else:
        board["implied_pts"] = np.nan

    board["adp_gap"] = board["sleeper_adp_pos_rank"] - board["consensus_pos_rank"]
    board["edge"] = board["probability"] * np.log1p(board["consensus_pos_rank"])
    return board


# --------------------------------------------------------------------------
# v2.4: low-octane-offense risk flag (measured this session, 2020-2025 pooled:
# startable-season rate 6.7% on bottom-5 preseason-implied-total offenses vs 33.3% on
# top-5; beat-price rates flat ~48-54% across both groups, so this is a RISK flag --
# "his team's offense might not generate enough plays/points for anyone on it to have a
# big season" -- NOT a model feature; the models already rejected Vegas team features
# twice on holdout accuracy grounds (see README's "Vegas lines: tested, not used"). Also
# a real confound worth stating alongside the stat: good players tend to play for good
# offenses (or make them good), so this is not proof a specific player is a bad bet, just
# a context flag.
# --------------------------------------------------------------------------

LOW_OCTANE_N = 5


def compute_low_octane_flags(schedules: pl.DataFrame, prior_season: int = SEASON - 1) -> pd.DataFrame:
    """team -> low_octane_offense (0/1), low_octane_source ("Vegas" or "est.").

    Source A (preferred): season-``SEASON`` preseason team lines
    (``data/external/odds_api/team_lines.parquet``, via ``src.ingest.vegas.build_vegas_team``
    -- same implied-points-per-game formula ``src.features.shared.attach_vegas_team`` feeds
    the (rejected-as-a-feature) Vegas columns from). Used only if that file has at least
    ``LOW_OCTANE_N`` teams actually priced for season ``SEASON`` -- otherwise this is not a
    real snapshot to flag from, and the function falls through to Source B.

    Source B (fallback -- current repo state, no season-2026 odds_api snapshot exists yet):
    bottom-``LOW_OCTANE_N`` teams by season ``prior_season`` (2025) ACTUAL points-per-game,
    computed directly from ``schedules.parquet``'s own REG ``home_score``/``away_score``
    (every team's own team-code normalized via ``src.features.shared.normalize_team``, same
    normalization every other team-bearing source in this module goes through). Clearly
    source-labeled ("est.") wherever it's used, per the brief -- never conflated with a real
    market read.

    Returns EVERY team with a resolvable score (all 32 in practice), not just the flagged
    five, so a caller can left-join on "team" and get a real 0 for everyone else -- exactly
    ``LOW_OCTANE_N`` teams read ``low_octane_offense == 1`` from either source by
    construction (a plain ascending sort + head(N)).
    """
    from src.ingest import vegas as vg_ingest

    if vg_ingest.TEAM_LINES_PATH.exists():
        team_lines = pl.read_parquet(vg_ingest.TEAM_LINES_PATH)
        season_lines = team_lines.filter(pl.col("season") == SEASON)
        if season_lines.height > 0:
            vt = vg_ingest.build_vegas_team(season_lines).filter(pl.col("season") == SEASON)
            vt = vt.filter(pl.col("implied_ppg").is_not_null())
            if vt.height >= LOW_OCTANE_N:
                bottom = set(vt.sort("implied_ppg").head(LOW_OCTANE_N).get_column("team").to_list())
                out = vt.with_columns(
                    pl.col("team").is_in(bottom).cast(pl.Int64).alias("low_octane_offense"),
                    pl.lit("Vegas").alias("low_octane_source"),
                ).select("team", "low_octane_offense", "low_octane_source")
                return out.to_pandas()

    # Source B fallback: season prior_season (2025) actual PPG from schedules' own scores.
    reg = schedules.filter(
        (pl.col("season") == prior_season) & (pl.col("game_type") == "REG") & pl.col("home_score").is_not_null()
    )
    home = reg.select(pl.col("home_team").alias("team"), pl.col("home_score").alias("pts"))
    away = reg.select(pl.col("away_team").alias("team"), pl.col("away_score").alias("pts"))
    allg = sh.normalize_team(pl.concat([home, away], how="vertical_relaxed"), "team")
    ppg = allg.group_by("team").agg(pl.col("pts").mean().alias("team_ppg"))
    bottom = set(ppg.sort("team_ppg").head(LOW_OCTANE_N).get_column("team").to_list())
    out = ppg.with_columns(
        pl.col("team").is_in(bottom).cast(pl.Int64).alias("low_octane_offense"),
        pl.lit("est.").alias("low_octane_source"),
    ).select("team", "low_octane_offense", "low_octane_source")
    return out.to_pandas()


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------


_PROJECTION_COLS = [
    "gsis_id", "floor_ppg", "expected_ppg", "ceiling_ppg", "predicted_finish_rank", "cohort_rank",
    "p_elite_raw", "p_startable_raw", "p_useful_raw", "value_gap", "value_gap_typical", "startable_k", "useful_k",
]


def build_veteran_board() -> pd.DataFrame:
    raw = load_raw_frames()
    labels_cfg = load_labels_config()
    thresholds = pj.load_or_build_thresholds()
    rows = []
    for pos in train.POSITIONS:
        spec = train.position_spec(pos)
        if not spec.artifact_path.exists():
            print(f"  {pos}: no bundle at {spec.artifact_path}, skipping")
            continue
        bundle = joblib.load(spec.artifact_path)
        feats = build_veteran_feature_matrix(pos, raw, bundle)
        df = feats.to_pandas()
        df["pos"] = pos.upper()

        scored = score_veterans_batch(bundle, df)
        scored = attach_shap_drivers(bundle, scored)
        scored["rationale"] = [
            build_rationale(pos, scored.iloc[i], scored.iloc[i]["_shap_top_features"]) for i in range(len(scored))
        ]

        # v2.0: breakout_eligible is computed directly here (not only via the quantile
        # projections module) so it is populated even for a position with no quantile
        # bundle yet -- same eligibility gate src.labels.build applies (expectation_pos_rank
        # >= adp_worse_than; a capped player's inflated rank makes them eligible
        # automatically, no special case).
        adp_worse_than = float(labels_cfg["thresholds"][spec.label_position]["adp_worse_than"])
        scored["breakout_eligible"] = scored["expectation_pos_rank"] >= adp_worse_than

        qspec = qmod.quantile_spec(pos)
        if qspec.artifact_path.exists():
            qbundle = joblib.load(qspec.artifact_path)
            qscored = qmod.score_quantiles(qbundle, df)
            proj = pj.compute_projections_for_position(qscored, spec.label_position, thresholds, labels_cfg)
            scored = scored.merge(proj[_PROJECTION_COLS], on="gsis_id", how="left")
        else:
            print(f"  {pos}: no quantile bundle at {qspec.artifact_path}, quantile columns left null")
            for c in _PROJECTION_COLS:
                if c != "gsis_id":
                    scored[c] = np.nan

        rows.append(scored)
        print(f"  {pos}: scored {len(scored)} veterans")
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    # v1.7: per-position base-rate renormalization -- needs every position's rows on
    # hand at once (the scale is sum-over-position), so it runs after the concat, not
    # inside the per-position loop above. See attach_renormalized_probability's docstring.
    mean_breakouts = historical_mean_breakouts_by_position()
    combined = attach_renormalized_probability(combined, mean_breakouts)
    # v2.0: outcome-ladder honesty renormalization (p_elite/p_startable/p_useful),
    # needs every position's rows on hand at once -- same shape as the classifier's own
    # renormalization above. Rows with no quantile bundle (p_*_raw all NaN) get NaN
    # ladder columns out of attach_ladder (NaN.sum() over an all-NaN slice is 0, so that
    # position's scale would divide by zero -- guarded inside attach_ladder's
    # `total > 0` check, which NaN also fails, falling back to scale=1.0 harmlessly).
    combined = pj.attach_ladder(combined, pos_col="pos")
    engine_decision = pj.load_engine_decision()
    combined["primary_engine"] = combined["pos"].map(engine_decision).fillna("classifier")
    combined["projection_sentence"] = [
        build_projection_sentence(combined.iloc[i]) for i in range(len(combined))
    ]
    combined["already_priced_as_starter"] = ~combined["breakout_eligible"].astype("boolean")

    # v2.4: low-octane-offense risk flag -- see compute_low_octane_flags's docstring.
    # Presentation/risk-flag only, never a model input.
    low_octane = compute_low_octane_flags(raw["schedules"])
    combined = combined.merge(low_octane, on="team", how="left")
    combined["low_octane_offense"] = combined["low_octane_offense"].fillna(0).astype(int)
    combined["low_octane_source"] = combined["low_octane_source"].fillna("unknown")

    return combined


def build_projection_sentence(row: pd.Series) -> str | float:
    """"Projects RB1 of 2026 (15.7 pts/gm, range 12.4-20.4); priced RB1 -> +0 spots of

    value; 74% to be a weekly starter, 18% top-5" -- the plain-language echo of every
    quantile-derived column on one player-detail line. Headlines ``cohort_rank`` (where
    he stacks up against every OTHER player scored at his position this year -- the same
    universe the market's own ECR ranks), peer-comparison only -- the "a season like his
    median projection historically finishes ~PosN" clause (``predicted_finish_rank``-
    based) is deliberately dropped from this sentence (peer-rank-only display, user
    preference this session): a median projection structurally can't reach an all-time
    #1's extreme order-statistic threshold, and the user reads that framing as unwanted
    "historically they finish so-and-so" noise vs a straight peer comparison. See
    ``src.inference.projections``' "Two ranks, not one" docstring section for the full
    cohort_rank-vs-predicted_finish_rank distinction (``predicted_finish_rank`` is still
    computed and kept on the board CSV, just no longer surfaced in any user-facing
    sentence/table). ``value_gap`` (displayed here) is the cohort-based gap
    (``expectation_pos_rank - cohort_rank``) -- the display AND sort default everywhere
    on this board as of this session, overriding the audit's own tercile-spread-measured
    preference for ``value_gap_typical`` (see ``write_board``'s note and
    ``outputs/holdout_board_audit.md``'s "Metric decision" subsection for the documented
    tradeoff: both metrics grade at roughly +30pt, and the user's peer-comparison
    preference wins the tie by explicit choice, not because cohort-based grades better).
    ``value_gap_typical``/``predicted_finish_rank`` are still computed and kept on the
    board CSV for continuity/analyst use, not repeated in this sentence. Returns NaN (not
    a string) when the quantile columns are null (no bundle for that position yet),
    matching every other quantile-derived column's null-when-absent convention.
    """
    if pd.isna(row.get("expected_ppg")):
        return np.nan
    pos = row["pos"]
    gap = row.get("value_gap")
    gap_str = f"+{gap:.0f} spots of value" if pd.notna(gap) and gap > 0 else (f"{gap:.0f} spots below market" if pd.notna(gap) else "n/a")
    market_rank = row.get("expectation_pos_rank")
    market_str = f"priced {pos}{int(market_rank)}" if pd.notna(market_rank) else "unpriced"
    cohort = row.get("cohort_rank")
    cohort_str = f"{pos}{int(cohort)} of {row.get('season', SEASON):.0f}" if pd.notna(cohort) else "unranked"
    # Peer-rank-only display (user preference, this session): the "a season like his
    # median projection historically finishes ~PosN" clause (predicted_finish_rank-based)
    # is dropped entirely -- the user wants players compared to this year's peers, not to
    # a historical typical-season mapping. predicted_finish_rank/value_gap_typical still
    # exist on the board CSV for analysts; see write_board's veteran_cols note and
    # README/outputs/holdout_board_audit.md's "Metric decision" subsection for the
    # documented tradeoff (value_gap_typical still measures slightly better as a sorter,
    # ~+33.2pt vs ~+30.0pt tercile spread -- overridden here by explicit user preference).
    return (
        f"Projects {cohort_str} ({row['expected_ppg']:.1f} pts/gm, range {row['floor_ppg']:.1f}-{row['ceiling_ppg']:.1f})"
        f"; {market_str} -> {gap_str}; "
        f"{row['p_startable']*100:.0f}% to be a weekly starter, {row['p_elite']*100:.0f}% top-5"
    )


def build_rookie_board() -> pd.DataFrame:
    if not rh.ARTIFACT_PATH.exists():
        print(f"  rookie heuristic bundle not found at {rh.ARTIFACT_PATH}; run `python -m src.models.rookie_heuristic` first")
        return pd.DataFrame()

    bundle = rh.load_bundle()
    raw = load_raw_frames()
    reg = sh.reg_with_points(raw["player_stats"], load_scoring_config())
    season_team = sh.season_roster_team(raw["rosters"], raw["rosters_weekly"])
    vacated_2026 = sh.vacated_shares(reg, season_team).filter(pl.col("season") == SEASON)

    feats = rh.rookie_2026_frame(
        draft_picks=raw["draft_picks"], rosters=raw["rosters"], combine=raw["combine"], vacated_2026=vacated_2026
    )
    feats.write_parquet(ROOKIE_FEATURES_PATH)

    df = feats.to_pandas()
    df["pos"] = df["position"]
    df["probability_heuristic"] = rh.score_rookies(bundle, df)
    print(f"  rookies: scored {len(df)} ({df['pos'].value_counts().to_dict()})")
    return df


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


# Most recent completed season -- the "broke out last season" transparency
# badge (v2.1 addendum) reads this year's breakout==1 population off
# labels.parquet so a player like Daniel Jones (2025 breakout, still
# 2026-eligible) shows up on the current board with a visible explanation
# instead of silently looking like a "new" breakout pick.
LAST_SEASON_BREAKOUT_YEAR = 2025


def attach_broke_out_last_season(df: pd.DataFrame, gsis_col: str = "gsis_id") -> pd.DataFrame:
    """broke_out_last_season: True if `gsis_col` had breakout==1 in labels.parquet's

    season==``LAST_SEASON_BREAKOUT_YEAR`` row. Presentation-only (no model input) -- see
    the module-level comment above ``LAST_SEASON_BREAKOUT_YEAR``. Works unchanged for the
    rookie board too: a 2026 rookie's gsis_id structurally cannot match a 2025 non-rookie
    breakout row, so it always comes out False there, no special-casing needed.
    """
    out = df.copy()
    if gsis_col not in out.columns or out.empty:
        out["broke_out_last_season"] = False
        return out
    labels = pl.read_parquet(train.LABELS_PATH)
    broke_out_ids = set(
        labels.filter((pl.col("season") == LAST_SEASON_BREAKOUT_YEAR) & (pl.col("breakout") == 1))
        .get_column("gsis_id")
        .to_list()
    )
    out["broke_out_last_season"] = out[gsis_col].isin(broke_out_ids)
    return out


def write_board(veteran_board: pd.DataFrame, rookie_board: pd.DataFrame) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    veteran_cols = [
        "player_name", "pos", "age", "team", "probability", "probability_calibrated_raw", "raw_score",
        "tier", "base_rate_multiple", "base_rate", "probability_saturated",
        "expected_rank_delta", "consensus_pos_rank", "adp", "sleeper_adp_pos_rank", "adp_gap",
        "implied_pts", "edge", "availability", "shap_top3", "rationale",
        # v2.0 quantile-derived columns (Deliverable 4) -- null for a position with no
        # quantile bundle yet, same convention as every other optional overlay column above.
        # cohort_rank (this year's field) and predicted_finish_rank (a typical season with
        # this q50) are two different questions -- see src.inference.projections' "Two
        # ranks, not one" docstring section. Peer-rank-only display (this session):
        # predicted_finish_rank/value_gap_typical are ANALYST-ONLY columns as of this
        # session -- every user-facing table/sentence (write_board's .md output, the
        # dashboard) shows cohort_rank/value_gap exclusively; both legacy columns stay
        # here on the CSV for continuity/analysts who want the audit's own originally-
        # validated (predicted_finish_rank-based) sorter -- see the "Metric decision"
        # subsection of outputs/holdout_board_audit.md and this function's own v2.3/
        # peer-rank-only notes below for the measured tradeoff.
        "floor_ppg", "expected_ppg", "ceiling_ppg", "predicted_finish_rank", "cohort_rank",
        "p_startable", "p_elite", "p_useful", "value_gap", "value_gap_typical", "breakout_eligible",
        "primary_engine", "projection_sentence", "already_priced_as_starter",
        "seg_elite", "seg_starter", "seg_useful", "seg_bust",
        "broke_out_last_season",
        # v2.4: low-octane-offense risk flag (measured 2020-2025 pooled: 6.7% startable-
        # season rate on bottom-5 preseason-implied-total offenses vs 33.3% on top-5) --
        # see compute_low_octane_flags's docstring. Presentation/risk-flag only, never a
        # model input (the models already rejected Vegas team features twice).
        "low_octane_offense", "low_octane_source",
    ]
    v = attach_broke_out_last_season(veteran_board.copy())
    if not v.empty:
        # raw_score (pre-calibration blend score) is the tiebreaker for ties -- the
        # per-position renormalization scale is a single positive constant, so it
        # cannot change within-position ordering; this tiebreak is belt-and-suspenders.
        v = v.sort_values(["pos", "probability", "raw_score"], ascending=[True, False, False])
    for c in veteran_cols:
        if c not in v.columns:
            v[c] = np.nan
    v_out = v[veteran_cols].rename(columns={"consensus_pos_rank": "consensus_ecr_pos_rank"})

    rookie_cols = [
        "player_name", "pos", "team", "draft_round", "draft_pick", "probability_heuristic",
        "availability", "note", "broke_out_last_season",
    ]
    r = attach_broke_out_last_season(rookie_board.copy())
    if not r.empty:
        r = r.sort_values(["pos", "probability_heuristic"], ascending=[True, False])
    for c in rookie_cols:
        if c not in r.columns:
            r[c] = np.nan
    r_out = r[rookie_cols] if not r.empty else pd.DataFrame(columns=rookie_cols)

    combined = pd.concat(
        [v_out.assign(section="veteran"), r_out.assign(section="rookie (heuristic)")], ignore_index=True, sort=False
    )
    combined.to_csv(BOARD_CSV_PATH, index=False)

    lines = [f"# BreakoutLab -- {SEASON} Breakout Board", "", "Sorted by probability within position. Rookies are scored"]
    lines.append(
        "by a separate, lower-confidence heuristic (`src.models.rookie_heuristic`) and are never on the same"
    )
    lines.append("probability scale as the veteran model — see the README's Known Limitations section.")
    lines.append("")
    lines.append(
        "> **Note (v1.7):** veteran `probability` is renormalized per position so it sums to that "
        "position's historical mean per-season breakout count (`probability_calibrated_raw * renorm_scale`, "
        f"capped at {PROB_DISPLAY_CAP:.2f}) — a calibrated probability is only ever calibrated against its own "
        "validation pool, nothing forces a position's probabilities to sum to how many breakouts that position "
        "actually produces in a season, and pre-renormalization they didn't (see README). `probability_calibrated_raw` "
        "(pre-rescale, Platt-calibrated) and `raw_score` (pre-calibration) are both kept alongside it; the rescale is "
        "a single positive per-position constant, so it cannot change within-position ranking. `tier` is this "
        "probability's multiple of the position's base rate (`base_rate` = historical mean breakouts / players scored "
        "at that position this year) — Elite target (8x+), Strong swing (4-8x), Worth a flier (2-4x), Long shot (under 2x)."
    )
    lines.append("")
    lines.append(
        "> **v2.0 (two lenses):** every veteran also carries a quantile-regression projection "
        "(`floor_ppg`/`expected_ppg`/`ceiling_ppg` = q10/q50/q90 of his predicted ppr-ppg distribution) and the "
        "outcome ladder `p_elite`/`p_startable`/`p_useful` -- see README's v2.0 section. **Breakout Hunt** below "
        "ranks only `breakout_eligible` players (the same eligibility gate `src.labels.build` uses to define a "
        "breakout at all) by whichever engine Deliverable 3's comparison gate picked as primary for that position "
        "(`primary_engine` column -- classifier `probability` or quantile `p_startable`); **Full Projections** "
        "ranks EVERY scored veteran by `expected_ppg`, eligible or not. Neither lens replaces the other; both read "
        "off the same underlying columns."
    )
    lines.append("")
    lines.append(
        "> **Peer-rank-only display (this session, user preference):** `cohort_rank` -- where a player's median "
        "projection stacks up against every OTHER player scored at his position this year, the same field the "
        "market's own ECR ranks -- is now the ONLY rank shown anywhere on this board, behind the tables' "
        "**Projects (2026)** column (see `src.inference.projections`' \"Two ranks, not one\" docstring section for "
        "the full cohort_rank-vs-predicted_finish_rank distinction). The historical-typical-season mapping "
        "(`predicted_finish_rank`, formerly shown as **Typical finish**) is dropped from every table/sentence here "
        "-- a user preference against the \"historically they finish so-and-so\" framing, not a correctness fix; "
        "it is still computed and kept on the board CSV for analysts. **The Value gap column is now `value_gap`** "
        "(cohort-based, `= consensus ECR rank - cohort_rank`) -- the display AND the sort default everywhere, also "
        "by explicit user preference, overriding the holdout audit's own tercile-spread-measured lean toward "
        "`value_gap_typical` (`outputs/holdout_board_audit.md`'s \"Metric decision\" subsection: both metrics "
        "grade at roughly +30pt spread, `value_gap_typical` a few points ahead -- close enough that the user's "
        "peer-comparison preference is the deciding factor, not a claim that cohort-based grades better). "
        "`value_gap_typical` remains on the board CSV alongside `predicted_finish_rank` for the same "
        "analyst-continuity reason; the threshold-curve/outcome-ladder machinery underneath both ranks is "
        "unchanged (the ladder derivation still needs `predicted_finish_rank` internally)."
    )
    lines.append("")
    lines.append("## Breakout Hunt")
    lines.append("")
    lines.append("Eligible players only (already-priced starters are excluded from this lens entirely).")
    lines.append("")
    for pos in ("QB", "RB", "WR", "TE"):
        sub = v_out[(v_out["pos"] == pos) & (v_out["breakout_eligible"] == True)]  # noqa: E712
        if sub.empty:
            continue
        engine = sub["primary_engine"].iloc[0] if pd.notna(sub["primary_engine"].iloc[0]) else "classifier"
        sort_col = "p_startable" if engine == "quantile" else "probability"
        sub = sub.sort_values(sort_col, ascending=False, na_position="last")
        lines.append(f"### {pos} — primary engine: **{engine}**")
        lines.append("")
        # Value gap = value_gap (cohort-based) -- the display/sort default everywhere on
        # this board as of this session; see the peer-rank-only note above.
        headers = ["Player", "Team", "Probability", "P(startable)", "P(elite)", "P(useful)", "Tier", "Projects (2026)", "Consensus (ECR)", "Value gap", "Availability"]
        rows = [
            [
                row["player_name"], row["team"], row["probability"], row["p_startable"], row["p_elite"], row["p_useful"],
                row["tier"], row["cohort_rank"], row["consensus_ecr_pos_rank"], row["value_gap"], row["availability"],
            ]
            for _, row in sub.iterrows()
        ]
        lines.append(_markdown_table(headers, rows))
        lines.append("")

    lines.append("## Full Projections")
    lines.append("")
    lines.append("Every scored veteran, eligible or not, ranked by expected ppg (q50).")
    lines.append("")
    for pos in ("QB", "RB", "WR", "TE"):
        sub = v_out[(v_out["pos"] == pos) & v_out["expected_ppg"].notna()]
        if sub.empty:
            continue
        sub = sub.sort_values("expected_ppg", ascending=False)
        lines.append(f"### {pos}")
        lines.append("")
        # Value gap = value_gap (cohort-based, peer-rank-only display -- see the note
        # above); no more "Typical finish" / "Value gap (cohort)" duplicate columns.
        headers = [
            "Player", "Team", "Floor", "Expected", "Ceiling", "Projects (2026)",
            "Consensus (ECR)", "Value gap", "P(startable)", "P(elite)", "P(useful)", "Badge",
        ]
        rows = [
            [
                row["player_name"], row["team"], row["floor_ppg"], row["expected_ppg"], row["ceiling_ppg"],
                row["cohort_rank"], row["consensus_ecr_pos_rank"], row["value_gap"],
                row["p_startable"], row["p_elite"], row["p_useful"],
                "already priced as a starter" if row["already_priced_as_starter"] else "",
            ]
            for _, row in sub.iterrows()
        ]
        lines.append(_markdown_table(headers, rows))
        lines.append("")

    lines.append("## Veterans (full detail)")
    lines.append("")
    for pos in ("QB", "RB", "WR", "TE"):
        sub = v_out[v_out["pos"] == pos]
        if sub.empty:
            continue
        lines.append(f"### {pos}")
        lines.append("")
        headers = [
            "Player", "Age", "Team", "Probability", "Tier", "Base-rate x", "Raw calibrated", "Raw score",
            "Exp. rank Δ", "Consensus (ECR)", "Sleeper ADP", "ADP gap", "Vegas implied", "Edge", "Availability",
            "Top drivers", "Rationale", "Floor", "Expected", "Ceiling", "Projects (2026)",
            "P(startable)", "P(elite)", "P(useful)", "Value gap", "Eligible", "Primary engine", "Low-octane offense", "Projection sentence",
        ]
        rows = [
            [
                row["player_name"], row["age"], row["team"], row["probability"], row["tier"],
                row["base_rate_multiple"], row["probability_calibrated_raw"], row["raw_score"],
                row["expected_rank_delta"], row["consensus_ecr_pos_rank"], row["sleeper_adp_pos_rank"],
                row["adp_gap"], row["implied_pts"], row["edge"], row["availability"], row["shap_top3"], row["rationale"],
                row["floor_ppg"], row["expected_ppg"], row["ceiling_ppg"], row["cohort_rank"],
                row["p_startable"], row["p_elite"], row["p_useful"], row["value_gap"],
                row["breakout_eligible"], row["primary_engine"],
                f"Low-octane offense ({row['low_octane_source']})" if row.get("low_octane_offense") else "",
                row["projection_sentence"],
            ]
            for _, row in sub.iterrows()
        ]
        lines.append(_markdown_table(headers, rows))
        lines.append("")

    lines.append("## Rookies (heuristic — lower confidence, separate model, not comparable to veteran probabilities)")
    lines.append("")
    for pos in ("QB", "RB", "WR", "TE"):
        sub = r_out[r_out["pos"] == pos]
        if sub.empty:
            continue
        lines.append(f"### {pos}")
        lines.append("")
        headers = ["Player", "Team", "Round", "Pick", "Heuristic probability", "Availability"]
        rows = [
            [row["player_name"], row["team"], row["draft_round"], row["draft_pick"], row["probability_heuristic"], row["availability"]]
            for _, row in sub.iterrows()
        ]
        lines.append(_markdown_table(headers, rows))
        lines.append("")

    BOARD_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print(f"2026 breakout board | veteran scoring")
    veteran_board = build_veteran_board()

    if not veteran_board.empty:
        n_saturated = int(veteran_board["probability_saturated"].sum())
        print(
            f"v1.7 calibration check: {n_saturated} veteran row(s) hit the pre-clamp 0.0/1.0 "
            f"boundary (out of {len(veteran_board)}) -- expected 0 now that scoring uses Platt "
            "(src.models.train.fit_platt) on the seed-ensemble OOF score. PROB_DISPLAY_LO/HI "
            "clamp stays on as a no-op safety net regardless."
        )
        for pos in sorted(veteran_board["pos"].unique()):
            sub = veteran_board[veteran_board["pos"] == pos]
            print(
                f"  {pos}: renorm_scale={sub['renorm_scale'].iloc[0]:.3f}, "
                f"sum(displayed probability)={sub['probability'].sum():.2f}, "
                f"base_rate={sub['base_rate'].iloc[0]:.4f} (n={len(sub)})"
            )

    if not veteran_board.empty:
        veteran_board = veteran_board.rename(columns={"expectation_pos_rank": "consensus_pos_rank"})
        crosswalk = build_id_map()
        availability = load_availability()
        veteran_board = attach_availability(veteran_board, availability)
        veteran_board = attach_overlay(veteran_board, crosswalk)

    print("2026 breakout board | rookie heuristic scoring")
    rookie_board = build_rookie_board()
    if not rookie_board.empty:
        availability = load_availability()
        rookie_board = attach_availability(rookie_board, availability)

    write_board(veteran_board, rookie_board)
    print(f"wrote {BOARD_CSV_PATH}")
    print(f"wrote {BOARD_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
