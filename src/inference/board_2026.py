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

Displayed-probability clamp (presentation only — no refit, no retune)
------------------------------------------------------------------------
Isotonic calibration (every position's frozen ``calibration_method`` as of
this build) fits a step function over pooled OOF validation predictions;
at these sample sizes its terminal buckets (the highest/lowest raw-score
bucket) can be entirely one class, which isotonic maps to an exact 0.0 or
1.0 — not evidence of certainty, just a small-bucket artifact. Verified
directly on this build: five 2026 QBs and three 2026 TEs land on an exact
1.000 calibrated probability, which (a) reads as false certainty and (b)
collapses their board ordering to ties. Two presentation-only fixes,
neither of which touches the classifier, the blend, or the calibrator
itself:

- ``probability`` (the board's headline number) is clamped to
  ``[PROB_DISPLAY_LO, PROB_DISPLAY_HI]`` = ``[0.01, 0.95]`` for display.
  ``probability_saturated`` (bool) flags every row whose *pre-clamp*
  calibrated value was exactly 0.0 or 1.0, so a reader can tell "hit the
  clamp boundary" apart from "genuinely scored near there."
- ``raw_score`` (the pre-calibration blended classifier score) is kept as
  a secondary sort key — ``write_board`` sorts by
  ``(probability desc, raw_score desc)`` within position, so saturated
  ties still get a real, reproducible ordering instead of an arbitrary one.

A principled fix (e.g. Laplace-smoothed isotonic buckets, or reporting a
confidence interval instead of a point estimate) needs persisted
per-fold OOF predictions and is out of scope for this pass — noted in the
README as a v1.5 follow-up.
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
from src.ingest.id_map import build_id_map, match_to_gsis, normalize_name
from src.labels.build import PLAYER_STATS_PATH, add_expectation, load_scoring_config
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
    },
    "rb": {
        "build_fn": feat_rb.build_features_rb,
        "ngs_kwarg": "ngs_rushing",
        "ngs_path": feat_rb.NGS_RUSHING_PATH,
        "needs_snap": True,
    },
    "te": {
        "build_fn": feat_te.build_features_te,
        "ngs_kwarg": "ngs_receiving",
        "ngs_path": feat_te.NGS_RECEIVING_PATH,
        "needs_snap": True,
    },
    "qb": {
        "build_fn": feat_qb.build_features_qb,
        "ngs_kwarg": "ngs_passing",
        "ngs_path": feat_qb.NGS_PASSING_PATH,
        "needs_snap": False,
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
    """Attaches `probability` (calibrated, clamped for display), `raw_score` (the

    pre-calibration blended classifier score), `probability_saturated` (bool: the
    pre-clamp calibrated value was exactly 0.0 or 1.0 -- an isotonic terminal-bucket
    artifact at this sample size, not model certainty -- see module docstring's
    "Displayed-probability clamp" section), and `expected_rank_delta` to every row.
    """
    tree_cols = bundle["tree_feature_cols"]
    weights = bundle["blend_weights"]
    preds = {}
    if weights.get("lgbm", 0) > 0:
        preds["lgbm"] = bundle["lgbm"].predict_proba(df[tree_cols])[:, 1]
    if weights.get("xgb", 0) > 0:
        preds["xgb"] = bundle["xgb"].predict_proba(df[tree_cols])[:, 1]
    if weights.get("logistic", 0) > 0:
        model, imputer, scaler = bundle["logistic"]
        logit_cols = bundle["logistic_feature_cols"]
        X = scaler.transform(imputer.transform(df[logit_cols]))
        preds["logistic"] = model.predict_proba(X)[:, 1]
    nonzero_weights = {k: w for k, w in weights.items() if k in preds}
    blended = train.apply_blend(nonzero_weights, **preds)
    calibrated_raw = train.apply_calibration(bundle["calibration_method"], bundle["calibrator"], blended)

    lgbm_reg_pred = bundle["lgbm_reg"].predict(df[tree_cols])
    xgb_reg_pred = bundle["xgb_reg"].predict(df[tree_cols])

    out = df.copy()
    out["raw_score"] = blended
    out["probability_saturated"] = (np.isclose(calibrated_raw, 0.0) | np.isclose(calibrated_raw, 1.0)).astype(int)
    out["probability"] = np.clip(calibrated_raw, PROB_DISPLAY_LO, PROB_DISPLAY_HI)
    out["expected_rank_delta"] = (lgbm_reg_pred + xgb_reg_pred) / 2.0
    return out


def attach_shap_drivers(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    tree_cols = bundle["tree_feature_cols"]
    sv = shp.blend_weighted_shap(bundle, df[tree_cols])
    compact, top_features = [], []
    for i in range(len(df)):
        order = np.argsort(-np.abs(sv[i]))[:TOP_DRIVERS_ON_BOARD]
        parts = [f"{tree_cols[j]}:{'+' if sv[i][j] >= 0 else '-'}" for j in order]
        compact.append(", ".join(parts))
        top_features.append([(tree_cols[j], float(sv[i][j])) for j in order])
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
}


def _humanize_feature(name: str) -> str:
    base = re.sub(r"_(n1|yoy_delta)$", "", name)
    label = _FEATURE_LABELS.get(base, base.replace("_", " "))
    if name.endswith("_yoy_delta"):
        label = f"{label} trend"
    return label


def build_rationale(pos: str, row: pd.Series, top_features: list[tuple[str, float]]) -> str:
    """Template sentence from the top-3 SHAP drivers, e.g. "Year-2 WR: boosted by

    efficiency residual on volume; boosted by vacated targets ahead; held back by age."
    A plain-language echo of ``shap_top3``, not a new signal.
    """
    year = row.get("year_in_league")
    year_str = f"Year-{int(year) + 1}" if pd.notna(year) else "Veteran"
    phrases = []
    for feature, value in top_features:
        verb = "boosted by" if value >= 0 else "held back by"
        phrases.append(f"{verb} {_humanize_feature(feature)}")
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
# Full pipeline
# --------------------------------------------------------------------------


def build_veteran_board() -> pd.DataFrame:
    raw = load_raw_frames()
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
        rows.append(scored)
        print(f"  {pos}: scored {len(scored)} veterans")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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


def write_board(veteran_board: pd.DataFrame, rookie_board: pd.DataFrame) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    veteran_cols = [
        "player_name", "pos", "age", "team", "probability", "raw_score", "probability_saturated",
        "expected_rank_delta", "consensus_pos_rank", "adp", "sleeper_adp_pos_rank", "adp_gap",
        "implied_pts", "edge", "availability", "shap_top3", "rationale",
    ]
    v = veteran_board.copy()
    if not v.empty:
        # raw_score (pre-calibration blend score) is the tiebreaker for saturated
        # probability ties (see module docstring's "Displayed-probability clamp").
        v = v.sort_values(["pos", "probability", "raw_score"], ascending=[True, False, False])
    for c in veteran_cols:
        if c not in v.columns:
            v[c] = np.nan
    v_out = v[veteran_cols].rename(columns={"consensus_pos_rank": "consensus_ecr_pos_rank"})

    rookie_cols = ["player_name", "pos", "team", "draft_round", "draft_pick", "probability_heuristic", "availability", "note"]
    r = rookie_board.copy()
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
        f"> **Note:** veteran `probability` is clamped to [{PROB_DISPLAY_LO:.2f}, {PROB_DISPLAY_HI:.2f}] for "
        "display — isotonic calibration saturates to an exact 0/1 in terminal OOF buckets at this sample size, "
        "which is a small-bucket artifact, not model certainty. `probability_saturated=1` flags a row that hit "
        "that boundary pre-clamp; `raw_score` (the pre-calibration blend score) breaks the resulting ties and is "
        "the real ordering signal among saturated rows — see `src/inference/board_2026.py`'s module docstring."
    )
    lines.append("")
    lines.append("## Veterans")
    lines.append("")
    for pos in ("QB", "RB", "WR", "TE"):
        sub = v_out[v_out["pos"] == pos]
        if sub.empty:
            continue
        lines.append(f"### {pos}")
        lines.append("")
        headers = ["Player", "Age", "Team", "Probability", "Raw score", "Saturated", "Exp. rank Δ", "Consensus (ECR)", "Sleeper ADP", "ADP gap", "Vegas implied", "Edge", "Availability", "Top drivers", "Rationale"]
        rows = [
            [
                row["player_name"], row["age"], row["team"], row["probability"], row["raw_score"],
                int(row["probability_saturated"]) if pd.notna(row["probability_saturated"]) else "",
                row["expected_rank_delta"], row["consensus_ecr_pos_rank"], row["sleeper_adp_pos_rank"],
                row["adp_gap"], row["implied_pts"], row["edge"], row["availability"], row["shap_top3"], row["rationale"],
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

    BOARD_MD_PATH.write_text("\n".join(lines))


def main() -> int:
    print(f"2026 breakout board | veteran scoring")
    veteran_board = build_veteran_board()

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
