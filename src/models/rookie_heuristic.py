"""Rookie breakout heuristic (Phase 6.2).

The main Phase 4 model is trained exclusively on non-rookie seasons
(``in_training_pool`` excludes every ``is_rookie`` row, per
``src.labels.build``'s design and every ``src.features.{wr,rb,te,qb}``
population filter) because a rookie has no N-1 season to build the bulk of
those features from. Scoring 2026's rookie class therefore needs a
separate, much simpler model — this one — trained on a feature set a
rookie genuinely has before Week 1 of his rookie year: **draft capital**
(round, log(pick)), **combine athleticism** (40 time, and a standard
speed-score derived from it — both nullable, not every draftee tests, and
combine invites for late/UDFA prospects are sparse), and **landing-team
context** (the team's vacated target/carry share and how much of that
team's own draft capital that season went to the same position, i.e. his
immediate positional competition). A single shallow L2 logistic
regression — deliberately not a tuned LGBM/XGB/blend/calibration
pipeline: the brief calls this a heuristic, and ~450-550 historical
rookie-seasons across four positions is too thin a sample to justify the
Phase-4 machinery. See ``configs/model_wr.yaml`` and siblings for what
"the real thing" looks like when the sample supports it.

Training population: every ``is_rookie=1`` QB/RB/WR/TE row in
``labels.parquet`` (2014-2025), regardless of ``in_training_pool`` (that
flag's rookie exclusion is specifically what carves out this table's
population — reusing it here, not around it). Validation is a plain
time-based split (train seasons <= ``TRAIN_SEASONS_END``, "check" ordering
on the seasons after) — a sanity check that the model's ranking holds up
out-of-time, not a tuned expanding-window CV like Phase 4's; no
hyperparameter search runs against the check split.

2026 scoring: 2026 rookies come from ``draft_picks`` (season == 2026),
restricted to QB/RB/WR/TE. One 2026-specific wrinkle, verified directly
against the cached data before writing this: **``draft_picks``' own
``gsis_id`` for the 2026 class is a temporary, non-nflverse id** (upstream
hasn't back-filled the real crosswalk yet — 0/257 rows start with the
standard "00-" prefix nflverse uses everywhere else, vs. 100% for every
prior draft season). ``resolve_2026_rookie_ids`` recovers the real roster
gsis_id two rungs deep (pfr_id exact join, then normalized-name + position
for the pfr_id misses — 72/80 and 8/80 of the 2026 skill-position class
respectively, 100% combined at ingest time) so the board can actually join
these rookies to a real player identity; any pick that still fails both
rungs keeps its own temp id (visible, not dropped) but can't carry a real
combine/vacated-share join.

Output is explicitly lower-confidence: every column this module produces
downstream is prefixed/labeled "heuristic" (see ``src.inference.board_2026``),
never merged into the calibrated veteran probability scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features import shared as sh
from src.ingest.id_map import normalize_name
from src.labels.build import PLAYER_STATS_PATH, load_scoring_config
from src.models import metrics as mt

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = REPO_ROOT / "data" / "models"

LABELS_PATH = PROCESSED_DIR / "labels.parquet"
DRAFT_PICKS_PATH = sh.PATHS["draft_picks"]
COMBINE_PATH = RAW_DIR / "combine.parquet"
ARTIFACT_PATH = MODELS_DIR / "rookie_heuristic_bundle.joblib"
REPORT_PATH = OUTPUTS_DIR / "rookie_heuristic_report.md"

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
ROOKIE_SEASON_START = 2014
ROOKIE_SEASON_END = 2025
DRAFT_2026_SEASON = 2026

# Time-based sanity split, not a tuned CV boundary (see module docstring).
TRAIN_SEASONS_END = 2023

FEATURE_COLUMNS = [
    "draft_round",
    "log_draft_pick",
    "forty",
    "speed_score",
    "vacated_target_share",
    "vacated_carry_share",
    "competition_draft_capital",
]

SEED = 42
LOGISTIC_C = 1.0


# --------------------------------------------------------------------------
# Shared feature attachment (historical training + 2026 scoring alike)
# --------------------------------------------------------------------------


def _attach_rookie_features(
    base: pl.DataFrame, draft_picks: pl.DataFrame, combine: pl.DataFrame, vacated: pl.DataFrame
) -> pl.DataFrame:
    """``base``: season, gsis_id, player_name, position, team, draft_round, draft_pick,

    pfr_player_id (nullable). Attaches log_draft_pick, forty, speed_score,
    vacated_target_share, vacated_carry_share, competition_draft_capital —
    every one nullable by construction (undrafted, no combine invite, or
    unknown landing team).

    ``competition_draft_capital`` here means the rookie's *own* positional
    draft-class competition: the same (season, team, position) sum
    ``src.features.shared.competition_draft_capital`` computes for the
    veteran models, minus the rookie's own pick's capital (he is not his
    own competition).
    """
    out = base.with_columns(pl.col("draft_pick").log().alias("log_draft_pick"))
    out = out.with_columns((pl.lit(sh.UNDRAFTED_PICK) - pl.col("draft_pick")).clip(lower_bound=0).alias("_own_capital"))

    cb = combine.select(pl.col("pfr_id"), pl.col("forty").cast(pl.Float64), pl.col("wt").cast(pl.Float64))
    out = out.join(cb, left_on="pfr_player_id", right_on="pfr_id", how="left")
    out = out.with_columns(
        pl.when(pl.col("forty") > 0).then((pl.col("wt") * 200) / (pl.col("forty") ** 4)).otherwise(None).alias("speed_score")
    )

    out = out.join(vacated, on=["season", "team"], how="left")

    dp_cap = draft_picks.filter(pl.col("pick").is_not_null())
    dp_cap = sh.normalize_team(dp_cap, "team")
    dp_cap = dp_cap.with_columns((pl.lit(sh.UNDRAFTED_PICK) - pl.col("pick")).clip(lower_bound=0).alias("capital"))
    team_pos_capital = dp_cap.group_by(["season", "team", "position"]).agg(pl.col("capital").sum().alias("_team_pos_capital"))
    out = out.join(team_pos_capital, on=["season", "team", "position"], how="left")
    out = out.with_columns(
        (pl.col("_team_pos_capital").fill_null(0.0) - pl.col("_own_capital").fill_null(0.0)).alias("competition_draft_capital")
    )
    return out.drop("_own_capital", "_team_pos_capital", "wt")


# --------------------------------------------------------------------------
# Historical training frame (2014-2025)
# --------------------------------------------------------------------------


def historical_rookie_frame(
    *,
    labels: pl.DataFrame,
    draft_picks: pl.DataFrame,
    combine: pl.DataFrame,
    player_stats: pl.DataFrame,
    rosters: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
    scoring_profile: dict | None = None,
) -> pl.DataFrame:
    """season, gsis_id -> every FEATURE_COLUMNS entry + breakout, for every historical rookie season."""
    if scoring_profile is None:
        scoring_profile = load_scoring_config()

    reg = sh.reg_with_points(player_stats, scoring_profile)
    season_team = sh.season_roster_team(rosters, rosters_weekly)
    vacated = sh.vacated_shares(reg, season_team)

    rookies = labels.filter(pl.col("is_rookie") & pl.col("position").is_in(SKILL_POSITIONS)).select(
        "season", "gsis_id", "player_name", "position", "breakout", "finish_rank_delta"
    )
    rookies = rookies.join(season_team.select("season", "gsis_id", "team"), on=["season", "gsis_id"], how="left")

    dp = draft_picks.filter(pl.col("gsis_id").is_not_null() & pl.col("pick").is_not_null())
    dp = sh.normalize_team(dp, "team").select(
        "season", "gsis_id", pl.col("round").alias("draft_round"), pl.col("pick").alias("draft_pick"), "pfr_player_id"
    )
    out = rookies.join(dp, on=["season", "gsis_id"], how="left")
    out = out.with_columns(
        pl.col("draft_round").fill_null(sh.UNDRAFTED_ROUND),
        pl.col("draft_pick").fill_null(sh.UNDRAFTED_PICK),
    )

    out = _attach_rookie_features(out, draft_picks, combine, vacated)
    return out.select(
        "season", "gsis_id", "player_name", "position", "breakout", "finish_rank_delta", *FEATURE_COLUMNS
    ).sort(["season", "gsis_id"])


def load_historical_rookie_frame(
    *,
    labels_path: Path = LABELS_PATH,
    draft_picks_path: Path = DRAFT_PICKS_PATH,
    combine_path: Path = COMBINE_PATH,
    player_stats_path: Path = PLAYER_STATS_PATH,
    rosters_path: Path = sh.PATHS["rosters"],
    rosters_weekly_path: Path = sh.PATHS["rosters_weekly"],
) -> pl.DataFrame:
    return historical_rookie_frame(
        labels=pl.read_parquet(labels_path),
        draft_picks=pl.read_parquet(draft_picks_path),
        combine=pl.read_parquet(combine_path),
        player_stats=pl.read_parquet(player_stats_path),
        rosters=pl.read_parquet(rosters_path),
        rosters_weekly=pl.read_parquet(rosters_weekly_path),
    )


# --------------------------------------------------------------------------
# 2026 id resolution + rookie population
# --------------------------------------------------------------------------


def resolve_2026_rookie_ids(draft_picks_2026: pl.DataFrame, rosters_2026: pl.DataFrame) -> pl.DataFrame:
    """Resolve 2026 draft_picks rows to a real roster gsis_id — see module docstring.

    Rung 1: pfr_player_id (draft_picks) == pfr_id (rosters), exact.
    Rung 2 (pfr_id misses only): normalized name + position, exact.
    A pick that fails both rungs keeps its own (temporary, non-standard)
    draft_picks gsis_id, visibly, rather than being dropped.
    """
    dp = draft_picks_2026.rename({"gsis_id": "draft_gsis_id"})
    rr = rosters_2026.select(
        "pfr_id",
        pl.col("gsis_id").alias("roster_gsis_id"),
        pl.col("position").alias("roster_pos"),
    )
    joined = dp.join(rr, left_on="pfr_player_id", right_on="pfr_id", how="left")

    matched = joined.filter(pl.col("roster_gsis_id").is_not_null())
    unmatched = joined.filter(pl.col("roster_gsis_id").is_null())

    r_names = rosters_2026.with_columns(pl.col("full_name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("_norm_name"))
    r_names = r_names.select("_norm_name", pl.col("position").alias("_rpos"), pl.col("gsis_id").alias("_name_gsis"))

    u2 = unmatched.drop("roster_gsis_id", "roster_pos").with_columns(
        pl.col("pfr_player_name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("_norm_name")
    )
    u2 = u2.join(r_names, on="_norm_name", how="left")
    u2 = u2.with_columns(
        pl.when(pl.col("position") == pl.col("_rpos")).then(pl.col("_name_gsis")).otherwise(None).alias("roster_gsis_id")
    ).drop("_norm_name", "_rpos", "_name_gsis")

    out = pl.concat([matched.drop("roster_pos"), u2], how="diagonal_relaxed")
    out = out.with_columns(pl.coalesce(["roster_gsis_id", "draft_gsis_id"]).alias("gsis_id"))
    return out.drop("draft_gsis_id", "roster_gsis_id")


def rookie_2026_frame(
    *, draft_picks: pl.DataFrame, rosters: pl.DataFrame, combine: pl.DataFrame, vacated_2026: pl.DataFrame
) -> pl.DataFrame:
    """season=2026, gsis_id -> every FEATURE_COLUMNS entry, for the 2026 skill-position draft class."""
    dp_2026 = draft_picks.filter((pl.col("season") == DRAFT_2026_SEASON) & pl.col("position").is_in(SKILL_POSITIONS))
    rosters_2026 = rosters.filter(pl.col("season") == DRAFT_2026_SEASON)
    resolved = resolve_2026_rookie_ids(dp_2026, rosters_2026)

    resolved = sh.normalize_team(resolved, "team")
    base = resolved.select(
        "season",
        "gsis_id",
        pl.col("pfr_player_name").alias("player_name"),
        "position",
        "team",
        pl.col("round").alias("draft_round"),
        pl.col("pick").alias("draft_pick"),
        "pfr_player_id",
    )
    out = _attach_rookie_features(base, draft_picks, combine, vacated_2026)
    # "draft_pick" (raw) is kept for display even though only its log transform
    # (already in FEATURE_COLUMNS) feeds the model; "draft_round" is skipped
    # here since FEATURE_COLUMNS already carries it -- listing both would
    # duplicate the projection.
    return out.select("season", "gsis_id", "player_name", "position", "team", "draft_pick", *FEATURE_COLUMNS).sort(
        ["position", "draft_pick"]
    )


# --------------------------------------------------------------------------
# Model: shallow L2 logistic, time-based sanity split only
# --------------------------------------------------------------------------


def train_rookie_model(frame: pd.DataFrame, seed: int = SEED, C: float = LOGISTIC_C) -> dict:
    """Median-impute -> standardize -> L2 logistic, fit on seasons <= TRAIN_SEASONS_END.

    ``check_*`` metrics score the held-out later seasons purely as a sanity
    check ("does the ranking still make sense out-of-time") — no
    hyperparameter search ever looks at them, per the brief ("NO heavy
    tuning — it's a heuristic").
    """
    train_df = frame[frame["season"] <= TRAIN_SEASONS_END].reset_index(drop=True)
    check_df = frame[frame["season"] > TRAIN_SEASONS_END].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[FEATURE_COLUMNS]))

    pos = int(train_df["breakout"].sum())
    neg = len(train_df) - pos
    spw = float(neg / pos) if pos else 1.0
    model = LogisticRegression(C=C, class_weight={0: 1.0, 1: spw}, max_iter=2000, random_state=seed)
    model.fit(X_train, train_df["breakout"])

    check_pr_auc = float("nan")
    check_top10 = float("nan")
    if len(check_df):
        X_check = scaler.transform(imputer.transform(check_df[FEATURE_COLUMNS]))
        check_pred = model.predict_proba(X_check)[:, 1]
        check_pr_auc = mt.pr_auc(check_df["breakout"], check_pred)
        check_top10 = mt.top_k_precision(check_df["breakout"], check_pred, 10)

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "seed": seed,
        "C": C,
        "train_seasons_end": TRAIN_SEASONS_END,
        "train_n": int(len(train_df)),
        "train_pos": pos,
        "check_n": int(len(check_df)),
        "check_pos": int(check_df["breakout"].sum()) if len(check_df) else 0,
        "check_pr_auc": check_pr_auc,
        "check_top10_precision": check_top10,
    }


def score_rookies(bundle: dict, features_df: pd.DataFrame) -> np.ndarray:
    """Rookie-heuristic probability for every row of ``features_df`` (must carry FEATURE_COLUMNS)."""
    imputer, scaler, model = bundle["imputer"], bundle["scaler"], bundle["model"]
    X = scaler.transform(imputer.transform(features_df[bundle["feature_columns"]]))
    return model.predict_proba(X)[:, 1]


def save_bundle(bundle: dict, path: Path = ARTIFACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path = ARTIFACT_PATH) -> dict:
    return joblib.load(path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    print(f"rookie heuristic train | seasons {ROOKIE_SEASON_START}-{ROOKIE_SEASON_END}, sanity split at {TRAIN_SEASONS_END}")
    pl_frame = load_historical_rookie_frame()
    df = pl_frame.to_pandas()
    print(f"{len(df):,} historical rookie-season rows ({int(df['breakout'].sum())} breakouts)")

    bundle = train_rookie_model(df)
    print(
        f"train n={bundle['train_n']} (pos={bundle['train_pos']}) | "
        f"check n={bundle['check_n']} (pos={bundle['check_pos']}) seasons {TRAIN_SEASONS_END + 1}-{ROOKIE_SEASON_END}: "
        f"PR-AUC={bundle['check_pr_auc']:.3f} top10-precision={bundle['check_top10_precision']:.3f}"
    )

    save_bundle(bundle)
    print(f"wrote {ARTIFACT_PATH}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {k: v for k, v in bundle.items() if k not in ("model", "imputer", "scaler")}
    (OUTPUTS_DIR / "rookie_heuristic_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {OUTPUTS_DIR / 'rookie_heuristic_metrics.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
