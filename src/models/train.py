"""Position-parameterized breakout modeling & validation (Phase 4).

Generalizes ``src.models.train_wr`` (the WR-only Phase-4 module, kept as a
thin position-bound wrapper around this one — see its module docstring)
to any of the four skill positions. Every function that used to read a
WR-only module constant (``FEATURES_WR_PATH``, ``CONFIG_PATH``,
``REPORT_PATH``, ...) now takes a ``PositionSpec`` (``position_spec(pos)``)
instead; the modeling logic itself — fold fitting, Optuna tuning, blend
grid search, calibration, holdout evaluation — is unchanged from WR's
original implementation, because none of it was ever WR-specific to begin
with (it operates on an already-joined, already-position-filtered
DataFrame and a feature-column list).

Pipeline (``run_full_pipeline``): tune classifier head (LGBM + XGB + L2
logistic, per ``configs/model_{pos}.yaml``) on the four expanding-window
validation folds from ``src.models.cv`` -> blend the trio by grid search on
pooled out-of-fold validation predictions -> calibrate the blend (isotonic,
falling back to Platt if isotonic hurts pooled val Brier) -> tune a
regression head (finish_rank_delta) the same way -> freeze every
hyperparameter/weight/calibrator choice -> run the ONE holdout evaluation
(retrain on <=2023, score 2024 and 2025) -> save artifacts to
``data/models/``.

Modeling universe
------------------
``in_training_pool == 1`` rows of ``features_{pos}.parquet`` joined to
``labels.parquet`` on ``(season, gsis_id)`` — this already excludes rookies
and thin/low-signal seasons (see ``src/labels/build.py``). Nothing here
re-derives that filter; it is read straight off the label table.

Feature columns
----------------
- **Tree heads** (LGBM/XGB, both classifier and regressor) get every
  numeric column in the joined frame except identifiers and label columns
  (``tree_feature_columns``) — trees handle nulls and mixed scales
  natively, no reason to hand-curate. ``expectation_pos_rank`` and
  ``adp_source`` (one-hot, ``ecr`` dropped as baseline) are included: both
  are preseason-knowable, not a leak (see the brief).
- **Logistic head** gets a curated ~15-column subset
  (``configs/model_{pos}.yaml``'s ``logistic_feature_subset`` + the same
  ``adp_source`` dummies), median-imputed and standardized *within each
  fold's training data only* — logistic regression has no native null
  handling and is scale-sensitive the way trees are not.

Determinism
------------
Every model gets ``n_jobs=1``/``nthread=1`` and a fixed ``random_state``;
LightGBM additionally sets ``deterministic=True, force_row_wise=True``
(its own documented recipe for bit-identical reruns). Optuna's sampler is
seeded (``TPESampler(seed=...)``). Two calls to ``run_full_pipeline`` with
the same config must produce identical holdout predictions — locked by
``tests/test_models_positions.py::test_determinism`` (parameterized over
all four positions).

No early stopping at holdout time
-----------------------------------
LGBM/XGB tuning early-stops each fold against that fold's *own* validation
season — never against a holdout year. The final holdout retrain (<=2023,
no validation split available without touching 2024/2025) instead fixes
``n_estimators`` at the mean of the best_iteration each of the 4 frozen-
hyperparameter OOF fold models reached; see ``_model_best_iteration``.

Small-sample caveat (RB/TE/QB, vs. WR)
----------------------------------------
WR's training pool runs ~1,200-1,460 rows per fold with a 3-6% positive
rate. RB/TE/QB pools are smaller (roughly 500-850 rows) with as few as
1-2 positives in some individual validation folds/seasons — PR-AUC on
that few positives is expected to be extremely noisy, and a position can
legitimately fail to beat the prior-season-ppg baseline on holdout PR-AUC
at this sample size without indicating a pipeline bug. Every report
states the verdict plainly rather than hiding it.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl
import xgboost as xgb
import yaml
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models import baselines as bl
from src.models import metrics as mt
from src.models.cv import HOLDOUT_SEASONS, VALIDATION_SEASONS, Fold, holdout_split, validation_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)
# LightGBM's sklearn wrapper warns that the `eval_set` kwarg is being
# renamed; harmless and noisy across ~hundreds of fold fits during tuning.
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "data" / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CONFIG_DIR = REPO_ROOT / "configs"

LABELS_PATH = PROCESSED_DIR / "labels.parquet"

IDENTITY_COLS = ["season", "gsis_id", "player_name", "team"]
LABEL_JOIN_COLS = [
    "breakout",
    "finish_rank_delta",
    "finish_pos_rank",
    "expectation_pos_rank",
    "adp_source",
    "in_training_pool",
]
# Columns that are identity or label/outcome data, never model inputs.
NON_FEATURE_COLS = IDENTITY_COLS + [
    "breakout",
    "finish_rank_delta",
    "finish_pos_rank",
    "in_training_pool",
    "adp_source",
]

EARLY_STOPPING_ROUNDS = 30
REPORT_ID_COLS = ["season", "gsis_id", "player_name", "breakout", "expectation_pos_rank", "finish_pos_rank"]

POSITIONS = ("wr", "rb", "te", "qb")
_LABEL_POSITION = {"wr": "WR", "rb": "RB", "te": "TE", "qb": "QB"}


# --------------------------------------------------------------------------
# Position wiring
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionSpec:
    """Every path/name a position's modeling run needs, resolved once from its lowercase code."""

    position: str  # "wr", "rb", "te", "qb"
    label_position: str  # "WR", "RB", "TE", "QB" — matches labels.parquet's position column
    config_path: Path
    features_path: Path
    labels_path: Path
    report_path: Path
    metrics_json_path: Path
    artifact_path: Path


def position_spec(position: str) -> PositionSpec:
    """Build a ``PositionSpec`` from a lowercase position code (wr/rb/te/qb).

    Every path follows the repo's fixed naming convention
    (``configs/model_{pos}.yaml``, ``data/processed/features_{pos}.parquet``,
    ``outputs/model_{pos}_report.md``, ``data/models/{pos}_model_bundle.joblib``)
    so nothing here is guessed per-call.
    """
    pos = position.lower()
    if pos not in POSITIONS:
        raise ValueError(f"unknown position {position!r}; expected one of {POSITIONS}")
    return PositionSpec(
        position=pos,
        label_position=_LABEL_POSITION[pos],
        config_path=CONFIG_DIR / f"model_{pos}.yaml",
        features_path=PROCESSED_DIR / f"features_{pos}.parquet",
        labels_path=LABELS_PATH,
        report_path=OUTPUTS_DIR / f"model_{pos}_report.md",
        metrics_json_path=OUTPUTS_DIR / f"model_{pos}_metrics.json",
        artifact_path=MODELS_DIR / f"{pos}_model_bundle.joblib",
    )


# --------------------------------------------------------------------------
# Config + data loading
# --------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_modeling_frame(
    features_path: Path,
    labels_path: Path,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """features_{pos}.parquet joined to labels.parquet, restricted to in_training_pool==1.

    Pandas, not polars, from here on: everything downstream is
    sklearn/lightgbm/xgboost, which are array/pandas-native. src.features
    and src.labels stay polars because they are the data-pipeline layer;
    this module is the modeling layer. ``cfg`` is required for the
    adp_source dummy columns; callers that only need the raw joined frame
    (none, currently) would have to pass one anyway.
    """
    feats = pl.read_parquet(features_path)
    labels = pl.read_parquet(labels_path).select(["season", "gsis_id"] + LABEL_JOIN_COLS)
    joined = feats.join(labels, on=["season", "gsis_id"], how="inner")
    joined = joined.filter(pl.col("in_training_pool"))
    df = joined.sort(["season", "gsis_id"]).to_pandas()

    categories = cfg["adp_source_categories"]
    baseline_cat = cfg["adp_source_baseline"]
    for cat in categories:
        if cat == baseline_cat:
            continue
        df[f"adp_source_{cat}"] = (df["adp_source"] == cat).astype(float)

    return df.reset_index(drop=True)


def tree_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column except identity/label columns. Includes the adp_source_* dummies

    (already attached by load_modeling_frame) and expectation_pos_rank.
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def logistic_feature_columns(cfg: dict) -> list[str]:
    dummy_cols = [f"adp_source_{c}" for c in cfg["adp_source_categories"] if c != cfg["adp_source_baseline"]]
    return list(cfg["logistic_feature_subset"]) + dummy_cols


# --------------------------------------------------------------------------
# Fold utilities
# --------------------------------------------------------------------------


def fold_frames(df: pd.DataFrame, train_seasons: tuple[int, ...], val_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["season"].isin(train_seasons)]
    val_df = df[df["season"] == val_season]
    return train_df, val_df


def _scale_pos_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    return float(neg / pos) if pos > 0 else 1.0


def _model_best_iteration(model, model_type: str, fallback: int) -> int:
    """Trees actually used at prediction time, for the no-early-stopping holdout retrain.

    LightGBM's best_iteration_ is already a tree *count*. XGBoost's
    best_iteration is a 0-indexed round number, so +1 to get a count.
    """
    if model_type == "lgbm":
        val = getattr(model, "best_iteration_", None)
        return int(val) if val else fallback
    val = getattr(model, "best_iteration", None)
    return int(val) + 1 if val is not None else fallback


# --------------------------------------------------------------------------
# Classifier head: fold fit functions
# --------------------------------------------------------------------------


def fit_lgbm_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], params: dict, seed: int):
    spw = _scale_pos_weight(train_df["breakout"])
    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbosity=-1,
        metric="average_precision",
        scale_pos_weight=spw,
        **params,
    )
    model.fit(
        train_df[feature_cols],
        train_df["breakout"],
        eval_set=[(val_df[feature_cols], val_df["breakout"])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def fit_xgb_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], params: dict, seed: int):
    spw = _scale_pos_weight(train_df["breakout"])
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        eval_metric="aucpr",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        scale_pos_weight=spw,
        **params,
    )
    model.fit(train_df[feature_cols], train_df["breakout"], eval_set=[(val_df[feature_cols], val_df["breakout"])], verbose=False)
    return model


def fit_logistic(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], C: float, seed: int):
    """Median-impute (fit on train only) -> standardize (fit on train only) -> L2 logistic.

    scale_pos_weight's logistic analogue: class_weight={0: 1, 1: neg/pos}.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[feature_cols]))
    spw = _scale_pos_weight(train_df["breakout"])
    # penalty="l2" is sklearn's default; left implicit to dodge a 1.8+ FutureWarning
    # about the `penalty` kwarg without changing behavior.
    model = LogisticRegression(C=C, class_weight={0: 1.0, 1: spw}, max_iter=2000, random_state=seed)
    model.fit(X_train, train_df["breakout"])
    return model, imputer, scaler


def _logistic_predict(model, imputer, scaler, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = scaler.transform(imputer.transform(frame[feature_cols]))
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# Regression head: fold fit functions
# --------------------------------------------------------------------------


def fit_lgbm_regressor(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], params: dict, seed: int):
    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbosity=-1,
        metric="rmse",
        **params,
    )
    model.fit(
        train_df[feature_cols],
        train_df["finish_rank_delta"],
        eval_set=[(val_df[feature_cols], val_df["finish_rank_delta"])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def fit_xgb_regressor(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], params: dict, seed: int):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        eval_metric="rmse",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **params,
    )
    model.fit(
        train_df[feature_cols], train_df["finish_rank_delta"], eval_set=[(val_df[feature_cols], val_df["finish_rank_delta"])], verbose=False
    )
    return model


# --------------------------------------------------------------------------
# Optuna search spaces
# --------------------------------------------------------------------------


def _suggest_lgbm_params(trial: optuna.Trial, space: dict) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "num_leaves": trial.suggest_int("num_leaves", *space["num_leaves"]),
        "min_child_samples": trial.suggest_int("min_child_samples", *space["min_child_samples"]),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"]),
        "subsample": trial.suggest_float("subsample", *space["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
    }


def _suggest_xgb_params(trial: optuna.Trial, space: dict) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *space["min_child_weight"]),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"]),
        "subsample": trial.suggest_float("subsample", *space["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
    }


_SUGGEST_FN = {"lgbm": _suggest_lgbm_params, "xgb": _suggest_xgb_params}
_CLASSIFIER_FIT_FN = {"lgbm": fit_lgbm_classifier, "xgb": fit_xgb_classifier}
_REGRESSOR_FIT_FN = {"lgbm": fit_lgbm_regressor, "xgb": fit_xgb_regressor}


def params_at_boundary(params: dict, space: dict) -> list[str]:
    """Which of `params`' values sit at (or within 1% of the range of) their search-space edge."""
    flagged = []
    for name, bounds in space.items():
        if name not in params:
            continue
        lo, hi = float(bounds[0]), float(bounds[1])
        val = float(params[name])
        rng = hi - lo if hi > lo else 1.0
        tol = max(rng * 0.01, 1e-9)
        if val <= lo + tol or val >= hi - tol:
            flagged.append(f"{name}={params[name]} (range [{bounds[0]}, {bounds[1]}])")
    return flagged


# --------------------------------------------------------------------------
# Classifier tuning + OOF
# --------------------------------------------------------------------------


def tune_classifier(
    df: pd.DataFrame, feature_cols: list[str], folds: list[Fold], model_type: str, cfg: dict, n_trials: int, seed: int
) -> optuna.Study:
    space = cfg["search_space"][model_type]
    suggest_fn = _SUGGEST_FN[model_type]
    fit_fn = _CLASSIFIER_FIT_FN[model_type]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial, space)
        scores = []
        for fold in folds:
            train_df, val_df = fold_frames(df, fold.train_seasons, fold.val_season)
            model = fit_fn(train_df, val_df, feature_cols, params, seed)
            pred = model.predict_proba(val_df[feature_cols])[:, 1]
            scores.append(mt.pr_auc(val_df["breakout"], pred))
        return float(np.nanmean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def classifier_oof(
    df: pd.DataFrame, feature_cols: list[str], folds: list[Fold], model_type: str, params: dict, seed: int
) -> tuple[pd.DataFrame, list[dict], list[int]]:
    """Refit at the frozen params on each fold; return pooled OOF val predictions, per-fold

    PR-AUC (the fold-level-spread numbers the brief wants reported), and each fold's
    best_iteration (used to fix n_estimators for the no-early-stopping holdout retrain).
    """
    fit_fn = _CLASSIFIER_FIT_FN[model_type]
    rows, fold_metrics, best_iters = [], [], []
    for fold in folds:
        train_df, val_df = fold_frames(df, fold.train_seasons, fold.val_season)
        model = fit_fn(train_df, val_df, feature_cols, params, seed)
        pred = model.predict_proba(val_df[feature_cols])[:, 1]
        fold_metrics.append(
            {
                "val_season": fold.val_season,
                "pr_auc": mt.pr_auc(val_df["breakout"], pred),
                "n": int(len(val_df)),
                "n_pos": int(val_df["breakout"].sum()),
            }
        )
        best_iters.append(_model_best_iteration(model, model_type, params.get("n_estimators", 100)))
        out = val_df[REPORT_ID_COLS].copy()
        out["pred"] = pred
        rows.append(out)
    pooled = pd.concat(rows, ignore_index=True)
    return pooled, fold_metrics, best_iters


def tune_logistic(
    df: pd.DataFrame, feature_cols: list[str], folds: list[Fold], cfg: dict, seed: int
) -> tuple[float, float, dict[float, list[float]]]:
    """Small fixed grid over C (not Optuna — see configs/model_{pos}.yaml), same fold-mean-PR-AUC objective."""
    candidates = cfg["search_space"]["logistic"]["C"]
    best_C, best_score = None, -np.inf
    fold_scores_by_C: dict[float, list[float]] = {}
    for C in candidates:
        scores = []
        for fold in folds:
            train_df, val_df = fold_frames(df, fold.train_seasons, fold.val_season)
            model, imputer, scaler = fit_logistic(train_df, val_df, feature_cols, C, seed)
            pred = _logistic_predict(model, imputer, scaler, val_df, feature_cols)
            scores.append(mt.pr_auc(val_df["breakout"], pred))
        fold_scores_by_C[C] = scores
        mean_score = float(np.nanmean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
    return best_C, best_score, fold_scores_by_C


def logistic_oof(df: pd.DataFrame, feature_cols: list[str], folds: list[Fold], C: float, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    rows, fold_metrics = [], []
    for fold in folds:
        train_df, val_df = fold_frames(df, fold.train_seasons, fold.val_season)
        model, imputer, scaler = fit_logistic(train_df, val_df, feature_cols, C, seed)
        pred = _logistic_predict(model, imputer, scaler, val_df, feature_cols)
        fold_metrics.append(
            {
                "val_season": fold.val_season,
                "pr_auc": mt.pr_auc(val_df["breakout"], pred),
                "n": int(len(val_df)),
                "n_pos": int(val_df["breakout"].sum()),
            }
        )
        out = val_df[REPORT_ID_COLS].copy()
        out["pred"] = pred
        rows.append(out)
    pooled = pd.concat(rows, ignore_index=True)
    return pooled, fold_metrics


# --------------------------------------------------------------------------
# Blend + calibration
# --------------------------------------------------------------------------


def blend_grid_search(pred_frames: dict[str, pd.DataFrame], grid_step: float) -> tuple[dict[str, float], float, pd.DataFrame]:
    """Simplex grid search (integer steps to avoid float drift) over the classifier trio's

    weights, maximizing pooled validation PR-AUC. Returns (best_weights, best_pr_auc,
    merged frame carrying every model's pred_<name> column + breakout).
    """
    names = list(pred_frames.keys())
    base = pred_frames[names[0]][["season", "gsis_id", "breakout"]].copy()
    for name in names:
        base = base.merge(
            pred_frames[name][["season", "gsis_id", "pred"]].rename(columns={"pred": f"pred_{name}"}),
            on=["season", "gsis_id"],
            how="inner",
        )
    assert len(base) == len(pred_frames[names[0]]), "pooled OOF rows misaligned across model types"

    n_steps = int(round(1.0 / grid_step))
    best_weights, best_score = None, -np.inf
    for combo in product(range(n_steps + 1), repeat=len(names)):
        if sum(combo) != n_steps:
            continue
        weights = {name: c / n_steps for name, c in zip(names, combo)}
        blended = sum(weights[name] * base[f"pred_{name}"] for name in names)
        score = mt.pr_auc(base["breakout"], blended)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights, best_score, base


def apply_blend(weights: dict[str, float], **preds: np.ndarray) -> np.ndarray:
    total = None
    for name, w in weights.items():
        term = w * np.asarray(preds[name])
        total = term if total is None else total + term
    return total


def choose_calibration(y_true, raw_score) -> tuple[str, object, dict]:
    """Isotonic regression on pooled OOF val predictions; falls back to Platt (1-feature

    logistic) if isotonic doesn't beat raw pooled Brier — never both compared against
    holdout, only pooled validation, and the choice is frozen before holdout ever runs.
    """
    y_true = np.asarray(y_true)
    raw_score = np.asarray(raw_score, dtype=float)
    raw_brier = mt.brier(y_true, raw_score)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_score, y_true)
    iso_brier = mt.brier(y_true, iso.predict(raw_score))

    if iso_brier <= raw_brier:
        return "isotonic", iso, {"raw_brier": raw_brier, "isotonic_brier": iso_brier}

    platt = LogisticRegression()
    platt.fit(raw_score.reshape(-1, 1), y_true)
    platt_brier = mt.brier(y_true, platt.predict_proba(raw_score.reshape(-1, 1))[:, 1])
    return "platt", platt, {"raw_brier": raw_brier, "isotonic_brier": iso_brier, "platt_brier": platt_brier}


def apply_calibration(method: str, calibrator, score) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if method == "isotonic":
        return calibrator.predict(score)
    if method == "platt":
        return calibrator.predict_proba(score.reshape(-1, 1))[:, 1]
    raise ValueError(f"unknown calibration method: {method}")


# --------------------------------------------------------------------------
# Regression tuning + OOF
# --------------------------------------------------------------------------


def tune_regressor(
    df: pd.DataFrame, feature_cols: list[str], folds: list[Fold], model_type: str, cfg: dict, n_trials: int, seed: int
) -> optuna.Study:
    reg_df = df[df["finish_rank_delta"].notna()]
    space = cfg["search_space"][model_type]
    suggest_fn = _SUGGEST_FN[model_type]
    fit_fn = _REGRESSOR_FIT_FN[model_type]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial, space)
        scores = []
        for fold in folds:
            train_df, val_df = fold_frames(reg_df, fold.train_seasons, fold.val_season)
            model = fit_fn(train_df, val_df, feature_cols, params, seed)
            pred = model.predict(val_df[feature_cols])
            scores.append(mt.rmse(val_df["finish_rank_delta"], pred))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def regressor_oof(
    df: pd.DataFrame, feature_cols: list[str], folds: list[Fold], model_type: str, params: dict, seed: int
) -> tuple[pd.DataFrame, list[dict], list[int]]:
    reg_df = df[df["finish_rank_delta"].notna()]
    fit_fn = _REGRESSOR_FIT_FN[model_type]
    rows, fold_metrics, best_iters = [], [], []
    for fold in folds:
        train_df, val_df = fold_frames(reg_df, fold.train_seasons, fold.val_season)
        model = fit_fn(train_df, val_df, feature_cols, params, seed)
        pred = model.predict(val_df[feature_cols])
        fold_metrics.append({"val_season": fold.val_season, "rmse": mt.rmse(val_df["finish_rank_delta"], pred), "n": int(len(val_df))})
        best_iters.append(_model_best_iteration(model, model_type, params.get("n_estimators", 100)))
        out = val_df[["season", "gsis_id"]].copy()
        out["pred"] = pred
        out["actual"] = val_df["finish_rank_delta"].to_numpy()
        rows.append(out)
    pooled = pd.concat(rows, ignore_index=True)
    return pooled, fold_metrics, best_iters


# --------------------------------------------------------------------------
# Holdout evaluation (run once)
# --------------------------------------------------------------------------


def holdout_predictions(df: pd.DataFrame, seed: int, frozen: dict) -> pd.DataFrame:
    """Retrain classifier trio on <=2023 with every frozen hyperparameter, predict 2024+2025,

    apply the frozen blend weights and the frozen calibrator. No tuning happens here.
    """
    split = holdout_split()
    train_df = df[df["season"].isin(split.train_seasons)]
    holdout_df = df[df["season"].isin(split.holdout_seasons)]

    tree_cols = frozen["tree_feature_cols"]
    logit_cols = frozen["logistic_feature_cols"]
    spw = _scale_pos_weight(train_df["breakout"])

    lgbm_params = dict(frozen["lgbm_params"])
    lgbm_params["n_estimators"] = frozen["lgbm_final_n_estimators"]
    lgbm_model = lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbosity=-1,
        scale_pos_weight=spw,
        **lgbm_params,
    )
    lgbm_model.fit(train_df[tree_cols], train_df["breakout"])
    lgbm_pred = lgbm_model.predict_proba(holdout_df[tree_cols])[:, 1]

    xgb_params = dict(frozen["xgb_params"])
    xgb_params["n_estimators"] = frozen["xgb_final_n_estimators"]
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=seed, n_jobs=1, verbosity=0, scale_pos_weight=spw, **xgb_params)
    xgb_model.fit(train_df[tree_cols], train_df["breakout"])
    xgb_pred = xgb_model.predict_proba(holdout_df[tree_cols])[:, 1]

    logit_model, logit_imputer, logit_scaler = fit_logistic(train_df, holdout_df, logit_cols, frozen["logistic_C"], seed)
    logit_pred = _logistic_predict(logit_model, logit_imputer, logit_scaler, holdout_df, logit_cols)

    blended = apply_blend(frozen["blend_weights"], lgbm=lgbm_pred, xgb=xgb_pred, logistic=logit_pred)
    calibrated = apply_calibration(frozen["calibration_method"], frozen["calibrator"], blended)

    out = holdout_df[REPORT_ID_COLS].copy()
    out["pred_lgbm"] = lgbm_pred
    out["pred_xgb"] = xgb_pred
    out["pred_logistic"] = logit_pred
    out["pred_blend_raw"] = blended
    out["pred_calibrated"] = calibrated

    frozen["_holdout_models"] = {
        "lgbm": lgbm_model,
        "xgb": xgb_model,
        "logistic": (logit_model, logit_imputer, logit_scaler),
    }
    return out


def holdout_regression_eval(df: pd.DataFrame, seed: int, frozen: dict) -> dict:
    """Retrain the regression head on <=2023 at its frozen hyperparameters, score 2024+2025.

    Stashes the two fitted regressors on ``frozen["_holdout_reg_models"]`` — the same
    "compute + persist in one pass" pattern ``holdout_predictions`` uses for the classifier
    trio — so ``save_artifacts`` can ship a bundle Phase 6 can score 2026 rows with, instead
    of only ever reporting an RMSE and discarding the fitted models.
    """
    split = holdout_split()
    reg_df = df[df["finish_rank_delta"].notna()]
    train_df = reg_df[reg_df["season"].isin(split.train_seasons)]
    holdout_df = reg_df[reg_df["season"].isin(split.holdout_seasons)]
    tree_cols = frozen["tree_feature_cols"]

    lgbm_params = dict(frozen["lgbm_reg_params"])
    lgbm_params["n_estimators"] = frozen["lgbm_reg_final_n_estimators"]
    lgbm_model = lgb.LGBMRegressor(
        objective="regression", random_state=seed, deterministic=True, force_row_wise=True, n_jobs=1, verbosity=-1, **lgbm_params
    )
    lgbm_model.fit(train_df[tree_cols], train_df["finish_rank_delta"])
    lgbm_rmse = mt.rmse(holdout_df["finish_rank_delta"], lgbm_model.predict(holdout_df[tree_cols]))

    xgb_params = dict(frozen["xgb_reg_params"])
    xgb_params["n_estimators"] = frozen["xgb_reg_final_n_estimators"]
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1, verbosity=0, **xgb_params)
    xgb_model.fit(train_df[tree_cols], train_df["finish_rank_delta"])
    xgb_rmse = mt.rmse(holdout_df["finish_rank_delta"], xgb_model.predict(holdout_df[tree_cols]))

    frozen["_holdout_reg_models"] = {"lgbm_reg": lgbm_model, "xgb_reg": xgb_model}
    return {"lgbm_rmse": lgbm_rmse, "xgb_rmse": xgb_rmse, "n": int(len(holdout_df))}


# --------------------------------------------------------------------------
# Evaluation tables (validation + holdout, model + baselines)
# --------------------------------------------------------------------------


def evaluate_scores(
    df_subset: pd.DataFrame, score_cols: dict[str, pd.Series], seasons: list[int], brier_names: frozenset[str] = frozenset()
) -> pd.DataFrame:
    """Tidy per-season + pooled metrics: one row per (model, season|'pooled')."""
    rows = []
    for name, score in score_cols.items():
        score = pd.Series(np.asarray(score), index=df_subset.index)
        for season in list(seasons) + ["pooled"]:
            mask = df_subset["season"] == season if season != "pooled" else pd.Series(True, index=df_subset.index)
            y = df_subset.loc[mask, "breakout"]
            s = score.loc[mask]
            rows.append(
                {
                    "model": name,
                    "season": season,
                    "n": int(mask.sum()),
                    "n_pos": int(y.sum()),
                    "pr_auc": mt.pr_auc(y, s),
                    "top10_precision": mt.top_k_precision(y, s, 10),
                    "brier": mt.brier(y, s) if name in brier_names else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def named_top10(df_subset: pd.DataFrame, season: int, score_col: str) -> pd.DataFrame:
    sub = df_subset[df_subset["season"] == season].reset_index(drop=True)
    idx = mt.top_k_indices(sub[score_col].to_numpy(), 10)
    cols = ["player_name", score_col, "expectation_pos_rank", "finish_pos_rank", "breakout"]
    return sub.iloc[idx][cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------


def run_full_pipeline(
    spec: PositionSpec,
    cfg: dict | None = None,
    classifier_trials: int | None = None,
    regression_trials: int | None = None,
    seed: int | None = None,
) -> dict:
    """Tune -> blend -> calibrate -> ONE holdout eval -> save artifacts. Returns everything

    the report/metrics-JSON writers and the tests need. `classifier_trials` /
    `regression_trials` / `seed` override cfg (used by tests for a tiny, fast config);
    leave None for the real Deliverable-4 run.
    """
    if cfg is None:
        cfg = load_config(spec.config_path)
    seed = cfg["seed"] if seed is None else seed
    n_trials_clf = cfg["optuna"]["classifier_trials"] if classifier_trials is None else classifier_trials
    n_trials_reg = cfg["optuna"]["regression_trials"] if regression_trials is None else regression_trials

    np.random.seed(seed)

    df = load_modeling_frame(spec.features_path, spec.labels_path, cfg=cfg)
    tree_cols = tree_feature_columns(df)
    logit_cols = logistic_feature_columns(cfg)
    folds = validation_folds()

    # ---- classifier tuning (Optuna: LGBM, XGB; small grid: logistic) ----
    lgbm_study = tune_classifier(df, tree_cols, folds, "lgbm", cfg, n_trials_clf, seed)
    xgb_study = tune_classifier(df, tree_cols, folds, "xgb", cfg, n_trials_clf, seed)
    logit_C, logit_val_score, logit_fold_scores_by_C = tune_logistic(df, logit_cols, folds, cfg, seed)

    lgbm_best_params, xgb_best_params = lgbm_study.best_params, xgb_study.best_params

    # ---- OOF predictions at frozen best params (feeds blend + calibration) ----
    lgbm_oof, lgbm_fold_metrics, lgbm_best_iters = classifier_oof(df, tree_cols, folds, "lgbm", lgbm_best_params, seed)
    xgb_oof, xgb_fold_metrics, xgb_best_iters = classifier_oof(df, tree_cols, folds, "xgb", xgb_best_params, seed)
    logit_oof, logit_fold_metrics = logistic_oof(df, logit_cols, folds, logit_C, seed)

    # ---- blend weight grid search on pooled OOF val predictions ----
    blend_weights, blend_pr_auc, pooled = blend_grid_search({"lgbm": lgbm_oof, "xgb": xgb_oof, "logistic": logit_oof}, cfg["blend"]["grid_step"])
    pooled["pred_blend"] = apply_blend(blend_weights, lgbm=pooled["pred_lgbm"], xgb=pooled["pred_xgb"], logistic=pooled["pred_logistic"])

    # ---- calibration on pooled OOF val predictions of the blend ----
    calib_method, calibrator, calib_diag = choose_calibration(pooled["breakout"], pooled["pred_blend"])
    pooled["pred_calibrated"] = apply_calibration(calib_method, calibrator, pooled["pred_blend"])

    # ---- regression head (separate: RMSE, own Optuna studies) ----
    lgbm_reg_study = tune_regressor(df, tree_cols, folds, "lgbm", cfg, n_trials_reg, seed)
    xgb_reg_study = tune_regressor(df, tree_cols, folds, "xgb", cfg, n_trials_reg, seed)
    lgbm_reg_oof, lgbm_reg_fold_metrics, lgbm_reg_best_iters = regressor_oof(df, tree_cols, folds, "lgbm", lgbm_reg_study.best_params, seed)
    xgb_reg_oof, xgb_reg_fold_metrics, xgb_reg_best_iters = regressor_oof(df, tree_cols, folds, "xgb", xgb_reg_study.best_params, seed)

    # ---- freeze everything ----
    frozen = {
        "tree_feature_cols": tree_cols,
        "logistic_feature_cols": logit_cols,
        "lgbm_params": lgbm_best_params,
        "xgb_params": xgb_best_params,
        "logistic_C": logit_C,
        "lgbm_final_n_estimators": int(round(np.mean(lgbm_best_iters))),
        "xgb_final_n_estimators": int(round(np.mean(xgb_best_iters))),
        "blend_weights": blend_weights,
        "calibration_method": calib_method,
        "calibrator": calibrator,
        "lgbm_reg_params": lgbm_reg_study.best_params,
        "xgb_reg_params": xgb_reg_study.best_params,
        "lgbm_reg_final_n_estimators": int(round(np.mean(lgbm_reg_best_iters))),
        "xgb_reg_final_n_estimators": int(round(np.mean(xgb_reg_best_iters))),
    }

    # ---- the ONE holdout evaluation ----
    holdout_preds = holdout_predictions(df, seed, frozen)
    holdout_reg = holdout_regression_eval(df, seed, frozen)

    save_artifacts(spec, frozen, seed, cfg)

    return {
        "spec": spec,
        "cfg": cfg,
        "seed": seed,
        "n_trials_clf": n_trials_clf,
        "n_trials_reg": n_trials_reg,
        "df": df,
        "tree_cols": tree_cols,
        "logit_cols": logit_cols,
        "folds": folds,
        "lgbm_study": lgbm_study,
        "xgb_study": xgb_study,
        "logit_C": logit_C,
        "logit_val_score": logit_val_score,
        "logit_fold_scores_by_C": logit_fold_scores_by_C,
        "lgbm_fold_metrics": lgbm_fold_metrics,
        "xgb_fold_metrics": xgb_fold_metrics,
        "logit_fold_metrics": logit_fold_metrics,
        "blend_weights": blend_weights,
        "blend_pr_auc": blend_pr_auc,
        "pooled_val": pooled,
        "calib_method": calib_method,
        "calib_diag": calib_diag,
        "lgbm_reg_study": lgbm_reg_study,
        "xgb_reg_study": xgb_reg_study,
        "lgbm_reg_fold_metrics": lgbm_reg_fold_metrics,
        "xgb_reg_fold_metrics": xgb_reg_fold_metrics,
        "holdout_preds": holdout_preds,
        "holdout_reg": holdout_reg,
        "frozen": frozen,
    }


def save_artifacts(spec: PositionSpec, frozen: dict, seed: int, cfg: dict) -> None:
    """Persist the frozen holdout-retrained bundle: classifier trio + blend/calibration

    metadata (unchanged) plus the two holdout-retrained regressors
    (``lgbm_reg``/``xgb_reg``, keyed distinctly from the classifier's ``lgbm``/``xgb`` so
    both survive in one payload) — Phase 6 scores 2026 rows with exactly these objects,
    never re-fitting anything.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _PRIVATE_KEYS = ("_holdout_models", "_holdout_reg_models")
    to_save = {k: v for k, v in frozen.items() if k not in _PRIVATE_KEYS}
    to_save["seed"] = seed
    to_save["cfg"] = cfg
    to_save["position"] = spec.position
    payload = {**to_save, **frozen.get("_holdout_models", {}), **frozen.get("_holdout_reg_models", {})}
    joblib.dump(payload, spec.artifact_path)


# --------------------------------------------------------------------------
# Reporting (Deliverable 4). No new dependency for markdown tables (no
# `tabulate` in pyproject.toml) -- a small manual formatter instead.
# --------------------------------------------------------------------------

_MODEL_LABEL = {
    "model": "**Model (blend, calibrated)**",
    "adp_knows_best": "Baseline 1: ADP-knows-best",
    "prior_season_ppg_rank": "Baseline 2: prior-season ppg rank",
    "age_adjusted_adp": "Baseline 3: age-adjusted ADP",
}


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _metrics_table_markdown(metrics_df: pd.DataFrame, seasons: list) -> str:
    headers = ["Model", "Season", "n", "n_pos", "PR-AUC", "Brier", "Top-10 precision"]
    rows = []
    for season in list(seasons) + ["pooled"]:
        for model_key in ["model", "adp_knows_best", "prior_season_ppg_rank", "age_adjusted_adp"]:
            r = metrics_df[(metrics_df["season"] == season) & (metrics_df["model"] == model_key)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(
                [_MODEL_LABEL[model_key], season, int(r["n"]), int(r["n_pos"]), r["pr_auc"], r["brier"], r["top10_precision"]]
            )
    return _markdown_table(headers, rows)


def _top10_table_markdown(top10: pd.DataFrame, score_label: str) -> str:
    headers = ["Player", score_label, "Expectation rank", "Actual finish rank", "Breakout"]
    rows = []
    for _, r in top10.iterrows():
        hit = "YES" if r["breakout"] == 1 else "no"
        score_col = [c for c in top10.columns if c not in ("player_name", "expectation_pos_rank", "finish_pos_rank", "breakout")][0]
        finish = r["finish_pos_rank"]
        finish_str = int(finish) if pd.notna(finish) else "n/a (<8 games)"
        rows.append([r["player_name"], r[score_col], int(r["expectation_pos_rank"]), finish_str, hit])
    return _markdown_table(headers, rows)


def _fold_spread_lines(fold_metrics: list[dict], key: str) -> str:
    parts = [f"{fm['val_season']}: {fm[key]:.3f} (n={fm['n']}, n_pos={fm.get('n_pos', '?')})" for fm in fold_metrics]
    return "; ".join(parts)


def generate_report(result: dict, pytest_summary: str | None = None) -> str:
    spec: PositionSpec = result["spec"]
    cfg = result["cfg"]
    df = result["df"]
    pooled = result["pooled_val"]
    holdout_preds = result["holdout_preds"]

    val_full = df[df["season"].isin(VALIDATION_SEASONS)].merge(
        pooled[["season", "gsis_id", "pred_calibrated"]], on=["season", "gsis_id"], how="left"
    )
    assert val_full["pred_calibrated"].notna().all(), "every validation row must have a pooled OOF prediction"
    val_scores = {"model": val_full["pred_calibrated"], **bl.compute_all_baselines(val_full, cfg)}
    val_metrics = evaluate_scores(val_full, val_scores, list(VALIDATION_SEASONS), brier_names=frozenset({"model"}))

    holdout_full = df[df["season"].isin(HOLDOUT_SEASONS)].merge(
        holdout_preds[["season", "gsis_id", "pred_calibrated"]], on=["season", "gsis_id"], how="left"
    )
    assert holdout_full["pred_calibrated"].notna().all(), "every holdout row must have a prediction"
    holdout_scores = {"model": holdout_full["pred_calibrated"], **bl.compute_all_baselines(holdout_full, cfg)}
    holdout_metrics = evaluate_scores(holdout_full, holdout_scores, list(HOLDOUT_SEASONS), brier_names=frozenset({"model"}))

    decile = mt.calibration_decile_table(val_full["breakout"], val_full["pred_calibrated"])

    top10_2024 = named_top10(holdout_full, 2024, "pred_calibrated")
    top10_2025 = named_top10(holdout_full, 2025, "pred_calibrated")

    # ---- honesty verdict: model vs baseline 2 (prior-season ppg rank), HOLDOUT only ----
    pooled_row = lambda m, model_key: holdout_metrics[(holdout_metrics["season"] == "pooled") & (holdout_metrics["model"] == model_key)].iloc[0][m]
    model_top10 = pooled_row("top10_precision", "model")
    base2_top10 = pooled_row("top10_precision", "prior_season_ppg_rank")
    model_prauc = pooled_row("pr_auc", "model")
    base2_prauc = pooled_row("pr_auc", "prior_season_ppg_rank")
    beats_top10 = model_top10 > base2_top10
    beats_prauc = model_prauc > base2_prauc
    verdict = (
        f"**Verdict:** on the HOLDOUT years (2024+2025 pooled), the model "
        f"{'BEATS' if beats_top10 else 'DOES NOT beat'} baseline 2 (prior-season ppg rank) on top-10 precision "
        f"({model_top10:.3f} vs {base2_top10:.3f}, delta {model_top10 - base2_top10:+.3f}) and "
        f"{'BEATS' if beats_prauc else 'DOES NOT beat'} it on PR-AUC "
        f"({model_prauc:.3f} vs {base2_prauc:.3f}, delta {model_prauc - base2_prauc:+.3f}). "
        + ("Both this run's holdout numbers are reported as-is, from the single frozen holdout evaluation -- no retuning followed." )
    )

    lgbm_space = cfg["search_space"]["lgbm"]
    xgb_space = cfg["search_space"]["xgb"]
    lgbm_boundary = params_at_boundary(result["frozen"]["lgbm_params"], lgbm_space)
    xgb_boundary = params_at_boundary(result["frozen"]["xgb_params"], xgb_space)
    lgbm_reg_boundary = params_at_boundary(result["frozen"]["lgbm_reg_params"], lgbm_space)
    xgb_reg_boundary = params_at_boundary(result["frozen"]["xgb_reg_params"], xgb_space)

    pool_n = len(df)
    pos_rates = df.groupby("season")["breakout"].mean()

    lines = []
    lines.append(f"# BreakoutLab -- {spec.label_position} Model Report (Phase 4)")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append(
        f"{pool_n:,} rows in the {spec.label_position} training pool (in_training_pool==1, "
        f"{df['season'].min()}-{df['season'].max()}), positive rate {pos_rates.min() * 100:.1f}%-"
        f"{pos_rates.max() * 100:.1f}% per season (see per-fold `n_pos` below): PR-AUC is expected "
        "to be noisy fold-to-fold, more so than WR's larger pool. Fold-level numbers are reported "
        "alongside every pooled mean for exactly that reason -- read the spread, not just the mean."
    )
    lines.append("")

    lines.append("## Validation: per-fold + pooled (2020-2023)")
    lines.append("")
    lines.append(_metrics_table_markdown(val_metrics, list(VALIDATION_SEASONS)))
    lines.append("")
    lines.append("### Fold-level spread (classifier trio, frozen best params)")
    lines.append("")
    lines.append(f"- LGBM PR-AUC by fold: {_fold_spread_lines(result['lgbm_fold_metrics'], 'pr_auc')}")
    lines.append(f"- XGB PR-AUC by fold: {_fold_spread_lines(result['xgb_fold_metrics'], 'pr_auc')}")
    lines.append(f"- Logistic PR-AUC by fold: {_fold_spread_lines(result['logit_fold_metrics'], 'pr_auc')}")
    lines.append("")

    lines.append("## Holdout: per-year + pooled (2024-2025, ONE evaluation)")
    lines.append("")
    lines.append(_metrics_table_markdown(holdout_metrics, list(HOLDOUT_SEASONS)))
    lines.append("")

    lines.append("## Calibration check: decile table (pooled validation)")
    lines.append("")
    decile_headers = ["Predicted-prob bucket", "n", "Mean predicted", "Realized rate"]
    decile_rows = [[str(r["bucket"]), int(r["n"]), r["mean_predicted"], r["realized_rate"]] for _, r in decile.iterrows()]
    lines.append(_markdown_table(decile_headers, decile_rows))
    lines.append("")
    lines.append(f"Calibration method chosen: **{result['calib_method']}** (pooled val Brier -- {result['calib_diag']}).")
    lines.append("")

    lines.append("## Holdout top-10 named lists")
    lines.append("")
    lines.append("### 2024")
    lines.append("")
    lines.append(_top10_table_markdown(top10_2024, "Calibrated probability"))
    lines.append("")
    lines.append("### 2025")
    lines.append("")
    lines.append(_top10_table_markdown(top10_2025, "Calibrated probability"))
    lines.append("")

    lines.append("## Chosen hyperparameters, blend weights, calibration")
    lines.append("")
    lines.append(f"- **LGBM classifier** best params: `{result['frozen']['lgbm_params']}`")
    lines.append(f"  - at search-space boundary: {lgbm_boundary if lgbm_boundary else 'none'}")
    lines.append(f"  - final n_estimators for holdout retrain (mean OOF best_iteration, no early stop against holdout): {result['frozen']['lgbm_final_n_estimators']}")
    lines.append(f"- **XGB classifier** best params: `{result['frozen']['xgb_params']}`")
    lines.append(f"  - at search-space boundary: {xgb_boundary if xgb_boundary else 'none'}")
    lines.append(f"  - final n_estimators for holdout retrain: {result['frozen']['xgb_final_n_estimators']}")
    lines.append(f"- **Logistic classifier**: C={result['logit_C']} (grid search, mean val PR-AUC={result['logit_val_score']:.3f}), curated {len(result['logit_cols'])}-feature subset (incl. adp_source dummies): `{result['logit_cols']}`")
    lines.append(f"- **Blend weights** (grid step {cfg['blend']['grid_step']}, pooled val PR-AUC={result['blend_pr_auc']:.3f}): `{result['blend_weights']}`")
    lines.append(f"- **Calibration**: {result['calib_method']}")
    lines.append(f"- **LGBM regressor** best params: `{result['frozen']['lgbm_reg_params']}` (boundary: {lgbm_reg_boundary if lgbm_reg_boundary else 'none'}); holdout RMSE={result['holdout_reg']['lgbm_rmse']:.3f}")
    lines.append(f"- **XGB regressor** best params: `{result['frozen']['xgb_reg_params']}` (boundary: {xgb_reg_boundary if xgb_reg_boundary else 'none'}); holdout RMSE={result['holdout_reg']['xgb_rmse']:.3f}")
    lines.append(f"- Regression fold RMSE -- LGBM: {_fold_spread_lines(result['lgbm_reg_fold_metrics'], 'rmse')}")
    lines.append(f"- Regression fold RMSE -- XGB: {_fold_spread_lines(result['xgb_reg_fold_metrics'], 'rmse')}")
    lines.append(f"- seed={result['seed']}, classifier Optuna trials={result['n_trials_clf']}, regression Optuna trials={result['n_trials_reg']}")
    lines.append("")

    if pytest_summary:
        lines.append("## pytest summary")
        lines.append("")
        lines.append("```")
        lines.append(pytest_summary.strip())
        lines.append("```")
        lines.append("")

    lines.append("## Deviations from the brief")
    lines.append("")
    lines.append(
        "- Logistic head's regularization strength (C) is chosen by a small fixed grid "
        f"(`configs/model_{spec.position}.yaml`'s `search_space.logistic.C`), not Optuna -- the brief "
        "scopes Optuna tuning to LGBM/XGB only (\"Optuna per model type (LGBM/XGB)\")."
    )
    lines.append(
        "- LGBM's early-stopping/validation metric is `average_precision` (LightGBM's native "
        "PR-AUC metric) and XGB's is `aucpr` (XGBoost's native PR-AUC metric) -- both are the "
        "PR-AUC family the brief asks Optuna to maximize, so early stopping and the tuning "
        "objective agree on what \"better\" means."
    )
    lines.append(
        "- The holdout retrain cannot early-stop against 2024/2025 without leaking the holdout "
        "into training. n_estimators for the holdout retrain is fixed at the mean best_iteration "
        "each frozen-hyperparameter OOF fold model reached during validation (see per-model lines "
        "above) -- a validation-only decision, never touching holdout years."
    )
    lines.append(
        "- `adp_source` (ecr/proxy/capped) is one-hot encoded with `ecr` as the dropped baseline "
        "category, applied identically to both the logistic subset and the LGBM/XGB full feature set."
    )
    lines.append("")

    return "\n".join(lines)


def _metrics_to_jsonable(result: dict) -> dict:
    cfg = result["cfg"]
    df = result["df"]
    pooled = result["pooled_val"]
    holdout_preds = result["holdout_preds"]

    val_full = df[df["season"].isin(VALIDATION_SEASONS)].merge(
        pooled[["season", "gsis_id", "pred_calibrated"]], on=["season", "gsis_id"], how="left"
    )
    val_scores = {"model": val_full["pred_calibrated"], **bl.compute_all_baselines(val_full, cfg)}
    val_metrics = evaluate_scores(val_full, val_scores, list(VALIDATION_SEASONS), brier_names=frozenset({"model"}))

    holdout_full = df[df["season"].isin(HOLDOUT_SEASONS)].merge(
        holdout_preds[["season", "gsis_id", "pred_calibrated"]], on=["season", "gsis_id"], how="left"
    )
    holdout_scores = {"model": holdout_full["pred_calibrated"], **bl.compute_all_baselines(holdout_full, cfg)}
    holdout_metrics = evaluate_scores(holdout_full, holdout_scores, list(HOLDOUT_SEASONS), brier_names=frozenset({"model"}))

    def _records(d: pd.DataFrame) -> list[dict]:
        return json.loads(d.to_json(orient="records"))

    return {
        "position": result["spec"].position,
        "seed": result["seed"],
        "validation_metrics": _records(val_metrics),
        "holdout_metrics": _records(holdout_metrics),
        "blend_weights": result["blend_weights"],
        "blend_pooled_val_pr_auc": result["blend_pr_auc"],
        "calibration_method": result["calib_method"],
        "calibration_diagnostics": result["calib_diag"],
        "lgbm_classifier_params": result["frozen"]["lgbm_params"],
        "xgb_classifier_params": result["frozen"]["xgb_params"],
        "logistic_C": result["logit_C"],
        "lgbm_fold_metrics": result["lgbm_fold_metrics"],
        "xgb_fold_metrics": result["xgb_fold_metrics"],
        "logit_fold_metrics": result["logit_fold_metrics"],
        "lgbm_regressor_params": result["frozen"]["lgbm_reg_params"],
        "xgb_regressor_params": result["frozen"]["xgb_reg_params"],
        "holdout_regression_rmse": result["holdout_reg"],
    }


def write_outputs(spec: PositionSpec, result: dict, pytest_summary: str | None = None) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_md = generate_report(result, pytest_summary=pytest_summary)
    spec.report_path.write_text(report_md)
    metrics = _metrics_to_jsonable(result)
    spec.metrics_json_path.write_text(json.dumps(metrics, indent=2, default=str))


def _run_pytest_summary() -> str:
    """Run the full test suite and return its final summary line(s) for the report."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=600
    )
    tail_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-5:]
    return "\n".join(tail_lines)


def main(position: str) -> int:
    spec = position_spec(position)
    print(f"{spec.label_position} model training | Phase 4 (full config: see {spec.config_path})")
    print("running pytest first, for the report's pytest-summary section ...")
    pytest_summary = _run_pytest_summary()
    print(pytest_summary)

    cfg = load_config(spec.config_path)
    result = run_full_pipeline(spec, cfg=cfg)
    write_outputs(spec, result, pytest_summary=pytest_summary)
    print(f"wrote {spec.report_path}")
    print(f"wrote {spec.metrics_json_path}")
    print(f"wrote {spec.artifact_path}")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in POSITIONS:
        print(f"usage: python -m src.models.train <{'|'.join(POSITIONS)}>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
