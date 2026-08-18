"""v2.0 quantile regression head: predicts each player's PPR points-per-game

*distribution* (5 quantiles: 0.10/0.25/0.50/0.75/0.90) via LightGBM
``objective="quantile"``, one model per position per alpha.

Why this exists (see README's v2.0 section for the full writeup): the v1.7
binary breakout classifier predicts a rare joint event (top-K finish AND
cheap preseason ADP). At RB/TE's sample sizes that event is so rare the
calibrated probabilities collapse toward the position's flat base rate --
every TE reads ~1.8% regardless of who they are -- and a player who is
*definitionally* ineligible (already priced as a starter) still gets a
"breakout odds" number that means nothing for him. Predicting the full
per-game scoring distribution sidesteps both problems: it is a continuous
regression target with no class-imbalance noise, and every derived quantity
(finish-rank probability, floor/ceiling range, startable odds) is computed
from that one distribution rather than needing its own bespoke label and
classifier. ``src.inference.projections`` turns this module's quantile
predictions into those derived quantities; this module only trains and
evaluates the distribution itself. The v1.7 classifier is NOT replaced --
it stays the comparator (``src.models.train``, unchanged) and the two are
compared head-to-head in ``comparison_gate`` below.

Modeling universe
------------------
``in_training_pool == 1`` AND ``games >= 8`` rows of
``features_{pos}.parquet`` joined to ``labels.parquet`` -- the same
``in_training_pool`` gate the classifier uses, further restricted to
``games >= 8`` because that is the exact basis ``src.labels.build`` uses to
compute ``finish_pos_rank`` in the first place (a 2-game per-game rate is
not a real season rate to regress on). Same per-position pool start
(``configs/model_{pos}.yaml``'s ``train_season_start`` -- RB's 2020+ real-
market-only restriction carries over unchanged) and the same
``excluded_features`` gate (the Vegas-feature promotion rule) as the
classifier, via the identical config file.

Target: ``ppr_ppg`` (points per game, not season total) -- this module
predicts a *rate*, matching the finish-rank basis and letting per-game
predictions be compared across players with different games-played
(``src.inference.projections`` documents the "no injury/availability
modeling" caveat that follows from this).

Tuning discipline
-------------------
Hyperparameters (LightGBM tree shape/regularization -- the identical
``search_space.lgbm`` block ``configs/model_{pos}.yaml`` already carries
for the classifier) are tuned ONCE, at alpha=0.50, on the four expanding-
window validation folds (``src.models.cv``), Optuna TPE, single seed 42 --
see ``configs/model_{pos}.yaml``'s ``quantile`` block for why one seed is
enough here (unlike the classifier's v1.7 seed ensemble). The frozen params
are then reused unmodified for all 5 alphas; only ``n_estimators`` is
refit per alpha (via that alpha's own early stopping against its own
pinball loss), matching how the classifier head fixes ``n_estimators`` at
holdout-retrain time (``src.models.train._model_best_iteration``).

Non-crossing
-------------
Nothing in the 5 independently-fit quantile regressors *guarantees*
q10 <= q25 <= q50 <= q75 <= q90 for a given row -- each is its own model.
Enforced post-hoc, everywhere predictions are produced (OOF and holdout
alike), by sorting the 5 predicted values ascending per row
(``enforce_noncrossing``) -- simple, exact, and the brief's prescribed fix.

Evaluation
-----------
Validation (pooled OOF across the four folds) and the ONE frozen holdout
(2024-2025, retrain on <=2023, exactly the classifier's holdout split):
pinball loss per alpha, empirical coverage of the q10-q90 interval (target
80%), and Spearman(q50, actual ppg). ``comparison_gate`` (Deliverable 3) is
the head-to-head decision: on the identical holdout rows the classifier
was judged on, does ranking ELIGIBLE players by quantile-derived
p_startable beat the classifier's own top-10 precision? Ties favor the
quantile head (richer output).
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl

from src.models import metrics as mt
from src.models import train
from src.models.cv import Fold, holdout_split, validation_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

REPO_ROOT = train.REPO_ROOT
PROCESSED_DIR = train.PROCESSED_DIR
MODELS_DIR = train.MODELS_DIR
OUTPUTS_DIR = train.OUTPUTS_DIR
CONFIG_DIR = train.CONFIG_DIR
LABELS_PATH = train.LABELS_PATH

ALPHAS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
ALPHA_COL = {a: f"q{int(round(a * 100)):02d}" for a in ALPHAS}  # 0.10 -> "q10", ...
EARLY_STOPPING_ROUNDS = 30

# Columns pulled off labels.parquet beyond what src.models.train.load_modeling_frame
# already joins -- games (the pool gate) and ppr_ppg (the regression target) --
# neither of which the classifier's LABEL_JOIN_COLS carries.
_LABEL_COLS = ["games", "ppr_ppg", "expectation_pos_rank", "adp_source", "in_training_pool"]


@dataclass(frozen=True)
class QuantileSpec:
    position: str
    label_position: str
    config_path: Path
    features_path: Path
    labels_path: Path
    report_path: Path
    metrics_json_path: Path
    artifact_path: Path


def quantile_spec(position: str) -> QuantileSpec:
    base = train.position_spec(position)
    return QuantileSpec(
        position=base.position,
        label_position=base.label_position,
        config_path=base.config_path,
        features_path=base.features_path,
        labels_path=base.labels_path,
        report_path=OUTPUTS_DIR / f"model_{base.position}_quantile_report.md",
        metrics_json_path=OUTPUTS_DIR / f"model_{base.position}_quantile_metrics.json",
        artifact_path=MODELS_DIR / f"{base.position}_quantile_bundle.joblib",
    )


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_modeling_frame(features_path: Path, labels_path: Path, cfg: dict, min_games: int = 8) -> pd.DataFrame:
    """features_{pos}.parquet joined to labels.parquet, restricted to in_training_pool==1

    AND games>=min_games -- see module docstring's "Modeling universe" section. Structurally
    parallel to ``src.models.train.load_modeling_frame`` (same train_season_start /
    adp_source-dummy handling) but selects the two extra label columns (games, ppr_ppg)
    that module never needs.
    """
    feats = pl.read_parquet(features_path)
    labels = pl.read_parquet(labels_path).select(["season", "gsis_id"] + _LABEL_COLS)
    joined = feats.join(labels, on=["season", "gsis_id"], how="inner")
    joined = joined.filter(pl.col("in_training_pool") & (pl.col("games") >= min_games))
    df = joined.sort(["season", "gsis_id"]).to_pandas()

    train_season_start = cfg.get("train_season_start", train.TRAIN_START_SEASON)
    df = df[df["season"] >= train_season_start].reset_index(drop=True)

    categories = cfg["adp_source_categories"]
    baseline_cat = cfg["adp_source_baseline"]
    for cat in categories:
        if cat == baseline_cat:
            continue
        df[f"adp_source_{cat}"] = (df["adp_source"] == cat).astype(float)
    return df.reset_index(drop=True)


_NON_FEATURE_EXTRA = {"games", "ppr_ppg", "in_training_pool", "adp_source"}


def tree_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Every numeric column except identity/label/target columns -- identical exclusion

    rule to ``src.models.train.tree_feature_columns`` (including that module's
    ``excluded_features`` config gate, the Vegas-promotion rule) plus this module's own
    extra non-feature columns (the target ``ppr_ppg`` and the pool-gate ``games``).
    """
    excluded = set(cfg.get("excluded_features", []))
    non_feature = set(train.IDENTITY_COLS) | _NON_FEATURE_EXTRA
    return [c for c in df.columns if c not in non_feature and c not in excluded]


def fold_frames(df: pd.DataFrame, train_seasons: tuple[int, ...], val_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train.fold_frames(df, train_seasons, val_season)


# --------------------------------------------------------------------------
# Fit / tune
# --------------------------------------------------------------------------


def fit_lgbm_quantile(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], params: dict, alpha: float, seed: int):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbosity=-1,
        **params,
    )
    model.fit(
        train_df[feature_cols],
        train_df["ppr_ppg"],
        eval_set=[(val_df[feature_cols], val_df["ppr_ppg"])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def tune_quantile(
    df: pd.DataFrame, feature_cols: list[str], folds: list[Fold], cfg: dict, n_trials: int, seed: int, alpha: float
) -> optuna.Study:
    """Optuna TPE search at ``alpha`` (the tuning alpha, 0.50 in production), minimizing

    mean pooled-fold pinball loss -- reuses the classifier's ``search_space.lgbm`` bounds
    (``train._suggest_lgbm_params``, same private helper the classifier head's own
    ``tune_classifier`` calls) since it is the same LightGBM/feature-count regime, just a
    different objective.
    """
    space = cfg["search_space"]["lgbm"]

    def objective(trial: optuna.Trial) -> float:
        params = train._suggest_lgbm_params(trial, space)
        scores = []
        for fold in folds:
            tr_df, val_df = fold_frames(df, fold.train_seasons, fold.val_season)
            model = fit_lgbm_quantile(tr_df, val_df, feature_cols, params, alpha, seed)
            pred = model.predict(val_df[feature_cols])
            scores.append(mt.pinball_loss(val_df["ppr_ppg"], pred, alpha))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def alpha_oof(
    df: pd.DataFrame, feature_cols: list[str], folds: list[Fold], params: dict, alpha: float, seed: int
) -> tuple[pd.DataFrame, list[dict], list[int]]:
    """Refit ``alpha``'s model on each fold at the frozen (alpha=0.50-tuned) params;

    return pooled OOF predictions, per-fold pinball loss, and each fold's best_iteration
    (for the no-early-stopping holdout retrain -- same discipline as
    ``src.models.train.regressor_oof``).
    """
    rows, fold_metrics, best_iters = [], [], []
    for fold in folds:
        tr_df, val_df = fold_frames(df, fold.train_seasons, fold.val_season)
        model = fit_lgbm_quantile(tr_df, val_df, feature_cols, params, alpha, seed)
        pred = model.predict(val_df[feature_cols])
        fold_metrics.append({"val_season": fold.val_season, "pinball": mt.pinball_loss(val_df["ppr_ppg"], pred, alpha), "n": int(len(val_df))})
        best_iters.append(train._model_best_iteration(model, "lgbm", params.get("n_estimators", 100)))
        out = val_df[["season", "gsis_id", "player_name", "ppr_ppg"]].copy()
        out["pred"] = pred
        rows.append(out)
    pooled = pd.concat(rows, ignore_index=True)
    return pooled, fold_metrics, best_iters


def enforce_noncrossing(wide: pd.DataFrame, alphas: tuple[float, ...] = ALPHAS) -> pd.DataFrame:
    """Sort the 5 per-row alpha predictions ascending so q10<=q25<=...<=q90 always holds.

    Nothing about 5 independently-fit quantile regressors guarantees this row-by-row; sorting
    is the brief's prescribed fix and is exact (no clipping/averaging that would distort any
    individual alpha's marginal calibration -- only within-row ORDER changes, and only for the
    (rare, verified in practice) rows where it was violated to begin with).
    """
    cols = [ALPHA_COL[a] for a in sorted(alphas)]
    wide = wide.copy()
    arr = np.sort(wide[cols].to_numpy(dtype=float), axis=1)
    for i, c in enumerate(cols):
        wide[c] = arr[:, i]
    return wide


def _pool_alphas(per_alpha_oof: dict[float, pd.DataFrame]) -> pd.DataFrame:
    """{alpha: pooled OOF frame} -> one wide frame (season, gsis_id, player_name, ppr_ppg,

    q10..q90), non-crossing enforced.
    """
    alphas = sorted(per_alpha_oof)
    base = per_alpha_oof[alphas[0]][["season", "gsis_id", "player_name", "ppr_ppg"]].copy()
    base[ALPHA_COL[alphas[0]]] = per_alpha_oof[alphas[0]]["pred"].to_numpy()
    for a in alphas[1:]:
        merged = per_alpha_oof[a][["season", "gsis_id", "pred"]].rename(columns={"pred": ALPHA_COL[a]})
        base = base.merge(merged, on=["season", "gsis_id"], how="inner")
    assert len(base) == len(per_alpha_oof[alphas[0]]), "pooled OOF rows misaligned across alphas"
    return enforce_noncrossing(base, tuple(alphas))


def distribution_metrics(wide: pd.DataFrame, alphas: tuple[float, ...] = ALPHAS) -> dict:
    """Pinball loss per alpha (on that alpha's own column vs actual), 80% (q10-q90) interval

    coverage, and Spearman(q50, actual) -- the three metrics the brief wants reported for
    both validation (pooled OOF) and holdout.
    """
    out = {"pinball_by_alpha": {}, "n": int(len(wide))}
    for a in alphas:
        out["pinball_by_alpha"][f"{a:.2f}"] = mt.pinball_loss(wide["ppr_ppg"], wide[ALPHA_COL[a]], a)
    out["coverage_q10_q90"] = mt.coverage(wide["ppr_ppg"], wide[ALPHA_COL[0.10]], wide[ALPHA_COL[0.90]])
    out["coverage_target"] = 0.80
    out["spearman_q50_actual"] = mt.spearman(wide["ppr_ppg"], wide[ALPHA_COL[0.50]])
    return out


# --------------------------------------------------------------------------
# Holdout
# --------------------------------------------------------------------------


def _fit_holdout_alpha(train_df, tree_cols, params, n_estimators, alpha, seed):
    p = dict(params)
    p["n_estimators"] = n_estimators
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, random_state=seed, deterministic=True, force_row_wise=True, n_jobs=1, verbosity=-1, **p
    )
    model.fit(train_df[tree_cols], train_df["ppr_ppg"])
    return model


def holdout_predictions(df: pd.DataFrame, seed: int, frozen: dict) -> tuple[pd.DataFrame, dict[float, object]]:
    """Retrain every alpha's model on <=2023 (frozen params + that alpha's own frozen

    n_estimators), predict 2024+2025, non-crossing enforced. Returns (wide predictions
    frame, {alpha: fitted model}) -- the fitted models are what ``save_artifacts`` ships.
    """
    split = holdout_split(train_start_season=frozen.get("train_season_start", train.TRAIN_START_SEASON))
    train_df = df[df["season"].isin(split.train_seasons)]
    holdout_df = df[df["season"].isin(split.holdout_seasons)]
    tree_cols = frozen["tree_feature_cols"]

    models: dict[float, object] = {}
    base = holdout_df[["season", "gsis_id", "player_name", "ppr_ppg"]].copy()
    for a in ALPHAS:
        model = _fit_holdout_alpha(train_df, tree_cols, frozen["params"], frozen["final_n_estimators"][a], a, seed)
        models[a] = model
        base[ALPHA_COL[a]] = model.predict(holdout_df[tree_cols])
    base = enforce_noncrossing(base)
    return base, models


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------


def run_full_pipeline(
    spec: QuantileSpec,
    cfg: dict | None = None,
    n_trials: int | None = None,
    seed: int | None = None,
    output_root: Path | None = None,
) -> dict:
    """Tune (alpha=0.50 only) -> OOF every alpha -> ONE holdout eval -> save bundle.

    ``output_root`` redirects every artifact this run writes, exactly like
    ``src.models.train.run_full_pipeline`` -- any non-production run (tests above all)
    MUST pass it; omitting it clobbers the shipped bundle with a tiny-trial run.
    """
    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        spec = dataclasses.replace(
            spec,
            report_path=output_root / spec.report_path.name,
            metrics_json_path=output_root / spec.metrics_json_path.name,
            artifact_path=output_root / spec.artifact_path.name,
        )
    if cfg is None:
        cfg = train.load_config(spec.config_path)
    qcfg = cfg["quantile"]
    seed = qcfg["seed"] if seed is None else seed
    n_trials = qcfg["optuna"]["trials"] if n_trials is None else n_trials
    tune_alpha = qcfg["tune_alpha"]

    train_season_start = cfg.get("train_season_start", train.TRAIN_START_SEASON)
    df = load_modeling_frame(spec.features_path, spec.labels_path, cfg=cfg)
    tree_cols = tree_feature_columns(df, cfg=cfg)
    folds = validation_folds(train_start_season=train_season_start)

    study = tune_quantile(df, tree_cols, folds, cfg, n_trials, seed, tune_alpha)
    params = study.best_params

    per_alpha_oof: dict[float, pd.DataFrame] = {}
    per_alpha_fold_metrics: dict[float, list[dict]] = {}
    final_n_estimators: dict[float, int] = {}
    for a in ALPHAS:
        pooled, fold_metrics, best_iters = alpha_oof(df, tree_cols, folds, params, a, seed)
        per_alpha_oof[a] = pooled
        per_alpha_fold_metrics[a] = fold_metrics
        final_n_estimators[a] = int(round(np.mean(best_iters)))

    val_wide = _pool_alphas(per_alpha_oof)
    val_metrics = distribution_metrics(val_wide)

    frozen = {
        "position": spec.position,
        "tree_feature_cols": tree_cols,
        "train_season_start": train_season_start,
        "params": params,
        "final_n_estimators": final_n_estimators,
        "tune_alpha": tune_alpha,
        "alphas": ALPHAS,
    }

    holdout_wide, holdout_models = holdout_predictions(df, seed, frozen)
    holdout_metrics = distribution_metrics(holdout_wide)

    save_artifacts(spec, frozen, holdout_models, seed, cfg)
    write_report(spec, val_metrics, holdout_metrics, per_alpha_fold_metrics, params, seed, n_trials)
    write_metrics_json(spec, val_metrics, holdout_metrics, params, seed)

    return {
        "spec": spec,
        "cfg": cfg,
        "seed": seed,
        "n_trials": n_trials,
        "df": df,
        "tree_cols": tree_cols,
        "folds": folds,
        "study": study,
        "params": params,
        "per_alpha_oof": per_alpha_oof,
        "per_alpha_fold_metrics": per_alpha_fold_metrics,
        "final_n_estimators": final_n_estimators,
        "val_wide": val_wide,
        "val_metrics": val_metrics,
        "holdout_wide": holdout_wide,
        "holdout_metrics": holdout_metrics,
        "frozen": frozen,
    }


def save_artifacts(spec: QuantileSpec, frozen: dict, holdout_models: dict[float, object], seed: int, cfg: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(frozen)
    payload["seed"] = seed
    payload["cfg"] = cfg
    payload["models"] = holdout_models  # {alpha: fitted LGBMRegressor}, <=2023-trained
    joblib.dump(payload, spec.artifact_path)


# --------------------------------------------------------------------------
# Scoring an arbitrary (e.g. 2026) feature frame with a saved bundle
# --------------------------------------------------------------------------


def score_quantiles(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """{q10..q90} predictions for every row of ``df`` (must carry every column in

    ``bundle["tree_feature_cols"]``), non-crossing enforced. ``df`` is returned with the
    5 quantile columns attached (original columns preserved).
    """
    tree_cols = bundle["tree_feature_cols"]
    out = df.copy()
    for a in ALPHAS:
        out[ALPHA_COL[a]] = bundle["models"][a].predict(out[tree_cols])
    wide = enforce_noncrossing(out, ALPHAS)
    return wide


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(
    spec: QuantileSpec, val_metrics: dict, holdout_metrics: dict, per_alpha_fold_metrics: dict, params: dict, seed: int, n_trials: int
) -> None:
    lines = [f"# {spec.position.upper()} quantile regression report (v2.0)", ""]
    lines.append(f"Tuned once at alpha=0.50 ({n_trials} Optuna trials, seed={seed}), reused for all 5 alphas.")
    lines.append("")
    lines.append("## Frozen hyperparameters (tuned at alpha=0.50)")
    lines.append("")
    lines.append(_md_table(["param", "value"], [[k, v] for k, v in params.items()]))
    lines.append("")
    lines.append("## Validation (pooled OOF, 4 expanding-window folds)")
    lines.append("")
    lines.append(_pinball_table(val_metrics))
    lines.append("")
    lines.append(f"- 80% (q10-q90) interval coverage: **{val_metrics['coverage_q10_q90']:.3f}** (target 0.80)")
    lines.append(f"- Spearman(q50, actual ppg): **{val_metrics['spearman_q50_actual']:.3f}**")
    lines.append(f"- n = {val_metrics['n']}")
    lines.append("")
    lines.append("## Holdout (2024+2025, retrain <=2023, ONE evaluation)")
    lines.append("")
    lines.append(_pinball_table(holdout_metrics))
    lines.append("")
    lines.append(f"- 80% (q10-q90) interval coverage: **{holdout_metrics['coverage_q10_q90']:.3f}** (target 0.80)")
    lines.append(f"- Spearman(q50, actual ppg): **{holdout_metrics['spearman_q50_actual']:.3f}**")
    lines.append(f"- n = {holdout_metrics['n']}")
    lines.append("")
    spec.report_path.parent.mkdir(parents=True, exist_ok=True)
    spec.report_path.write_text("\n".join(lines))


def _pinball_table(metrics: dict) -> str:
    rows = [[a, metrics["pinball_by_alpha"][a]] for a in sorted(metrics["pinball_by_alpha"])]
    return _md_table(["alpha", "pinball loss"], rows)


def write_metrics_json(spec: QuantileSpec, val_metrics: dict, holdout_metrics: dict, params: dict, seed: int) -> None:
    payload = {"position": spec.position, "seed": seed, "params": params, "validation_metrics": val_metrics, "holdout_metrics": holdout_metrics}
    spec.metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    spec.metrics_json_path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> int:
    import sys

    positions = sys.argv[1:] or list(train.POSITIONS)
    for pos in positions:
        spec = quantile_spec(pos)
        cfg = train.load_config(spec.config_path)
        print(f"== {pos} quantile pipeline ==")
        result = run_full_pipeline(spec, cfg=cfg)
        print(f"  validation coverage(q10-q90)={result['val_metrics']['coverage_q10_q90']:.3f} spearman={result['val_metrics']['spearman_q50_actual']:.3f}")
        print(f"  holdout    coverage(q10-q90)={result['holdout_metrics']['coverage_q10_q90']:.3f} spearman={result['holdout_metrics']['spearman_q50_actual']:.3f}")
        print(f"  wrote {spec.artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
