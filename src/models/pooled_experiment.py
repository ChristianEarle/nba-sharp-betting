"""v2.1 Deliverable 2: pooled-position model experiment.

Fit budget context (see the v2.1 brief): every per-position classifier trains
on 27-48 positives with ~50 candidate features -- at or over capacity for a
tree ensemble to learn stable splits from. This module asks whether pooling
all four positions into ONE classifier (~166 positives total, position
captured as a one-hot feature) generalizes better than the four separate
models, and whether restricting that pooled model to its own top-15
SHAP-important features helps further.

Two arms, both trained with the identical CV folds and v1.7 seed-ensemble
config (``configs/model_wr.yaml``'s ``optuna`` block: 60 classifier trials
per model type, seeds ``[42, 1337, 2024]``) the four per-position classifiers
already use -- reusing ``src.models.train``'s tuning/blend/calibration
machinery verbatim (every function there is already position-agnostic, it
just takes a DataFrame + a feature-column list):

1. **pooled** -- every position's rows concatenated, position captured as
   four ``position_{POS}`` one-hot columns, tree features = the UNION of
   every position's own ``tree_feature_columns`` (a feature that doesn't
   exist for a given position's rows -- e.g. QB has no ``target_share_n1``
   -- is simply null there; LightGBM/XGBoost handle nulls natively, no
   different from how a single position's own model already tolerates
   nullable columns like ``snap_share_n1``).
2. **pruned pooled** -- an entirely independent retrain (its own Optuna
   search, not just a column slice of arm 1's fitted models) restricted to
   the top-15 features by mean(|SHAP|) on the POOLED VALIDATION rows only
   (never holdout) -- computed once, off the base-seed LGBM refit on each
   of the four validation folds (see ``pooled_validation_shap_importance``).

Only the CLASSIFIER head is in scope here (per the brief) -- no regression
head, no quantile head; this experiment answers one question (does pooling
beat four separate per-position classifiers on holdout top-10 precision),
not a full second production pipeline.

GATE (pre-stated, binding, evaluated once holdout is unblinded)
-------------------------------------------------------------------------
Per position: pick whichever arm (pooled or pruned-pooled) has the BETTER
validation-fold PR-AUC restricted to that position's own OOF validation
rows -- chosen from validation alone, before holdout is ever touched.  That
arm's position replaces the incumbent ranking engine (the v1.7 classifier at
WR/RB/QB, the v2.0 quantile head at TE -- see ``outputs/comparison_gate.json``)
iff its holdout top-10 precision, measured on the IDENTICAL population and
metric the incumbent's own recorded number uses (all ``in_training_pool``
holdout rows for a classifier incumbent; ``breakout_eligible`` holdout rows
ranked by raw p_startable for the TE quantile incumbent), is STRICTLY GREATER
than the incumbent's. Ties keep the incumbent (simpler system wins ties;
pooled must win outright to displace it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.labels.build import load_labels_config
from src.models import metrics as mt
from src.models import train
from src.models.cv import Fold, validation_folds

REPO_ROOT = train.REPO_ROOT
PROCESSED_DIR = train.PROCESSED_DIR
OUTPUTS_DIR = train.OUTPUTS_DIR

POSITIONS = train.POSITIONS  # ("wr", "rb", "te", "qb")
POSITION_ONEHOT_COLS = [f"position_{p}" for p in POSITIONS]
POSITION_LOGIT_BASELINE = "wr"  # dropped for the logistic head only, same convention as adp_source

PRUNED_TOP_N = 15

POOLED_REPORT_MD_PATH = OUTPUTS_DIR / "pooled_experiment.md"
POOLED_METRICS_JSON_PATH = OUTPUTS_DIR / "pooled_experiment_metrics.json"
COMPARISON_GATE_JSON_PATH = OUTPUTS_DIR / "comparison_gate.json"


# --------------------------------------------------------------------------
# 1. Pooled frame + union feature columns
# --------------------------------------------------------------------------


def load_pooled_frame() -> tuple[pd.DataFrame, list[str], list[str], dict[str, dict]]:
    """Concat every position's ``in_training_pool==1`` modeling frame, one-hot position,

    union tree/logistic feature columns. Returns (pooled_df, tree_cols, logit_cols,
    cfg_by_pos) -- ``cfg_by_pos`` is each position's own loaded ``configs/model_{pos}.yaml``
    (needed downstream for ``train_season_start`` / label-eligibility thresholds).
    """
    frames = []
    cfg_by_pos: dict[str, dict] = {}
    tree_union: list[str] = []
    logit_union: list[str] = []
    for pos in POSITIONS:
        spec = train.position_spec(pos)
        cfg = train.load_config(spec.config_path)
        cfg_by_pos[pos] = cfg
        # Pool the FULL 2014-2025 history for every position, deliberately NOT honoring
        # RB's own per-position ``train_season_start: 2020`` proxy-era restriction (v1.7
        # Step 1) -- the brief's own "~166 positives total / 27-48 per position" baseline
        # is the unrestricted count (RB alone is 44 positives unrestricted vs far fewer
        # once capped to 2020+), and a pooled model with 3x the per-position sample size
        # is exactly the scenario where the extra pre-2020 RB rows are worth including
        # rather than re-litigating RB's own sensitivity check here.
        pool_cfg = dict(cfg)
        pool_cfg["train_season_start"] = train.TRAIN_START_SEASON
        df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=pool_cfg)
        # Compute this position's own tree/logistic feature columns BEFORE adding the
        # src_position/one-hot bookkeeping columns below -- train.tree_feature_columns
        # includes every column not in NON_FEATURE_COLS with no dtype check of its own
        # (safe on a plain load_modeling_frame output, where that invariant already
        # holds; src_position is a non-numeric column LightGBM would reject).
        for c in train.tree_feature_columns(df, cfg=cfg):
            if c not in tree_union:
                tree_union.append(c)
        for c in train.logistic_feature_columns(cfg):
            if c not in logit_union:
                logit_union.append(c)
        df["src_position"] = pos.upper()
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True, sort=False)
    for p in POSITIONS:
        pooled[f"position_{p}"] = (pooled["src_position"] == p.upper()).astype(float)

    tree_cols = list(tree_union) + POSITION_ONEHOT_COLS
    logit_position_cols = [c for c in POSITION_ONEHOT_COLS if c != f"position_{POSITION_LOGIT_BASELINE}"]
    logit_cols = list(dict.fromkeys(logit_union)) + logit_position_cols

    dupes = pooled.groupby(["season", "gsis_id"]).size()
    assert (dupes <= 1).all(), "pooled frame has a (season, gsis_id) row shared across positions -- unexpected"

    return pooled, tree_cols, logit_cols, cfg_by_pos


# --------------------------------------------------------------------------
# 2. Pooled-validation SHAP importance (for the pruned arm's feature selection)
# --------------------------------------------------------------------------


def pooled_validation_shap_importance(
    df: pd.DataFrame, tree_cols: list[str], folds: list[Fold], lgbm_params: dict, seed: int
) -> pd.DataFrame:
    """Mean(|SHAP|) per tree feature, pooled across every validation fold's OWN val rows --

    each fold gets its own LGBM refit at the frozen ``lgbm_params`` (the identical fold
    fit ``train.classifier_oof`` performs), SHAP-explained on ONLY that fold's val split,
    then every fold's (row, |shap|) pairs are stacked before averaging -- selection is
    validation-only by construction; holdout rows never enter this function.
    """
    all_abs = []
    for fold in folds:
        train_df, val_df = train.fold_frames(df, fold.train_seasons, fold.val_season)
        model = train.fit_lgbm_classifier(train_df, val_df, tree_cols, lgbm_params, seed)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(val_df[tree_cols])
        if isinstance(sv, list):
            sv = sv[-1]
        all_abs.append(np.abs(np.asarray(sv)))
    stacked = np.concatenate(all_abs, axis=0)
    mean_abs = stacked.mean(axis=0)
    return pd.DataFrame({"feature": tree_cols, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)


def select_pruned_features(
    df: pd.DataFrame, tree_cols: list[str], folds: list[Fold], lgbm_params: dict, seed: int, top_n: int = PRUNED_TOP_N
) -> list[str]:
    importance = pooled_validation_shap_importance(df, tree_cols, folds, lgbm_params, seed)
    return importance.head(top_n)["feature"].tolist()


# --------------------------------------------------------------------------
# 3. One arm's classifier-only ensemble (reuses src.models.train verbatim)
# --------------------------------------------------------------------------


def fit_pooled_classifier_arm(
    df: pd.DataFrame,
    tree_cols: list[str],
    logit_cols: list[str],
    cfg: dict,
    folds: list[Fold],
    n_trials: int,
    seeds: list[int],
    base_seed: int,
) -> dict:
    """Tune -> blend -> Platt-calibrate one classifier ensemble on `df`/`tree_cols`, exactly

    ``train.run_full_pipeline``'s classifier-only portion (no regression head -- out of this
    experiment's scope per the brief). Returns a dict carrying the OOF validation ensemble
    frame (with src_position attached), the frozen dict ``train.holdout_predictions`` needs,
    and the base seed's own lgbm_params (for SHAP-pruning the "pooled" arm's own top-15).
    """
    logit_C, _, _ = train.tune_logistic(df, logit_cols, folds, cfg, base_seed)
    logit_oof, _ = train.logistic_oof(df, logit_cols, folds, logit_C, base_seed)

    per_seed = [train.tune_and_oof_classifier_head(df, tree_cols, folds, cfg, n_trials, s) for s in seeds]
    per_seed, ensemble_pooled = train.ensemble_blend_and_pool(per_seed, logit_oof, cfg["blend"]["grid_step"])

    active_calibrator = train.fit_platt(ensemble_pooled["breakout"], ensemble_pooled["pred_blend"])
    ensemble_pooled["pred_calibrated"] = train.apply_calibration("platt", active_calibrator, ensemble_pooled["pred_blend"])

    # Attach src_position back for per-position validation metrics -- pooled (season,
    # gsis_id) is a valid join key (see load_pooled_frame's dupe assertion).
    pos_lookup = df[["season", "gsis_id", "src_position", "expectation_pos_rank"]].drop_duplicates()
    ensemble_pooled = ensemble_pooled.merge(pos_lookup, on=["season", "gsis_id"], how="left")

    base_ps = next(ps for ps in per_seed if ps["seed"] == base_seed)
    frozen = {
        "tree_feature_cols": tree_cols,
        "logistic_feature_cols": logit_cols,
        "train_season_start": cfg.get("train_season_start", train.TRAIN_START_SEASON),
        "logistic_C": logit_C,
        "per_seed": {
            ps["seed"]: {
                "lgbm_params": ps["lgbm_params"],
                "xgb_params": ps["xgb_params"],
                "lgbm_final_n_estimators": ps["lgbm_final_n_estimators"],
                "xgb_final_n_estimators": ps["xgb_final_n_estimators"],
                "blend_weights": ps["blend_weights"],
            }
            for ps in per_seed
        },
        "active_calibration_method": "platt",
        "active_calibrator": active_calibrator,
        # train.holdout_predictions' fallback branch does `frozen.get("active_...",
        # frozen["calibration_method"])` -- dict.get's default arg is evaluated eagerly
        # even when "active_calibration_method" IS present, so these legacy keys must
        # exist too even though they're never actually used as the active calibrator.
        "calibration_method": "platt",
        "calibrator": active_calibrator,
    }
    return {
        "ensemble_val_oof": ensemble_pooled,
        "frozen": frozen,
        "base_lgbm_params": base_ps["lgbm_params"],
        "per_seed": per_seed,
    }


# --------------------------------------------------------------------------
# 4. Metrics: validation PR-AUC / holdout top-10 precision, per position
# --------------------------------------------------------------------------


def validation_pr_auc_by_position(ensemble_val_oof: pd.DataFrame) -> dict[str, float]:
    out = {}
    for pos in POSITIONS:
        sub = ensemble_val_oof[ensemble_val_oof["src_position"] == pos.upper()]
        out[pos.upper()] = mt.pr_auc(sub["breakout"], sub["pred_calibrated"])
    return out


def holdout_top10_by_position(
    holdout_scored: pd.DataFrame, labels_cfg: dict, engine_decision: dict[str, str]
) -> dict[str, dict]:
    """Per position: top-10 precision on the SAME population/metric the incumbent's own

    recorded number represents -- all holdout rows for a classifier incumbent (matches
    ``train.evaluate_scores``' pooled "model" row), ``breakout_eligible`` holdout rows
    ranked by ``pred_calibrated`` for the TE quantile incumbent (matches
    ``src.inference.projections.comparison_gate``'s own p_startable methodology). Both
    variants are reported regardless, for transparency.
    """
    out = {}
    for pos in POSITIONS:
        label_pos = pos.upper()
        sub = holdout_scored[holdout_scored["src_position"] == label_pos].reset_index(drop=True)
        all_top10 = mt.top_k_precision(sub["breakout"], sub["pred_calibrated"], 10) if len(sub) else float("nan")

        adp_worse_than = float(labels_cfg["thresholds"][label_pos]["adp_worse_than"])
        eligible = sub[sub["expectation_pos_rank"] >= adp_worse_than]
        elig_top10 = (
            mt.top_k_precision(eligible["breakout"], eligible["pred_calibrated"], 10) if len(eligible) else float("nan")
        )

        engine = engine_decision.get(label_pos, "classifier")
        comparison_value = elig_top10 if engine == "quantile" else all_top10
        out[label_pos] = {
            "n_holdout": int(len(sub)),
            "n_pos_holdout": int(sub["breakout"].sum()) if len(sub) else 0,
            "top10_precision_all_holdout": all_top10,
            "n_eligible_holdout": int(len(eligible)),
            "top10_precision_eligible_holdout": elig_top10,
            "incumbent_engine": engine,
            "comparison_value": comparison_value,  # the value compared against the incumbent
        }
    return out


def load_incumbent_top10() -> dict[str, dict]:
    """{position: {"engine":..., "top10_precision":...}} straight off comparison_gate.json --

    the already-established Deliverable-3 (v2.0) verdict per position. Missing/absent ->
    classifier engine, NaN precision (nothing to beat, so pooled cannot promote there either).
    """
    if not COMPARISON_GATE_JSON_PATH.exists():
        return {}
    rows = json.loads(COMPARISON_GATE_JSON_PATH.read_text())
    out = {}
    for r in rows:
        engine = r["primary_engine"]
        value = r["quantile_top10_precision"] if engine == "quantile" else r["classifier_top10_precision"]
        out[r["position"]] = {"engine": engine, "top10_precision": value}
    return out


# --------------------------------------------------------------------------
# 5. Full experiment
# --------------------------------------------------------------------------


def run_experiment(
    classifier_trials: int | None = None,
    seeds: list[int] | None = None,
    base_seed: int | None = None,
    output_root: Path | None = None,
) -> dict:
    """Train both arms, score holdout once each, apply the per-position gate. `classifier_trials`

    / `seeds` / `base_seed` override the config (used by tests for a tiny fast run); leave
    None for the real Deliverable-2 run (60 trials, seeds [42, 1337, 2024], base seed 42 --
    read straight off configs/model_wr.yaml, the same config every per-position classifier
    already uses). ``output_root`` redirects the written report/metrics files into that
    directory instead of ``outputs/`` -- same non-clobbering discipline
    ``train.run_full_pipeline`` uses; any run that is not the real Deliverable-2 run (tests
    above all) MUST pass it.
    """
    pooled_df, tree_cols, logit_cols, cfg_by_pos = load_pooled_frame()
    wr_cfg = cfg_by_pos["wr"]  # v1.7 seed-ensemble config lives identically in every position's yaml
    n_trials = wr_cfg["optuna"]["classifier_trials"] if classifier_trials is None else classifier_trials
    run_seeds = list(dict.fromkeys(wr_cfg["optuna"]["seeds"])) if seeds is None else list(dict.fromkeys(seeds))
    seed = wr_cfg["seed"] if base_seed is None else base_seed
    if seed not in run_seeds:
        run_seeds = [seed] + run_seeds

    folds = validation_folds()
    labels_cfg = load_labels_config()

    # ---- arm 1: pooled (full union feature set) ----
    pooled_arm = fit_pooled_classifier_arm(pooled_df, tree_cols, logit_cols, wr_cfg, folds, n_trials, run_seeds, seed)

    # ---- arm 2: pruned pooled (top-15 features by pooled-validation SHAP, base seed) ----
    pruned_tree_cols = select_pruned_features(pooled_df, tree_cols, folds, pooled_arm["base_lgbm_params"], seed)
    pruned_arm = fit_pooled_classifier_arm(
        pooled_df, pruned_tree_cols, logit_cols, wr_cfg, folds, n_trials, run_seeds, seed
    )

    arms = {"pooled": pooled_arm, "pruned_pooled": pruned_arm}
    arm_feature_cols = {"pooled": tree_cols, "pruned_pooled": pruned_tree_cols}

    val_pr_auc = {name: validation_pr_auc_by_position(arm["ensemble_val_oof"]) for name, arm in arms.items()}

    # ---- ONE holdout scoring pass per arm ----
    holdout_scored = {}
    for name, arm in arms.items():
        scored = train.holdout_predictions(pooled_df, seed, arm["frozen"])
        # holdout_predictions' REPORT_ID_COLS already carries expectation_pos_rank --
        # only src_position needs joining back on here.
        pos_lookup = pooled_df[["season", "gsis_id", "src_position"]].drop_duplicates()
        scored = scored.merge(pos_lookup, on=["season", "gsis_id"], how="left")
        holdout_scored[name] = scored

    engine_decision_raw = load_incumbent_top10()
    engine_decision = {pos: info["engine"] for pos, info in engine_decision_raw.items()}
    holdout_top10 = {name: holdout_top10_by_position(df_, labels_cfg, engine_decision) for name, df_ in holdout_scored.items()}

    # ---- per-position arm selection (validation only, before holdout is used for anything
    # but reporting) + the gate decision ----
    decisions = []
    for pos in POSITIONS:
        label_pos = pos.upper()
        pooled_val = val_pr_auc["pooled"].get(label_pos, float("nan"))
        pruned_val = val_pr_auc["pruned_pooled"].get(label_pos, float("nan"))
        # Ties (or a NaN on one side) favor the simpler full-feature "pooled" arm.
        chosen_arm = "pruned_pooled" if (pruned_val > pooled_val) else "pooled"

        incumbent = engine_decision_raw.get(label_pos, {"engine": "classifier", "top10_precision": float("nan")})
        chosen_holdout = holdout_top10[chosen_arm][label_pos]
        pooled_value = chosen_holdout["comparison_value"]
        incumbent_value = incumbent["top10_precision"]

        promote = bool(
            not np.isnan(pooled_value) and not np.isnan(incumbent_value) and pooled_value > incumbent_value
        )
        decisions.append(
            {
                "position": label_pos,
                "chosen_arm": chosen_arm,
                "pooled_val_pr_auc": pooled_val,
                "pruned_val_pr_auc": pruned_val,
                "incumbent_engine": incumbent["engine"],
                "incumbent_top10_precision": incumbent_value,
                "pooled_holdout_top10_precision": pooled_value,
                "pooled_holdout_top10_all": chosen_holdout["top10_precision_all_holdout"],
                "pooled_holdout_top10_eligible": chosen_holdout["top10_precision_eligible_holdout"],
                "n_holdout": chosen_holdout["n_holdout"],
                "n_pos_holdout": chosen_holdout["n_pos_holdout"],
                "promote": promote,
            }
        )

    result = {
        "tree_cols_pooled": tree_cols,
        "tree_cols_pruned": pruned_tree_cols,
        "logit_cols": logit_cols,
        "val_pr_auc": val_pr_auc,
        "holdout_top10": holdout_top10,
        "decisions": decisions,
        "n_positives_total": int(pooled_df["breakout"].sum()),
        "n_rows_total": int(len(pooled_df)),
    }
    _write_outputs(result, output_root=output_root)
    return result


# --------------------------------------------------------------------------
# 6. Report writing
# --------------------------------------------------------------------------


def _write_outputs(result: dict, output_root: Path | None = None) -> None:
    out_dir = Path(output_root) if output_root is not None else OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / POOLED_REPORT_MD_PATH.name
    metrics_path = out_dir / POOLED_METRICS_JSON_PATH.name
    jsonable = {
        "n_positives_total": result["n_positives_total"],
        "n_rows_total": result["n_rows_total"],
        "tree_cols_pooled": result["tree_cols_pooled"],
        "tree_cols_pruned": result["tree_cols_pruned"],
        "val_pr_auc": result["val_pr_auc"],
        "holdout_top10": result["holdout_top10"],
        "decisions": result["decisions"],
    }
    metrics_path.write_text(json.dumps(jsonable, indent=2, default=str))

    lines = [
        "# v2.1 Deliverable 2 -- pooled-position model experiment",
        "",
        f"One classifier over all four positions ({result['n_positives_total']} positives, "
        f"{result['n_rows_total']} rows total). Two arms: `pooled` (union feature set + "
        "position one-hot) and `pruned_pooled` (top-15 features by pooled-validation SHAP "
        "importance). Same CV folds + v1.7 seed-ensemble config as the per-position "
        "classifiers. GATE (pre-stated, binding): per position, pick whichever arm wins on "
        "VALIDATION PR-AUC (chosen before holdout); promote iff that arm's holdout top-10 "
        "precision (same population/metric as the incumbent's own recorded number) is "
        "STRICTLY GREATER than the incumbent's. Ties keep the incumbent.",
        "",
        "## Validation PR-AUC by position (OOF, both arms)",
        "",
        "| Position | Pooled (full) | Pruned pooled (top-15) | Chosen arm |",
        "| --- | --- | --- | --- |",
    ]
    for d in result["decisions"]:
        lines.append(
            f"| {d['position']} | {d['pooled_val_pr_auc']:.3f} | {d['pruned_val_pr_auc']:.3f} | {d['chosen_arm']} |"
        )
    lines += [
        "",
        "## Holdout top-10 precision vs incumbent (ONE holdout pass per arm)",
        "",
        "| Position | Incumbent engine | Incumbent top-10 | Pooled top-10 (all holdout) | "
        "Pooled top-10 (eligible) | Comparison value | n holdout (n pos) | Promote? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d in result["decisions"]:
        lines.append(
            f"| {d['position']} | {d['incumbent_engine']} | {d['incumbent_top10_precision']:.3f} | "
            f"{d['pooled_holdout_top10_all']:.3f} | {d['pooled_holdout_top10_eligible']:.3f} | "
            f"{d['pooled_holdout_top10_precision']:.3f} | {d['n_holdout']} ({d['n_pos_holdout']}) | "
            f"{'**YES**' if d['promote'] else 'no'} |"
        )
    lines += [
        "",
        "## Feature sets",
        "",
        f"Pooled (union, {len(result['tree_cols_pooled'])} tree features): "
        + ", ".join(f"`{c}`" for c in result["tree_cols_pooled"]),
        "",
        f"Pruned pooled (top-{len(result['tree_cols_pruned'])} by pooled-validation SHAP): "
        + ", ".join(f"`{c}`" for c in result["tree_cols_pruned"]),
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    print("v2.1 Deliverable 2: pooled-position model experiment (real Optuna config -- this takes a while)")
    result = run_experiment()
    print(f"wrote {POOLED_REPORT_MD_PATH}")
    print(f"wrote {POOLED_METRICS_JSON_PATH}")
    for d in result["decisions"]:
        print(
            f"  {d['position']}: chosen_arm={d['chosen_arm']} incumbent={d['incumbent_engine']}"
            f"({d['incumbent_top10_precision']:.3f}) pooled={d['pooled_holdout_top10_precision']:.3f} "
            f"promote={d['promote']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
