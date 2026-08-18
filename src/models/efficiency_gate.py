"""v2.2 -- efficiency-proxy capacity gate (Part 2 of this session's brief).

True yards-per-route-run (YPRR) is impossible to compute for this pipeline: route-
participation data is dead after 2023 (see the README's Known Limitations). This module
gates three inference-safe PROXIES for it into (or out of) each position's shipped model,
mirroring ``src.models.vegas_experiment``'s promotion protocol almost exactly -- same
frozen-hyperparameter discipline, same "score both variants on the identical holdout rows"
structure, same v1.7 seed-ensemble-aware promotion (Finding 5's fix: the ACTIVE Platt
calibrator and every seed's own n_estimators are refit against the new score distribution,
never left stale).

The three candidate columns (``src.features.shared``, wired into
``src.features.{wr,rb,te}`` -- see each module's ``BASE_METRICS`` v2.2 addition)
------------------------------------------------------------------------------------
- ``yards_per_snap`` (WR/TE: receiving yards per offensive snap; RB: SCRIMMAGE
  (rushing + receiving) yards per offensive snap) -- from ``snap_counts.parquet``,
  nullable wherever the snap-counts pfr_id crosswalk doesn't resolve.
- ``targets_per_snap`` (WR/TE only -- an RB's touches aren't route-run-shaped the same
  way) -- same snap-counts source.
- ``catchable_target_rate`` (WR/RB/TE -- share of a player's targeted pass plays FTN
  charted as a catchable ball) -- 2022+ only (FTN's charting-coverage start), nullable
  before that and wherever FTN has no coverage for that game.
Each ships as its usual ``_n1``/``_yoy_delta`` pair (``src.features.wr.BASE_METRICS``
machinery), so "the new columns" below always means both suffixed forms together.

Protocol (frozen hyperparameters throughout -- no new Optuna anywhere in this module)
----------------------------------------------------------------------------------------
1. **Incumbent** score: the position's currently shipped bundle, unchanged -- its own
   holdout-retrained classifier trio (``bundle["per_seed_models"]``/``bundle["logistic"]``),
   blended at each seed's frozen weights and calibrated through
   ``bundle["active_calibrator"]``. Exactly ``src.inference.board_2026.score_veterans_batch``'s
   own math, just recomputed on the holdout rows for a side-by-side.
2. **+efficiency variant**: LGBM+XGB refit per v1.7 ensemble seed at that seed's FROZEN
   ``lgbm_params``/``xgb_params`` (``bundle["per_seed"]``), on <=2023, with
   ``tree_feature_cols + this position's new columns``. The shared logistic head is
   recomputed once (unaffected by tree-only columns, but its OOF still needs the new
   fold structure -- computed fresh rather than assumed identical to keep this module
   self-contained). Each seed's own frozen blend weights combine its own new-column LGBM/XGB
   OOF with that one shared logistic OOF; the ensemble raw score is the mean across seeds
   (identical construction to ``train.run_full_pipeline`` / ``vegas_experiment.promote_position``).
3. Metrics (top-10 precision, PR-AUC) computed on the RAW pre-calibration blended ensemble
   score for BOTH variants, pooled across the 2024+2025 holdout -- apples-to-apples, same
   convention ``vegas_experiment`` uses and for the same reason (calibration is only
   rank-preserving for a strictly monotone calibrator; Platt is, but comparing on the raw
   score sidesteps needing to prove that here too).

Gate rule (binding, from the brief)
--------------------------------------
KEEP a position's new columns iff holdout top-10 precision does NOT regress (variant >=
incumbent) AND PR-AUC either improves OR regresses by at most 0.01
(``variant_pr_auc >= incumbent_pr_auc - 0.01``). Applied mechanically, never re-litigated
after seeing the numbers -- the same "pre-stated, binding" discipline every other
experiment script in this repo (``proxy_sensitivity``, ``vegas_experiment``,
``pooled_experiment``) follows.

Disposition
-------------
- KEPT: the frozen-retrain-with-new-columns bundle becomes the position's shipped bundle
  (``promote_position`` below, closely mirroring ``vegas_experiment.promote_position``'s
  corrected v1.7-ensemble-aware refit -- see that function's docstring for why every seed's
  n_estimators AND the active calibrator must be refit, not just the legacy single-seed ones).
- NOT KEPT: the new columns stay in ``features_{pos}.parquet`` (nullable, auditable,
  available to a future experiment) but are added to ``configs/model_{pos}.yaml``'s
  ``excluded_features`` list, so a future ``python -m src.models.train {pos}`` real Optuna
  retrain doesn't silently start using them -- same convention
  ``vegas_experiment``/rejected positions established for the Vegas columns.
"""

from __future__ import annotations

import json
import re

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models import metrics as mt
from src.models import train

OUT_PATH = train.OUTPUTS_DIR / "efficiency_gate.md"
HOLDOUT_SEASONS = train.HOLDOUT_SEASONS  # (2024, 2025)

# Per-position new base metrics (v2.2) -- see src.features.{wr,rb,te}.BASE_METRICS'
# own v2.2 comment block. Each expands to both its "_n1" and "_yoy_delta" columns.
NEW_BASE_METRICS: dict[str, tuple[str, ...]] = {
    "wr": ("yards_per_snap", "targets_per_snap", "catchable_target_rate"),
    "te": ("yards_per_snap", "targets_per_snap", "catchable_target_rate"),
    "rb": ("yards_per_snap", "catchable_target_rate"),
}

PR_AUC_TOLERANCE = 0.01


def new_columns_for(pos: str) -> list[str]:
    metrics = NEW_BASE_METRICS.get(pos, ())
    return [f"{m}_n1" for m in metrics] + [f"{m}_yoy_delta" for m in metrics]


# --------------------------------------------------------------------------
# Score both variants on the holdout years (no retuning anywhere)
# --------------------------------------------------------------------------


def score_incumbent(bundle: dict, holdout_df: pd.DataFrame) -> np.ndarray:
    """The bundle's own v1.7 seed-ensemble raw score on ``holdout_df`` -- identical

    construction to ``src.inference.board_2026.score_veterans_batch``'s ``raw_score``
    (mean of every seed's own blended score), just recomputed here for a clean
    side-by-side against the +efficiency variant. Falls back to a synthetic single-seed
    ensemble for a pre-v1.7 bundle, same backward-compatibility contract
    ``train.holdout_predictions`` documents.
    """
    tree_cols = bundle["tree_feature_cols"]
    per_seed_models = bundle.get("per_seed_models") or {bundle.get("seed", 0): {"lgbm": bundle["lgbm"], "xgb": bundle["xgb"]}}
    per_seed_weights = bundle.get("per_seed") or {bundle.get("seed", 0): {"blend_weights": bundle["blend_weights"]}}

    logit_pred = None
    if any(per_seed_weights[s].get("blend_weights", {}).get("logistic", 0) > 0 for s in per_seed_models):
        logit_model, logit_imputer, logit_scaler = bundle["logistic"]
        logit_pred = train._logistic_predict(logit_model, logit_imputer, logit_scaler, holdout_df, bundle["logistic_feature_cols"])

    seed_scores = []
    for seed, models in per_seed_models.items():
        weights = per_seed_weights[seed]["blend_weights"]
        preds = {}
        if weights.get("lgbm", 0) > 0:
            preds["lgbm"] = models["lgbm"].predict_proba(holdout_df[tree_cols])[:, 1]
        if weights.get("xgb", 0) > 0:
            preds["xgb"] = models["xgb"].predict_proba(holdout_df[tree_cols])[:, 1]
        if weights.get("logistic", 0) > 0:
            preds["logistic"] = logit_pred
        nonzero_weights = {k: w for k, w in weights.items() if k in preds}
        seed_scores.append(train.apply_blend(nonzero_weights, **preds))
    return np.mean(np.vstack(seed_scores), axis=0)


def _fit_and_score_at_tree_cols(
    bundle: dict, df: pd.DataFrame, tree_cols: list[str], folds: list, split
) -> tuple[np.ndarray, dict]:
    """Refit every v1.7 ensemble seed's LGBM/XGB at that seed's FROZEN hyperparameters on

    exactly ``tree_cols`` (<=2023 train, per ``split``), reusing that seed's own frozen
    blend weights against a freshly-computed shared logistic OOF-fit. The low-level
    engine both ``fit_and_score_efficiency_variant`` (tree_cols = bundle's own +
    extra_cols) and ``reconstruct_incumbent_score`` (tree_cols = bundle's own MINUS
    extra_cols, for re-deriving a genuine "before promotion" comparison once a bundle has
    already been promoted -- see that function's docstring) delegate to, so both use the
    identical frozen-hyperparameter refit machinery. Returns (holdout raw ensemble score,
    {"per_seed_models": ..., "logistic": ..., "new_per_seed": {...}, "tree_cols": ...}).
    """
    logit_cols = bundle["logistic_feature_cols"]
    per_seed_spec = bundle.get("per_seed") or {
        bundle.get("seed", 0): {"lgbm_params": bundle["lgbm_params"], "xgb_params": bundle["xgb_params"], "blend_weights": bundle["blend_weights"]}
    }

    train_df = df[df["season"].isin(split.train_seasons)]
    holdout_df = df[df["season"].isin(split.holdout_seasons)]
    spw = train._scale_pos_weight(train_df["breakout"])

    logit_model, logit_imputer, logit_scaler = train.fit_logistic(train_df, holdout_df, logit_cols, bundle["logistic_C"], bundle["seed"])
    logit_pred = train._logistic_predict(logit_model, logit_imputer, logit_scaler, holdout_df, logit_cols)

    per_seed_models: dict[int, dict] = {}
    new_per_seed: dict[int, dict] = {}
    seed_scores = []
    for seed, info in per_seed_spec.items():
        lgbm_oof, _, lgbm_best_iters = train.classifier_oof(df, tree_cols, folds, "lgbm", info["lgbm_params"], seed)
        xgb_oof, _, xgb_best_iters = train.classifier_oof(df, tree_cols, folds, "xgb", info["xgb_params"], seed)
        lgbm_n_est = int(round(np.mean(lgbm_best_iters)))
        xgb_n_est = int(round(np.mean(xgb_best_iters)))

        lgbm_model = train._fit_holdout_lgbm(train_df, tree_cols, info["lgbm_params"], lgbm_n_est, seed, spw)
        xgb_model = train._fit_holdout_xgb(train_df, tree_cols, info["xgb_params"], xgb_n_est, seed, spw)
        lgbm_pred = lgbm_model.predict_proba(holdout_df[tree_cols])[:, 1]
        xgb_pred = xgb_model.predict_proba(holdout_df[tree_cols])[:, 1]

        weights = info["blend_weights"]
        preds = {}
        if weights.get("lgbm", 0) > 0:
            preds["lgbm"] = lgbm_pred
        if weights.get("xgb", 0) > 0:
            preds["xgb"] = xgb_pred
        if weights.get("logistic", 0) > 0:
            preds["logistic"] = logit_pred
        nonzero_weights = {k: w for k, w in weights.items() if k in preds}
        seed_scores.append(train.apply_blend(nonzero_weights, **preds))

        per_seed_models[seed] = {"lgbm": lgbm_model, "xgb": xgb_model}
        new_per_seed[seed] = {
            "lgbm_params": info["lgbm_params"], "xgb_params": info["xgb_params"],
            "lgbm_final_n_estimators": lgbm_n_est, "xgb_final_n_estimators": xgb_n_est,
            "blend_weights": weights, "blend_pr_auc": info.get("blend_pr_auc"),
        }

    blended = np.mean(np.vstack(seed_scores), axis=0)
    fitted = {
        "per_seed_models": per_seed_models,
        "new_per_seed": new_per_seed,
        "logistic": (logit_model, logit_imputer, logit_scaler),
        "tree_cols": tree_cols,
    }
    return blended, fitted


def fit_and_score_efficiency_variant(
    bundle: dict, df: pd.DataFrame, extra_cols: list[str], folds: list, split
) -> tuple[np.ndarray, dict]:
    """Refit every v1.7 ensemble seed's LGBM/XGB at that seed's FROZEN hyperparameters on

    ``tree_feature_cols + extra_cols`` (<=2023 train, per the ``split``), reusing that
    seed's own frozen blend weights against a freshly-computed shared logistic OOF-fit
    (the logistic subset never gains ``extra_cols`` -- tree-only columns, same scope
    decision ``vegas_experiment`` made for the Vegas columns). Returns (holdout raw
    ensemble score, {"per_seed_models": ..., "logistic": ..., "extra_cols": extra_cols,
    "new_per_seed": {...}}) -- the fitted models/derived n_estimators a KEEP decision's
    ``promote_position`` reuses instead of refitting a third time.
    """
    # Idempotency guard: a bundle the gate already KEPT (a prior run) carries extra_cols
    # in its tree_feature_cols already -- appending them again would hand LightGBM a
    # duplicate-named feature column and crash ("Feature (...) appears more than one
    # time"). Only add columns not already present.
    tree_cols = list(bundle["tree_feature_cols"]) + [c for c in extra_cols if c not in bundle["tree_feature_cols"]]
    return _fit_and_score_at_tree_cols(bundle, df, tree_cols, folds, split)


def reconstruct_incumbent_score(bundle: dict, df: pd.DataFrame, extra_cols: list[str], folds: list, split) -> np.ndarray:
    """The TRUE pre-promotion incumbent's holdout score, reconstructed from a bundle that

    has ALREADY been promoted (its ``tree_feature_cols`` already carries ``extra_cols``).

    Re-running this module against an already-promoted bundle must never silently compare
    the promoted bundle against ITSELF (incumbent_score == variant_score, a degenerate
    same-vs-same "comparison" that trivially always KEEPs and reports a fabricated zero
    lift) -- that is exactly the artifact-staleness/self-consistency failure mode this
    module exists to prevent. Since every hyperparameter/blend-weight stays FROZEN across
    a promotion (only tree_feature_cols, n_estimators, and the calibrator change), the
    incumbent's exact holdout predictions are fully reproducible: refit at the SAME frozen
    per-seed hyperparameters over ``tree_feature_cols`` with ``extra_cols`` stripped back
    out -- deterministically identical to what the original (never-promoted) incumbent
    bundle would have produced.
    """
    reduced_tree_cols = [c for c in bundle["tree_feature_cols"] if c not in extra_cols]
    score, _ = _fit_and_score_at_tree_cols(bundle, df, reduced_tree_cols, folds, split)
    return score


def bundle_already_promoted(bundle: dict, extra_cols: list[str]) -> bool:
    """True iff every one of this position's v2.2 columns is already in the bundle's own

    tree_feature_cols -- i.e. a prior run of this module already KEPT/promoted it.
    """
    return bool(extra_cols) and all(c in bundle["tree_feature_cols"] for c in extra_cols)


# --------------------------------------------------------------------------
# Metrics table + gate decision
# --------------------------------------------------------------------------


def metrics_table(holdout_df: pd.DataFrame, incumbent_score: np.ndarray, variant_score: np.ndarray) -> pd.DataFrame:
    rows = []
    for season in list(HOLDOUT_SEASONS) + ["pooled"]:
        mask = (holdout_df["season"] == season) if season != "pooled" else pd.Series(True, index=holdout_df.index)
        y = holdout_df.loc[mask, "breakout"]
        for name, score in (("incumbent", incumbent_score), ("plus_efficiency", variant_score)):
            s = pd.Series(score, index=holdout_df.index).loc[mask]
            rows.append(
                {
                    "season": season, "variant": name, "n": int(mask.sum()), "n_pos": int(y.sum()),
                    "top10_precision": mt.top_k_precision(y, s, 10), "pr_auc": mt.pr_auc(y, s),
                }
            )
    return pd.DataFrame(rows)


def gate_decision(table: pd.DataFrame) -> tuple[bool, str]:
    """KEEP iff pooled holdout top-10 precision does not regress AND PR-AUC either

    improves or regresses by at most PR_AUC_TOLERANCE. Returns (keep: bool, reason: str).
    """
    pooled = table[table["season"] == "pooled"].set_index("variant")
    base_top10 = pooled.loc["incumbent", "top10_precision"]
    var_top10 = pooled.loc["plus_efficiency", "top10_precision"]
    base_prauc = pooled.loc["incumbent", "pr_auc"]
    var_prauc = pooled.loc["plus_efficiency", "pr_auc"]

    top10_ok = var_top10 >= base_top10
    prauc_ok = var_prauc >= base_prauc - PR_AUC_TOLERANCE
    keep = bool(top10_ok and prauc_ok)

    reason = (
        f"top-10 precision {base_top10:.3f} -> {var_top10:.3f} ({'ok, no regression' if top10_ok else 'REGRESSED -- reject'}); "
        f"PR-AUC {base_prauc:.3f} -> {var_prauc:.3f} ({'ok' if prauc_ok else f'REGRESSED by more than {PR_AUC_TOLERANCE} -- reject'})"
    )
    return keep, ("KEEP: " if keep else "REJECT: ") + reason


# --------------------------------------------------------------------------
# Promotion: regenerate a position's bundle with the new columns baked in
# (only ever called for a position the gate KEEPs)
# --------------------------------------------------------------------------


def promote_position(spec: train.PositionSpec, bundle: dict, df: pd.DataFrame, fitted: dict) -> dict:
    """Rebuild spec.artifact_path with this position's new efficiency columns included in

    the tree heads, at every incumbent-frozen hyperparameter/blend weight per seed -- NO
    new Optuna, matching this module's whole premise. Reuses the per-seed models/OOF
    n_estimators ``fit_and_score_efficiency_variant`` already computed (no third refit).

    v1.7 seed ensemble (mirrors vegas_experiment.promote_position's Finding-5-fixed
    pattern exactly): the ACTIVE Platt calibrator is refit against the new-columns
    ensemble's OOF score distribution -- leaving it stale (fit on the pre-promotion
    distribution) would silently miscalibrate every probability this bundle ships.

    Also assembles a ``run_full_pipeline``-shaped ``result`` dict (fold metrics for every
    head, the regression fold OOF, the freshly-calibrated pooled ensemble frame, etc.) so
    the SAME ``train.generate_report``/``train._metrics_to_jsonable`` machinery every real
    retrain uses can write ``outputs/model_{pos}_{report.md,metrics.json}`` for this
    promoted bundle too -- see ``write_gate_outputs`` below. Regenerating those two files
    for a KEPT position is not optional: leaving them describing the PRE-promotion bundle
    while a new bundle ships is exactly the artifact-staleness class this repo has been
    burned by before (the v1.7 reconciliation).
    """
    tree_cols = fitted["tree_cols"]
    logit_cols = bundle["logistic_feature_cols"]
    folds = train.validation_folds()
    seed = bundle["seed"]

    logit_oof, logit_fold_metrics = train.logistic_oof(df, logit_cols, folds, bundle["logistic_C"], seed)

    seed_blend_cols = []
    base = None
    base_lgbm_fold_metrics = base_xgb_fold_metrics = None
    for s, info in fitted["new_per_seed"].items():
        lgbm_oof_s, lgbm_fold_metrics_s, _ = train.classifier_oof(df, tree_cols, folds, "lgbm", info["lgbm_params"], s)
        xgb_oof_s, xgb_fold_metrics_s, _ = train.classifier_oof(df, tree_cols, folds, "xgb", info["xgb_params"], s)

        merged = lgbm_oof_s[["season", "gsis_id", "breakout"]].copy()
        merged["pred_lgbm"] = lgbm_oof_s["pred"].to_numpy()
        merged = merged.merge(xgb_oof_s[["season", "gsis_id", "pred"]].rename(columns={"pred": "pred_xgb"}), on=["season", "gsis_id"], how="inner")
        merged = merged.merge(logit_oof[["season", "gsis_id", "pred"]].rename(columns={"pred": "pred_logistic"}), on=["season", "gsis_id"], how="inner")
        assert len(merged) == len(lgbm_oof_s), f"{spec.position}: gate promotion OOF rows misaligned across model types (seed {s})"
        merged["pred_blend"] = train.apply_blend(info["blend_weights"], lgbm=merged["pred_lgbm"], xgb=merged["pred_xgb"], logistic=merged["pred_logistic"])

        seed_blend_cols.append(merged.set_index(["season", "gsis_id"])["pred_blend"].rename(f"seed_{s}"))
        if s == seed:
            base = merged
            base_lgbm_fold_metrics, base_xgb_fold_metrics = lgbm_fold_metrics_s, xgb_fold_metrics_s

    assert base is not None, f"{spec.position}: base seed {seed} missing from gate per_seed spec"
    stacked = pd.concat(seed_blend_cols, axis=1)
    assert not stacked.isna().any().any(), f"{spec.position}: seed OOF pools misaligned across gate-refit seeds"
    ensemble_pooled = base[["season", "gsis_id", "breakout"]].copy()
    ensemble_pooled["pred_blend"] = stacked.mean(axis=1).to_numpy()

    active_calibrator = train.fit_platt(ensemble_pooled["breakout"], ensemble_pooled["pred_blend"])
    ensemble_pooled["pred_calibrated"] = train.apply_calibration("platt", active_calibrator, ensemble_pooled["pred_blend"])
    calib_method, calibrator, calib_diag = train.choose_calibration(base["breakout"], base["pred_blend"])
    smoothed = train.SmoothedIsotonic().fit(base["pred_blend"].to_numpy(), base["breakout"].to_numpy())
    # Base seed's own blend PR-AUC, recomputed fresh against the new-columns OOF -- purely
    # informational (mirrors run_full_pipeline's "blend_pr_auc": the grid search's chosen
    # weights' PR-AUC on the base seed's pooled OOF) and NOT the shipped bundle's real
    # holdout number (that's holdout_metrics' model/pooled row, computed separately below).
    blend_pr_auc = float(mt.pr_auc(base["breakout"], base["pred_blend"]))

    new_frozen = dict(bundle)
    new_frozen["tree_feature_cols"] = tree_cols
    new_frozen["lgbm_final_n_estimators"] = fitted["new_per_seed"][seed]["lgbm_final_n_estimators"]
    new_frozen["xgb_final_n_estimators"] = fitted["new_per_seed"][seed]["xgb_final_n_estimators"]
    new_frozen["calibration_method"] = calib_method
    new_frozen["calibrator"] = calibrator
    new_frozen["calibration_diag"] = calib_diag
    new_frozen["smoothed_calibration_method"] = "smoothed_isotonic"
    new_frozen["smoothed_calibrator"] = smoothed
    new_frozen["pooled_oof_val"] = base[["season", "gsis_id", "pred_blend", "breakout"]].copy()
    new_frozen["per_seed"] = fitted["new_per_seed"]
    new_frozen["pooled_oof_val_ensemble"] = ensemble_pooled[["season", "gsis_id", "breakout", "pred_blend", "pred_calibrated"]].copy()
    new_frozen["active_calibration_method"] = "platt"
    new_frozen["active_calibrator"] = active_calibrator
    new_frozen["cfg"] = dict(bundle["cfg"])

    holdout_preds = train.holdout_predictions(df, seed, new_frozen)
    holdout_reg = train.holdout_regression_eval(df, seed, new_frozen)

    # Regression-head fold OOF (frozen reg hyperparameters, over the new tree_cols) --
    # needed only for generate_report's fold-spread lines; the regressor itself was
    # already retrained (at these SAME frozen params) inside holdout_regression_eval above.
    lgbm_reg_oof, lgbm_reg_fold_metrics, _ = train.regressor_oof(
        df, tree_cols, folds, "lgbm", new_frozen["lgbm_reg_params"], seed
    )
    xgb_reg_oof, xgb_reg_fold_metrics, _ = train.regressor_oof(
        df, tree_cols, folds, "xgb", new_frozen["xgb_reg_params"], seed
    )

    train.save_artifacts(spec, new_frozen, seed, new_frozen["cfg"])

    # ---- run_full_pipeline-shaped result: feeds train.generate_report/_metrics_to_jsonable
    # directly (write_gate_outputs below), so the SAME tested report/metrics machinery every
    # real retrain uses describes this promoted bundle too -- no second, drift-prone schema.
    result = {
        "spec": spec,
        "cfg": new_frozen["cfg"],
        "seed": seed,
        "ensemble_seeds": list(fitted["new_per_seed"].keys()),
        "per_seed": fitted["new_per_seed"],
        "n_trials_clf": 0,  # no Optuna ran -- frozen hyperparameters throughout (see module docstring)
        "n_trials_reg": 0,
        "df": df,
        "tree_cols": tree_cols,
        "logit_cols": logit_cols,
        "folds": folds,
        "logit_C": bundle["logistic_C"],
        "logit_val_score": float("nan"),  # not re-tuned this run -- see module docstring
        "lgbm_fold_metrics": base_lgbm_fold_metrics,
        "xgb_fold_metrics": base_xgb_fold_metrics,
        "logit_fold_metrics": logit_fold_metrics,
        "blend_weights": fitted["new_per_seed"][seed]["blend_weights"],
        "blend_pr_auc": blend_pr_auc,
        "pooled_val": ensemble_pooled,
        "calib_method": "platt",
        "calib_diag": calib_diag,
        "legacy_calib_method": calib_method,
        "lgbm_reg_fold_metrics": lgbm_reg_fold_metrics,
        "xgb_reg_fold_metrics": xgb_reg_fold_metrics,
        "holdout_preds": holdout_preds,
        "holdout_reg": holdout_reg,
        "frozen": new_frozen,
    }
    return {"holdout_preds": holdout_preds, "holdout_reg": holdout_reg, "frozen": new_frozen, "result": result}


def add_to_excluded_features(config_path, new_cols: list[str]) -> None:
    """A REJECTed position's new columns are added to configs/model_{pos}.yaml's

    ``excluded_features`` list -- they stay in the feature matrix (nullable, auditable)
    but a future real Optuna retrain (``python -m src.models.train {pos}``) won't
    silently start using them. Idempotent: a column already listed is not duplicated.
    """
    text = config_path.read_text()
    m = re.search(r"excluded_features:\s*\[([^\]]*)\]", text)
    assert m is not None, f"{config_path}: no `excluded_features: [...]` line found"
    current = [c.strip() for c in m.group(1).split(",") if c.strip()]
    merged = current + [c for c in new_cols if c not in current]
    new_line = f"excluded_features: [{', '.join(merged)}]"
    text = text[: m.start()] + new_line + text[m.end() :]
    config_path.write_text(text)


# --------------------------------------------------------------------------
# outputs/model_{pos}_{report.md,metrics.json} for a KEPT (promoted) position
# --------------------------------------------------------------------------

_GATE_BANNER_TEMPLATE = """> **v2.2 efficiency gate -- frozen-hyperparameter retrain.** This report/bundle was
> regenerated by `src.models.efficiency_gate` (not a fresh `python -m src.models.train
> {pos}` Optuna run): every hyperparameter and blend weight below is FROZEN at the
> pre-v2.2 incumbent bundle's values, per seed -- the only thing that changed is the tree
> feature set, which now includes {extra_cols}. Promoted because pooled (2024+2025)
> holdout top-10 precision did not regress ({base_top10:.3f} -> {var_top10:.3f}) and
> PR-AUC {prauc_verb} ({base_prauc:.3f} -> {var_prauc:.3f}) -- see
> `outputs/efficiency_gate.md` for the full per-position gate table. The numbers in the
> rest of this report describe the SHIPPED bundle at `{artifact_path}` as of this
> promotion, computed the identical way every other position's report is (this position's
> own frozen-hyperparameter holdout retrain, scored once).
"""


def _gate_banner(pos: str, extra_cols: list[str], gate_table: pd.DataFrame, artifact_path) -> str:
    pooled = gate_table[gate_table["season"] == "pooled"].set_index("variant")
    base_top10 = float(pooled.loc["incumbent", "top10_precision"])
    var_top10 = float(pooled.loc["plus_efficiency", "top10_precision"])
    base_prauc = float(pooled.loc["incumbent", "pr_auc"])
    var_prauc = float(pooled.loc["plus_efficiency", "pr_auc"])
    return _GATE_BANNER_TEMPLATE.format(
        pos=pos, extra_cols=extra_cols, base_top10=base_top10, var_top10=var_top10,
        prauc_verb="improved" if var_prauc >= base_prauc else f"regressed by <= {PR_AUC_TOLERANCE}",
        base_prauc=base_prauc, var_prauc=var_prauc, artifact_path=artifact_path,
    )


def write_gate_outputs(spec: train.PositionSpec, result: dict, pos: str, extra_cols: list[str], gate_table: pd.DataFrame) -> dict:
    """Regenerate outputs/model_{pos}_{report.md,metrics.json} for a gate-KEPT position,

    describing the SHIPPED (promoted) bundle -- reuses ``train.generate_report``/
    ``train._metrics_to_jsonable`` verbatim (the SAME machinery a real Optuna retrain
    uses, so the dashboard's ``build_trust_view``/``parse_named_holdout_lists`` keep
    working unchanged), with a clearly-labeled v2.2 banner inserted so nobody mistakes
    this for a fresh full retrain. Returns the parsed metrics dict (for the consistency
    check below / the caller's own reporting).

    Leaving these two files describing the PRE-promotion bundle while a new bundle ships
    is exactly the staleness class ``vegas_experiment.promote_position`` left as a gap
    (never regenerated ``model_{pos}_report.md``/``metrics.json`` for a promoted
    position) -- not a precedent to repeat here.
    """
    report_md = train.generate_report(result, pytest_summary=None)
    title_line, _, rest = report_md.partition("\n")
    title_line = title_line.replace("(v1.7 fixes)", "(v2.2 efficiency gate, frozen-hyperparameter retrain)")
    banner = _gate_banner(pos, extra_cols, gate_table, spec.artifact_path)
    report_md = f"{title_line}\n\n{banner}\n{rest.lstrip(chr(10))}"
    spec.report_path.parent.mkdir(parents=True, exist_ok=True)
    spec.report_path.write_text(report_md)

    metrics = train._metrics_to_jsonable(result)
    metrics["v2_2_efficiency_gate"] = {
        "promoted": True,
        "new_columns": extra_cols,
        "note": "frozen-hyperparameter retrain via src.models.efficiency_gate -- see outputs/efficiency_gate.md",
    }
    spec.metrics_json_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


# --------------------------------------------------------------------------
# Per-position run + report
# --------------------------------------------------------------------------


def run_position(pos: str) -> dict | None:
    spec = train.position_spec(pos)
    extra_cols = new_columns_for(pos)
    if not extra_cols:
        return None
    if not spec.artifact_path.exists():
        print(f"  {pos}: no bundle at {spec.artifact_path}, skipping")
        return None
    bundle = joblib.load(spec.artifact_path)
    cfg = bundle["cfg"]
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=cfg)
    missing_cols = [c for c in extra_cols if c not in df.columns]
    if missing_cols:
        print(f"  {pos}: features_{pos}.parquet missing {missing_cols}; rebuild features first, skipping")
        return None

    split = train.holdout_split(train_start_season=cfg.get("train_season_start", train.TRAIN_START_SEASON))
    folds = train.validation_folds(train_start_season=cfg.get("train_season_start", train.TRAIN_START_SEASON))
    holdout_df = df[df["season"].isin(split.holdout_seasons)]

    if bundle_already_promoted(bundle, extra_cols):
        # Re-running this module against an already-promoted bundle: comparing the
        # promoted bundle against ITSELF would be a degenerate no-op (incumbent ==
        # variant, a fabricated zero lift) -- reconstruct the TRUE pre-promotion
        # incumbent instead (see reconstruct_incumbent_score's docstring). The variant
        # is the bundle's own current (already-fitted) score -- no refit needed for it.
        print(f"  {pos}: bundle already carries {extra_cols} -- reconstructing the true pre-promotion incumbent for a genuine comparison")
        variant_score = score_incumbent(bundle, holdout_df)
        incumbent_score = reconstruct_incumbent_score(bundle, df, extra_cols, folds, split)
        fitted = {
            "per_seed_models": bundle.get("per_seed_models") or {bundle.get("seed", 0): {"lgbm": bundle["lgbm"], "xgb": bundle["xgb"]}},
            "new_per_seed": bundle.get("per_seed") or {bundle.get("seed", 0): {"lgbm_params": bundle["lgbm_params"], "xgb_params": bundle["xgb_params"], "blend_weights": bundle["blend_weights"]}},
            "logistic": bundle["logistic"],
            "tree_cols": list(bundle["tree_feature_cols"]),
        }
    else:
        incumbent_score = score_incumbent(bundle, holdout_df)
        variant_score, fitted = fit_and_score_efficiency_variant(bundle, df, extra_cols, folds, split)

    table = metrics_table(holdout_df, incumbent_score, variant_score)
    keep, reason = gate_decision(table)

    promoted_info = None
    if keep:
        promoted_info = promote_position(spec, bundle, df, fitted)
        write_gate_outputs(spec, promoted_info["result"], pos, extra_cols, table)
        print(f"  {pos}: KEPT -- bundle regenerated at {spec.artifact_path}")
        print(f"  {pos}: wrote {spec.report_path}")
        print(f"  {pos}: wrote {spec.metrics_json_path}")
    else:
        add_to_excluded_features(spec.config_path, extra_cols)
        print(f"  {pos}: REJECTED -- {extra_cols} added to {spec.config_path}'s excluded_features")

    return {
        "position": pos, "label": spec.label_position, "extra_cols": extra_cols,
        "table": table, "keep": keep, "reason": reason, "promoted_info": promoted_info,
    }


def _null_rates(pos: str) -> dict[str, float]:
    spec = train.position_spec(pos)
    df = pd.read_parquet(spec.features_path)
    cols = new_columns_for(pos)
    return {c: float(df[c].isna().mean()) for c in cols if c in df.columns}


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(all_results: list[dict]) -> None:
    lines = ["# BreakoutLab -- Efficiency-Proxy Capacity Gate (v2.2)", ""]
    lines.append(
        "Every hyperparameter/blend weight is FROZEN at the currently shipped bundle's "
        "per-seed values (no Optuna retuning) -- only the tree feature set (+ this "
        "position's new efficiency-proxy columns) and the n_estimators/calibrator that "
        f"legitimately follow from it differ. Gate rule (binding): KEEP iff pooled "
        f"(2024+2025) holdout top-10 precision does not regress AND PR-AUC improves or "
        f"regresses by at most {PR_AUC_TOLERANCE}."
    )
    lines.append("")
    lines.append("## Per-position decisions")
    lines.append("")
    rows = []
    for r in all_results:
        pooled = r["table"][r["table"]["season"] == "pooled"].set_index("variant")
        rows.append(
            [
                r["label"], "KEEP" if r["keep"] else "REJECT",
                f"{pooled.loc['incumbent','top10_precision']:.3f}", f"{pooled.loc['plus_efficiency','top10_precision']:.3f}",
                f"{pooled.loc['incumbent','pr_auc']:.3f}", f"{pooled.loc['plus_efficiency','pr_auc']:.3f}",
            ]
        )
    lines.append(_markdown_table(
        ["Position", "Decision", "Incumbent top-10", "+Efficiency top-10", "Incumbent PR-AUC", "+Efficiency PR-AUC"], rows
    ))
    lines.append("")

    for r in all_results:
        lines.append(f"## {r['label']}")
        lines.append("")
        lines.append(f"New columns: {r['extra_cols']}")
        lines.append("")
        null_rates = _null_rates(r["position"])
        lines.append(_markdown_table(["Column", "Null rate"], [[c, f"{v:.3f}"] for c, v in null_rates.items()]))
        lines.append("")
        table_rows = [
            [row["variant"], row["season"], row["n"], row["n_pos"], f"{row['top10_precision']:.3f}", f"{row['pr_auc']:.3f}"]
            for row in r["table"].to_dict("records")
        ]
        lines.append(_markdown_table(["Variant", "Season", "n", "n_pos", "Top-10 precision", "PR-AUC"], table_rows))
        lines.append("")
        lines.append(f"**Decision: {r['reason']}**")
        lines.append("")

    OUT_PATH.write_text("\n".join(lines))


def main() -> int:
    all_results = []
    for pos in ("wr", "rb", "te"):
        print(f"efficiency gate: {pos} ...")
        r = run_position(pos)
        if r is not None:
            all_results.append(r)
    write_report(all_results)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
