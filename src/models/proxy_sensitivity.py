"""v1.7 Step 1 -- proxy-era training-pool sensitivity experiment.

``configs/labels.yaml``'s label table blends two eras: 2014-2019 breakout labels are
built off a *proxy* preseason-expectation signal (prior-season finish, since real ADP/ECR
market data isn't available that far back -- see ``src/labels/build.py``), while 2020-2025
uses the real market (``adp_source == "ecr"``). Every position's shipped model has always
trained on the full 2014-2025 pool (Arm A below). This script asks, per position, holding
every hyperparameter/blend-weight choice frozen at the shipped bundle's values (no Optuna
retuning -- this experiment is about the TRAINING POOL, not the model): does training on
ONLY the real-market era (2020+, Arm B) do at least as well on holdout?

Design (frozen throughout, both arms)
--------------------------------------
- Every hyperparameter (LGBM/XGB params + their early-stopping-derived n_estimators is
  the one exception -- see below, logistic C) and the classifier-trio BLEND WEIGHTS come
  straight from the currently shipped ``data/models/{pos}_model_bundle.joblib`` --
  reused identically for both arms, so the comparison isolates the training pool, not a
  second free hyperparameter search. (Blend weights are rank-invariant to arm anyway for
  this experiment's decision metrics -- PR-AUC/top-10 precision only depend on score
  order, which a fixed weighting doesn't change relative to itself across arms.)
- The one thing that DOES get refit per arm at those frozen params: LGBM/XGB's
  no-early-stopping holdout ``n_estimators`` (mean OOF best_iteration -- the exact
  mechanism ``src.models.train.run_full_pipeline`` uses) and the shared logistic head's
  coefficients, both of which legitimately depend on how much/what training data is on
  hand. That is what the brief's "shifted validation folds for any refit internals that
  need them" instruction is for.
- Arm A: the current pool, train_start_season=2014 -- ``cv.validation_folds()``, all 4
  folds (val 2020, 2021, 2022, 2023).
- Arm B: real-market-only, train_start_season=2020 -- ``cv.validation_folds(2020)``
  yields exactly 2 folds (train 2020-21 -> val 2022; train 2020-22 -> val 2023) via the
  MIN_FOLD_TRAIN_YEARS rule (see ``src.models.cv``) -- matching this brief's shifted-fold
  spec by construction, not by a special case in this script.
- Both arms retrain on train<=2023 (Arm A: 2014-2023; Arm B: 2020-2023) and are scored
  ONCE on the identical holdout rows (2024+2025 -- present in both arms' pool, since
  holdout years are always >= 2020).

Decision rule (binding, pre-stated -- see the brief)
-------------------------------------------------------
Choose Arm B for a position iff (holdout top-10 precision, B >= A) AND (holdout PR-AUC,
B >= A - 0.02). Otherwise keep Arm A. Applied mechanically below, never re-litigated
after seeing the numbers.
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.inference import board_2026 as bd
from src.models import cv
from src.models import metrics as mt
from src.models import train

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

OUT_PATH = train.OUTPUTS_DIR / "proxy_sensitivity.md"

ARM_TRAIN_START = {"A": 2014, "B": 2020}


def _holdout_retrain_and_predict(df_arm: pd.DataFrame, bundle: dict, folds: list, seed: int) -> dict:
    """One arm's full recipe at FROZEN bundle hyperparameters: fold-OOF (this arm's own

    folds) to derive this arm's final_n_estimators -> holdout retrain -> blended holdout
    raw score. Returns everything the report table + the 2026 scoring step need.
    """
    tree_cols = bundle["tree_feature_cols"]
    logit_cols = bundle["logistic_feature_cols"]
    weights = bundle["blend_weights"]

    lgbm_oof, lgbm_fold_metrics, lgbm_best_iters = train.classifier_oof(df_arm, tree_cols, folds, "lgbm", bundle["lgbm_params"], seed)
    xgb_oof, xgb_fold_metrics, xgb_best_iters = train.classifier_oof(df_arm, tree_cols, folds, "xgb", bundle["xgb_params"], seed)
    lgbm_n_est = int(round(np.mean(lgbm_best_iters)))
    xgb_n_est = int(round(np.mean(xgb_best_iters)))

    split = cv.holdout_split(train_start_season=min(f.train_seasons[0] for f in folds))
    train_df = df_arm[df_arm["season"].isin(split.train_seasons)]
    holdout_df = df_arm[df_arm["season"].isin(split.holdout_seasons)]
    spw = train._scale_pos_weight(train_df["breakout"])

    lgbm_model = train._fit_holdout_lgbm(train_df, tree_cols, bundle["lgbm_params"], lgbm_n_est, seed, spw)
    xgb_model = train._fit_holdout_xgb(train_df, tree_cols, bundle["xgb_params"], xgb_n_est, seed, spw)
    logit_model, logit_imputer, logit_scaler = train.fit_logistic(train_df, holdout_df, logit_cols, bundle["logistic_C"], seed)

    lgbm_pred = lgbm_model.predict_proba(holdout_df[tree_cols])[:, 1]
    xgb_pred = xgb_model.predict_proba(holdout_df[tree_cols])[:, 1]
    logit_pred = train._logistic_predict(logit_model, logit_imputer, logit_scaler, holdout_df, logit_cols)
    blended = train.apply_blend(weights, lgbm=lgbm_pred, xgb=xgb_pred, logistic=logit_pred)

    out = holdout_df[["season", "gsis_id", "player_name", "breakout"]].copy()
    out["raw_score"] = blended

    return {
        "lgbm_n_estimators": lgbm_n_est,
        "xgb_n_estimators": xgb_n_est,
        "holdout_preds": out,
        "models": {"lgbm": lgbm_model, "xgb": xgb_model, "logistic": (logit_model, logit_imputer, logit_scaler)},
        "n_train": int(len(train_df)),
    }


def _holdout_metric_table(holdout_preds: pd.DataFrame) -> dict:
    rows = {}
    for season in sorted(holdout_preds["season"].unique().tolist()) + ["pooled"]:
        mask = holdout_preds["season"] == season if season != "pooled" else pd.Series(True, index=holdout_preds.index)
        y = holdout_preds.loc[mask, "breakout"]
        s = holdout_preds.loc[mask, "raw_score"]
        rows[season] = {
            "n": int(mask.sum()),
            "n_pos": int(y.sum()),
            "pr_auc": mt.pr_auc(y, s),
            "top10_precision": mt.top_k_precision(y, s, 10),
        }
    return rows


def _score_2026(pos: str, bundle: dict, models: dict, raw: dict) -> pd.DataFrame:
    feats = bd.build_veteran_feature_matrix(pos, raw, bundle)
    df = feats.to_pandas()
    tree_cols = bundle["tree_feature_cols"]
    logit_cols = bundle["logistic_feature_cols"]
    weights = bundle["blend_weights"]

    lgbm_pred = models["lgbm"].predict_proba(df[tree_cols])[:, 1]
    xgb_pred = models["xgb"].predict_proba(df[tree_cols])[:, 1]
    logit_model, logit_imputer, logit_scaler = models["logistic"]
    logit_pred = train._logistic_predict(logit_model, logit_imputer, logit_scaler, df, logit_cols)
    blended = train.apply_blend(weights, lgbm=lgbm_pred, xgb=xgb_pred, logistic=logit_pred)
    return pd.DataFrame({"gsis_id": df["gsis_id"], "player_name": df["player_name"], "raw_score": blended})


def run_position(pos: str, raw: dict) -> dict:
    spec = train.position_spec(pos)
    bundle = joblib.load(spec.artifact_path)
    seed = bundle["seed"]

    arm_results = {}
    for arm, train_start in ARM_TRAIN_START.items():
        # Arm A must see the FULL 2014-2025 pool. A position's shipped bundle cfg may itself
        # carry a `train_season_start` restriction (e.g. RB's v1.7 2020+ real-market-only
        # decision) -- that restriction is exactly the axis this experiment is testing, so
        # loading with the raw bundle cfg would silently pre-filter Arm A to Arm B's pool
        # (collapsing the comparison and leaving Arm A's 2014-start folds empty). Override
        # `train_season_start` per arm, mirroring pooled_experiment.py's `pool_cfg` pattern.
        pool_cfg = dict(bundle["cfg"])
        pool_cfg["train_season_start"] = train_start
        df_full = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=pool_cfg)
        df_arm = df_full[df_full["season"] >= train_start].reset_index(drop=True)
        folds = cv.validation_folds(train_start_season=train_start)
        res = _holdout_retrain_and_predict(df_arm, bundle, folds, seed)
        res["metric_table"] = _holdout_metric_table(res["holdout_preds"])
        res["n_pool"] = int(len(df_arm))
        res["folds"] = folds
        res["board_2026"] = _score_2026(pos, bundle, res["models"], raw)
        arm_results[arm] = res

    a_pooled, b_pooled = arm_results["A"]["metric_table"]["pooled"], arm_results["B"]["metric_table"]["pooled"]
    keep_b = (b_pooled["top10_precision"] >= a_pooled["top10_precision"]) and (
        b_pooled["pr_auc"] >= a_pooled["pr_auc"] - 0.02
    )
    decision = "B (real-market-only, train_season_start=2020)" if keep_b else "A (full pool, train_season_start=2014)"

    board_a = arm_results["A"]["board_2026"].set_index("gsis_id")["raw_score"]
    board_b = arm_results["B"]["board_2026"].set_index("gsis_id")["raw_score"]
    common = board_a.index.intersection(board_b.index)
    rho, pval = spearmanr(board_a.loc[common], board_b.loc[common])

    return {
        "position": pos,
        "label_position": spec.label_position,
        "arm_results": arm_results,
        "keep_b": keep_b,
        "decision": decision,
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "n_common_2026": int(len(common)),
    }


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _arm_table_md(arm_results: dict) -> str:
    lines = ["| Arm | Season | n | n_pos | PR-AUC | Top-10 precision |", "| --- | --- | --- | --- | --- | --- |"]
    for arm, res in arm_results.items():
        for season, m in res["metric_table"].items():
            lines.append(f"| {arm} | {season} | {m['n']} | {m['n_pos']} | {_fmt(m['pr_auc'])} | {_fmt(m['top10_precision'])} |")
    return "\n".join(lines)


def write_report(all_results: list[dict]) -> None:
    lines = ["# BreakoutLab -- Proxy-Era Training-Pool Sensitivity (v1.7 Step 1)", ""]
    lines.append(
        "Arm A = full 2014-2025 pool (current shipped training pool, mixes the pre-2020 "
        "prior-season-finish PROXY expectation label with the 2020+ real-market ECR label). "
        "Arm B = real-market-only pool, season >= 2020. Every hyperparameter and blend weight "
        "is FROZEN at the currently shipped bundle's values for both arms (no Optuna retuning) "
        "-- only the training pool (and the fold structure/n_estimators/logistic refit that "
        "follows from it) differs. Both arms retrain on train<=2023 and are scored ONCE on the "
        "identical 2024+2025 holdout rows. Decision rule (binding, pre-stated): choose Arm B iff "
        "its holdout top-10 precision >= Arm A's AND its holdout PR-AUC >= Arm A's - 0.02; "
        "otherwise keep Arm A."
    )
    lines.append("")
    lines.append("## Per-position decisions")
    lines.append("")
    lines.append("| Position | Decision | A top-10 | B top-10 | A PR-AUC | B PR-AUC | Spearman(2026 board, A vs B) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in all_results:
        a, b = r["arm_results"]["A"]["metric_table"]["pooled"], r["arm_results"]["B"]["metric_table"]["pooled"]
        lines.append(
            f"| {r['label_position']} | {r['decision']} | {_fmt(a['top10_precision'])} | {_fmt(b['top10_precision'])} | "
            f"{_fmt(a['pr_auc'])} | {_fmt(b['pr_auc'])} | {_fmt(r['spearman_rho'])} (p={_fmt(r['spearman_p'], 4)}, n={r['n_common_2026']}) |"
        )
    lines.append("")

    for r in all_results:
        lines.append(f"## {r['label_position']}")
        lines.append("")
        lines.append(f"Pool sizes -- Arm A: {r['arm_results']['A']['n_pool']:,} rows; Arm B: {r['arm_results']['B']['n_pool']:,} rows.")
        lines.append(
            f"Arm B fold structure: {[(f.train_seasons[0], f.train_seasons[-1], f.val_season) for f in r['arm_results']['B']['folds']]} "
            f"(vs Arm A's {[(f.train_seasons[0], f.train_seasons[-1], f.val_season) for f in r['arm_results']['A']['folds']]})."
        )
        lines.append("")
        lines.append(_arm_table_md(r["arm_results"]))
        lines.append("")
        lines.append(f"**Decision: {r['decision']}**")
        lines.append("")
        lines.append(
            f"2026 board raw-score Spearman correlation between arms: rho={_fmt(r['spearman_rho'])} "
            f"(p={_fmt(r['spearman_p'], 4)}, n={r['n_common_2026']} common veterans)."
        )
        lines.append("")

    OUT_PATH.write_text("\n".join(lines))


def main() -> int:
    raw = bd.load_raw_frames()
    all_results = []
    for pos in train.POSITIONS:
        print(f"proxy sensitivity: {pos} ...")
        r = run_position(pos, raw)
        all_results.append(r)
        print(f"  decision: {r['decision']}")
    write_report(all_results)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
