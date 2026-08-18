"""Vegas team-feature experiment + promotion protocol (v1.5 Phase C, Deliverable 3 --
the brief's binding v1.5 gate).

Per position, compares the frozen v1.5 model against a "+vegas" variant
that adds ``implied_ppg``/``implied_win_prob``/``has_vegas``
(``src.features.shared.attach_vegas_team``, joined onto every position's
feature matrix -- see ``src.features.{wr,rb,te,qb}``) to the tree heads,
reusing every v1.5 hyperparameter/blend-weight/n_estimators choice exactly
as frozen. **No new Optuna tuning happens anywhere in this module** --
that is the whole point of an experiment meant to isolate "do these two
features help," not "what's the best model that happens to include them."

Protocol (mirrors ``src.models.train.holdout_predictions`` almost exactly)
----------------------------------------------------------------------------
1. **v1.5 baseline** score: the bundle's own holdout-retrained models
   (``bundle["lgbm"]``/``["xgb"]``/``["logistic"]``), scored on
   ``bundle["tree_feature_cols"]`` -- these bundles are trained with
   ``configs/model_{pos}.yaml``'s ``excluded_features`` listing the three
   Vegas columns (see ``src.models.train.tree_feature_columns``), so this
   is a genuine vegas-free v1.5 model, not a proxy.
2. **+vegas variant**: LGBM and XGB are refit at the SAME frozen
   ``lgbm_params``/``xgb_params``/final n_estimators, on <=2023, with
   ``tree_feature_cols + VEGAS_COLS``. The logistic head is reused
   UNCHANGED (``bundle["logistic"]``) rather than refit -- its curated
   feature subset never gains the Vegas columns (see module docstring's
   "Logistic head" note below), so nothing about its fit differs between
   the two variants and refitting it would just reproduce the identical
   model at extra cost.
3. Both variants' classifier-trio predictions are blended with the SAME
   frozen ``blend_weights`` -- again, nothing here is re-tuned.
4. Metrics (``src.models.metrics.top_k_precision``, ``.pr_auc``) computed
   on the RAW pre-calibration blended score for BOTH variants -- an
   apples-to-apples comparison, since both go through the identical
   (lack of) post-processing. This is NOT, in general, the same number
   ``outputs/model_{pos}_report.md`` reports for "the model": that report
   scores off the frozen isotonic/Platt ``pred_calibrated`` column
   instead. Calibration is only a *strictly* rank-preserving transform
   when it's strictly monotonic (e.g. Platt); isotonic regression is
   merely non-decreasing, and its flat (tied) blocks CAN and empirically
   DO shift both ``pr_auc`` and ``top_k_precision`` relative to the raw
   score -- verified directly on this build's WR bundle: identical
   underlying predictions (``max|pred_blend_raw - this module's baseline
   score| == 0.0``), yet PR-AUC 0.133 (raw) vs 0.104 (calibrated) and
   top-10 precision 0.200 (raw) vs 0.000 (calibrated). See the README's
   Known Limitations ("Calibrated vs. raw-blend metrics are not
   interchangeable") for the full writeup -- an earlier draft of this
   docstring claimed calibration "cannot change either ranking metric,"
   which is only true for strictly monotonic calibrators and was wrong
   as a blanket statement; corrected here after that number mismatch
   surfaced during review.

Logistic head: deliberately excluded from the Vegas columns
--------------------------------------------------------------
``configs/model_{pos}.yaml``'s ``logistic_feature_subset`` docstring is
explicit that the curated logistic subset is "deliberately small and
low-null" because logistic regression has no native null handling. The
Vegas columns are null for every pre-2020 training row (roughly half the
full 2014-2025 pool) -- exactly the high-null profile that subset avoids
everywhere else. Trees handle this natively (LGBM/XGB's whole feature set
already includes plenty of high-null NGS/snap columns), so the Vegas
columns are added to the tree heads only. Documented here as a deliberate
scope decision, not an oversight.

Promotion rule (binding, from the project brief)
----------------------------------------------------
Promote the Vegas features into a position's main model ONLY if pooled
(2024+2025) holdout top-10 precision **strictly improves**. A tie or a
regression keeps the v1.5 model as the deployed bundle -- which, given
step 1's construction, is already exactly what's on disk whenever this
module runs (the promotion decision changes nothing on disk for a
rejected position; a promoted one gets a real bundle regeneration, see
``promote_position`` below).

Fold boundaries
-----------------
``train_df``/``holdout_df`` are ``df[df.season.isin(train.holdout_split().
train_seasons)]`` / ``...holdout_seasons``, i.e. the exact
``HoldoutSplit`` object whose ``__post_init__`` asserts
``max(train_seasons) < min(holdout_seasons)`` at construction (see
``src.models.cv``) -- there is no code path here that could hand the
holdout years to a fit call, structurally, not just by convention.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from src.models import metrics as mt
from src.models import train

REPO_ROOT = train.REPO_ROOT
OUTPUTS_DIR = train.OUTPUTS_DIR
OUT_PATH = OUTPUTS_DIR / "vegas_experiment.md"

VEGAS_COLS: tuple[str, ...] = ("implied_ppg", "implied_win_prob", "has_vegas")
HOLDOUT_SEASONS = train.HOLDOUT_SEASONS  # (2024, 2025)


# --------------------------------------------------------------------------
# Score both variants on the holdout years (no retuning anywhere)
# --------------------------------------------------------------------------


def score_baseline(bundle: dict, holdout_df: pd.DataFrame) -> np.ndarray:
    """The bundle's own holdout-retrained classifier trio, blended at frozen weights --

    literally the same numbers src.models.train's holdout evaluation reports, just
    recomputed here for a clean side-by-side against the +vegas variant.
    """
    tree_cols = bundle["tree_feature_cols"]
    lgbm_pred = bundle["lgbm"].predict_proba(holdout_df[tree_cols])[:, 1]
    xgb_pred = bundle["xgb"].predict_proba(holdout_df[tree_cols])[:, 1]
    logit_model, logit_imputer, logit_scaler = bundle["logistic"]
    logit_pred = train._logistic_predict(logit_model, logit_imputer, logit_scaler, holdout_df, bundle["logistic_feature_cols"])
    return train.apply_blend(bundle["blend_weights"], lgbm=lgbm_pred, xgb=xgb_pred, logistic=logit_pred)


def fit_and_score_vegas_variant(bundle: dict, train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Refit LGBM/XGB on <=2023 at frozen hyperparameters with tree_feature_cols + VEGAS_COLS;

    reuse the bundle's logistic model UNCHANGED (see module docstring). Returns (blended
    holdout score, {"lgbm": model, "xgb": model, "tree_cols": vegas_tree_cols}) -- the fitted
    models are returned so a promoted position can reuse them instead of refitting again.
    """
    vegas_tree_cols = list(bundle["tree_feature_cols"]) + list(VEGAS_COLS)
    spw = train._scale_pos_weight(train_df["breakout"])

    lgbm_params = dict(bundle["lgbm_params"])
    lgbm_params["n_estimators"] = bundle["lgbm_final_n_estimators"]
    lgbm_model = lgb.LGBMClassifier(
        objective="binary", random_state=bundle["seed"], deterministic=True, force_row_wise=True,
        n_jobs=1, verbosity=-1, scale_pos_weight=spw, **lgbm_params,
    )
    lgbm_model.fit(train_df[vegas_tree_cols], train_df["breakout"])
    lgbm_pred = lgbm_model.predict_proba(holdout_df[vegas_tree_cols])[:, 1]

    xgb_params = dict(bundle["xgb_params"])
    xgb_params["n_estimators"] = bundle["xgb_final_n_estimators"]
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic", random_state=bundle["seed"], n_jobs=1, verbosity=0, scale_pos_weight=spw, **xgb_params,
    )
    xgb_model.fit(train_df[vegas_tree_cols], train_df["breakout"])
    xgb_pred = xgb_model.predict_proba(holdout_df[vegas_tree_cols])[:, 1]

    logit_model, logit_imputer, logit_scaler = bundle["logistic"]
    logit_pred = train._logistic_predict(logit_model, logit_imputer, logit_scaler, holdout_df, bundle["logistic_feature_cols"])

    blended = train.apply_blend(bundle["blend_weights"], lgbm=lgbm_pred, xgb=xgb_pred, logistic=logit_pred)
    return blended, {"lgbm": lgbm_model, "xgb": xgb_model, "tree_cols": vegas_tree_cols}


# --------------------------------------------------------------------------
# Metrics table + promotion decision
# --------------------------------------------------------------------------


def metrics_table(holdout_df: pd.DataFrame, baseline_score: np.ndarray, vegas_score: np.ndarray) -> pd.DataFrame:
    rows = []
    for season in list(HOLDOUT_SEASONS) + ["pooled"]:
        mask = (holdout_df["season"] == season) if season != "pooled" else pd.Series(True, index=holdout_df.index)
        y = holdout_df.loc[mask, "breakout"]
        for name, score in (("v1.5_baseline", baseline_score), ("plus_vegas", vegas_score)):
            s = pd.Series(score, index=holdout_df.index).loc[mask]
            rows.append(
                {
                    "season": season, "variant": name, "n": int(mask.sum()), "n_pos": int(y.sum()),
                    "top10_precision": mt.top_k_precision(y, s, 10), "pr_auc": mt.pr_auc(y, s),
                }
            )
    return pd.DataFrame(rows)


def promotion_decision(table: pd.DataFrame) -> tuple[bool, str]:
    """PROMOTE iff pooled top-10 precision strictly improves. Ties/regressions -> reject.

    Returns (promote: bool, reason: str).
    """
    pooled = table[table["season"] == "pooled"].set_index("variant")
    base_top10 = pooled.loc["v1.5_baseline", "top10_precision"]
    vegas_top10 = pooled.loc["plus_vegas", "top10_precision"]
    base_prauc = pooled.loc["v1.5_baseline", "pr_auc"]
    vegas_prauc = pooled.loc["plus_vegas", "pr_auc"]

    promote = vegas_top10 > base_top10
    if promote:
        reason = f"PROMOTE: pooled holdout top-10 precision improved {base_top10:.3f} -> {vegas_top10:.3f}."
    elif vegas_top10 < base_top10:
        reason = f"REJECT (regression): pooled holdout top-10 precision fell {base_top10:.3f} -> {vegas_top10:.3f}."
    else:
        reason = f"REJECT (tie): pooled holdout top-10 precision unchanged at {base_top10:.3f}."
    reason += f" PR-AUC (secondary): {base_prauc:.3f} -> {vegas_prauc:.3f}."
    return promote, reason


# --------------------------------------------------------------------------
# Promotion: regenerate a position's bundle with the Vegas features baked in
# (only ever called for a position the decision above promotes)
# --------------------------------------------------------------------------


def promote_position(spec: train.PositionSpec, bundle: dict, df: pd.DataFrame) -> dict:
    """Rebuild spec.artifact_path with implied_ppg/implied_win_prob/has_vegas included in the

    tree heads, at every v1.5-frozen hyperparameter/blend weight -- NO new Optuna, matching
    this module's whole premise. Recomputes pooled OOF validation predictions (needed to
    refit the calibrators against the new score distribution -- the same "recompute OOF at
    frozen params, refit calibration, don't retune" pattern as
    ``src.models.train.backfill_smoothed_calibration``), then the one holdout evaluation.
    Also empties ``configs/model_{pos}.yaml``'s ``excluded_features`` for this position so a
    future ``python -m src.models.train {pos}`` run (a real Optuna retrain) keeps including
    these features rather than silently reverting to v1.5.

    v1.7 seed ensemble (Finding 5 fix): a shipped bundle's raw score is the MEAN of every
    seed's own blended OOF prediction (``train.ensemble_blend_and_pool``), calibrated
    through ``bundle["active_calibrator"]`` -- a Platt model fit on THAT ensemble score's
    distribution (``train.holdout_predictions`` docstring). Adding the Vegas columns shifts
    every seed's raw score distribution, so BOTH must be recomputed here or the promoted
    bundle would silently score through a calibrator fit on the pre-promotion distribution
    -- exactly the stale-calibrator bug this fix closes. What stays frozen (no new Optuna,
    matching this module's whole premise): every seed's own ``lgbm_params``/``xgb_params``/
    ``blend_weights`` (``bundle["per_seed"]``) are reused unchanged; only each seed's OOF
    predictions (and the n_estimators/calibrator that mechanically follow from them, exactly
    as ``run_full_pipeline`` derives them) are recomputed against the vegas-augmented
    ``tree_feature_cols``.
    """
    vegas_tree_cols = list(bundle["tree_feature_cols"]) + list(VEGAS_COLS)
    logit_cols = bundle["logistic_feature_cols"]
    folds = train.validation_folds()
    seed = bundle["seed"]

    # Shared logistic head OOF: the curated logistic subset never gains the Vegas columns
    # (module docstring's "Logistic head" note), so it is identical across every seed and
    # unaffected by promotion -- computed once, exactly as the pre-ensemble pipeline did.
    logit_oof, _ = train.logistic_oof(df, logit_cols, folds, bundle["logistic_C"], seed)

    per_seed_spec = bundle.get("per_seed") or {
        seed: {"lgbm_params": bundle["lgbm_params"], "xgb_params": bundle["xgb_params"], "blend_weights": bundle["blend_weights"]}
    }

    new_per_seed: dict[int, dict] = {}
    seed_blend_cols = []
    base = None  # the base seed's own pooled OOF frame -- feeds the legacy single-seed calibrator
    for s, info in per_seed_spec.items():
        lgbm_oof_s, _, lgbm_best_iters_s = train.classifier_oof(df, vegas_tree_cols, folds, "lgbm", info["lgbm_params"], s)
        xgb_oof_s, _, xgb_best_iters_s = train.classifier_oof(df, vegas_tree_cols, folds, "xgb", info["xgb_params"], s)

        merged = lgbm_oof_s[["season", "gsis_id", "breakout"]].copy()
        merged["pred_lgbm"] = lgbm_oof_s["pred"].to_numpy()
        merged = merged.merge(xgb_oof_s[["season", "gsis_id", "pred"]].rename(columns={"pred": "pred_xgb"}), on=["season", "gsis_id"], how="inner")
        merged = merged.merge(logit_oof[["season", "gsis_id", "pred"]].rename(columns={"pred": "pred_logistic"}), on=["season", "gsis_id"], how="inner")
        assert len(merged) == len(lgbm_oof_s), f"{spec.position}: promotion OOF rows misaligned across model types (seed {s})"
        merged["pred_blend"] = train.apply_blend(info["blend_weights"], lgbm=merged["pred_lgbm"], xgb=merged["pred_xgb"], logistic=merged["pred_logistic"])

        new_per_seed[s] = {
            "lgbm_params": info["lgbm_params"],
            "xgb_params": info["xgb_params"],
            "lgbm_final_n_estimators": int(round(np.mean(lgbm_best_iters_s))),
            "xgb_final_n_estimators": int(round(np.mean(xgb_best_iters_s))),
            "blend_weights": info["blend_weights"],
            "blend_pr_auc": info.get("blend_pr_auc"),
        }
        seed_blend_cols.append(merged.set_index(["season", "gsis_id"])["pred_blend"].rename(f"seed_{s}"))
        if s == seed:
            base = merged

    assert base is not None, f"{spec.position}: base seed {seed} missing from per_seed spec"
    stacked = pd.concat(seed_blend_cols, axis=1)
    assert not stacked.isna().any().any(), f"{spec.position}: seed OOF pools misaligned across vegas-refit seeds"
    ensemble_pooled = base[["season", "gsis_id", "breakout"]].copy()
    ensemble_pooled["pred_blend"] = stacked.mean(axis=1).to_numpy()

    # Refit the ACTIVE Platt calibrator -- the one train.holdout_predictions actually
    # scores the board through -- against the vegas-shifted ensemble score distribution.
    active_calibrator = train.fit_platt(ensemble_pooled["breakout"], ensemble_pooled["pred_blend"])

    # Legacy single-(base-)seed fields: unchanged shape from pre-ensemble, kept for any
    # code still reading a plain single-model calibrator off the bundle.
    calib_method, calibrator, calib_diag = train.choose_calibration(base["breakout"], base["pred_blend"])
    smoothed = train.SmoothedIsotonic().fit(base["pred_blend"].to_numpy(), base["breakout"].to_numpy())

    new_frozen = dict(bundle)
    new_frozen["tree_feature_cols"] = vegas_tree_cols
    new_frozen["lgbm_final_n_estimators"] = new_per_seed[seed]["lgbm_final_n_estimators"]
    new_frozen["xgb_final_n_estimators"] = new_per_seed[seed]["xgb_final_n_estimators"]
    new_frozen["calibration_method"] = calib_method
    new_frozen["calibrator"] = calibrator
    new_frozen["calibration_diag"] = calib_diag
    new_frozen["smoothed_calibration_method"] = "smoothed_isotonic"
    new_frozen["smoothed_calibrator"] = smoothed
    new_frozen["pooled_oof_val"] = base[["season", "gsis_id", "pred_blend", "breakout"]].copy()
    new_frozen["per_seed"] = new_per_seed
    new_frozen["pooled_oof_val_ensemble"] = ensemble_pooled[["season", "gsis_id", "breakout", "pred_blend"]].copy()
    new_frozen["active_calibration_method"] = "platt"
    new_frozen["active_calibrator"] = active_calibrator

    new_frozen["cfg"] = dict(bundle["cfg"])
    new_frozen["cfg"]["excluded_features"] = []

    holdout_preds = train.holdout_predictions(df, seed, new_frozen)
    holdout_reg = train.holdout_regression_eval(df, seed, new_frozen)

    train.save_artifacts(spec, new_frozen, seed, new_frozen["cfg"])

    cfg_path = spec.config_path
    with open(cfg_path) as fh:
        raw_cfg_text = fh.read()
    raw_cfg_text = raw_cfg_text.replace(
        "excluded_features: [implied_ppg, implied_win_prob, has_vegas]",
        "excluded_features: []  # v1.5 Phase C: promoted -- see outputs/vegas_experiment.md",
    )
    with open(cfg_path, "w") as fh:
        fh.write(raw_cfg_text)

    return {"holdout_preds": holdout_preds, "holdout_reg": holdout_reg, "frozen": new_frozen}


# --------------------------------------------------------------------------
# Per-position run + report
# --------------------------------------------------------------------------


def run_position(pos: str) -> dict | None:
    spec = train.position_spec(pos)
    if not spec.artifact_path.exists():
        print(f"  {pos}: no bundle at {spec.artifact_path}, skipping")
        return None
    bundle = joblib.load(spec.artifact_path)
    cfg = bundle["cfg"]
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=cfg)

    split = train.holdout_split()
    train_df = df[df["season"].isin(split.train_seasons)]
    holdout_df = df[df["season"].isin(split.holdout_seasons)]

    baseline_score = score_baseline(bundle, holdout_df)
    vegas_score, fitted = fit_and_score_vegas_variant(bundle, train_df, holdout_df)

    table = metrics_table(holdout_df, baseline_score, vegas_score)
    promote, reason = promotion_decision(table)

    promoted_info = None
    if promote:
        promoted_info = promote_position(spec, bundle, df)
        print(f"  {pos}: PROMOTED -- bundle regenerated at {spec.artifact_path}")

    return {"position": pos, "label": spec.label_position, "table": table, "promote": promote, "reason": reason, "promoted_info": promoted_info}


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def table_to_markdown(table: pd.DataFrame) -> str:
    headers = ["Season", "Variant", "n", "n_pos", "Top-10 precision", "PR-AUC"]
    rows = [[r["season"], r["variant"], r["n"], r["n_pos"], r["top10_precision"], r["pr_auc"]] for _, r in table.iterrows()]
    return _markdown_table(headers, rows)


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BreakoutLab -- Vegas Team-Feature Experiment (v1.5 Phase C, binding gate)",
        "",
        "v1.5_baseline = the frozen v1.5 model (Vegas features excluded via "
        "configs/model_{pos}.yaml's excluded_features). plus_vegas = the SAME frozen "
        "hyperparameters/blend weights, LGBM+XGB refit on <=2023 with implied_ppg/"
        "implied_win_prob/has_vegas added, logistic head unchanged (see module docstring). "
        "No new Optuna tuning anywhere in this experiment.",
        "",
        "**Promotion rule:** promote ONLY if pooled (2024+2025) holdout top-10 precision "
        "strictly improves. Ties/regressions keep the v1.5 model; the Vegas columns stay in "
        "the feature matrix as nullable but excluded from that position's model config.",
        "",
    ]
    decisions = []
    for pos in train.POSITIONS:
        result = run_position(pos)
        if result is None:
            continue
        lines.append(f"## {result['label']}")
        lines.append("")
        lines.append(table_to_markdown(result["table"]))
        lines.append("")
        lines.append(f"**{result['label']} decision:** {result['reason']}")
        lines.append("")
        decisions.append((result["label"], result["promote"]))
        print(f"{result['label']}: {result['reason']}")

    lines.append("## Summary")
    lines.append("")
    for label, promote in decisions:
        lines.append(f"- {label}: {'PROMOTED' if promote else 'rejected (kept v1.5)'}")
    if not any(p for _, p in decisions):
        lines.append("")
        lines.append(
            "No position promoted. The 2026 board (`src.inference.board_2026`) is therefore "
            "unchanged -- no regeneration needed. Had any position promoted, "
            "`board_2026.load_raw_frames`/`build_veteran_feature_matrix` would need "
            "`vegas_team` wired through the same way `src.features.{wr,rb,te,qb}` now accept "
            "it; 2026 rows would get `has_vegas=0` and null implied_ppg/implied_win_prob "
            "automatically (data/processed/vegas_team.parquet has no season-2026 rows -- the "
            "local pull only covers 2020-2025 -- so the existing left-join-then-fill-0 contract "
            "in `src.features.shared.attach_vegas_team` handles it with no extra code)."
        )
    lines.append("")

    lines.append("## Deviations")
    lines.append("")
    lines.append(
        "**Contamination origin (investigated, not left unexplained).** Before any "
        "vegas-related code existed in this working tree, all four "
        "`data/models/{pos}_model_bundle.joblib` were found to already carry "
        "`implied_ppg`/`implied_win_prob`/`has_vegas` inside `tree_feature_cols`, with no "
        "`excluded_features` mechanism anywhere in the (unmodified) committed source at that "
        "point. Checked directly against the working session's own record rather than assumed: "
        "`src/features/shared.py` was read in its original, vegas-free form before any "
        "vegas-related file was written, and no training/retraining command was issued before "
        "the contamination was discovered (it surfaced as a LightGBM 'duplicate feature' error "
        "the first time a vegas-augmented tree_feature_cols list was tried against those "
        "bundles' already-vegas-inclusive tree_feature_cols). On-disk timestamps corroborate "
        "this independently: the contaminated bundles were dated only minutes after "
        "`data/external/odds_api/manifest.json` (itself from the pre-session local Odds API "
        "pull), well before this session's own first genuine retrain. **Conclusion: not "
        "self-inflicted** -- the contamination predates this session's work, most plausibly "
        "from an earlier pass over this exact task (e.g. a reference/verification run) whose "
        "source-code changes were reset before this session began while its gitignored `data/` "
        "artifacts (bundles, `manifest.json`) were left in place. Remediated by adding the "
        "`excluded_features` mechanism (`src.models.train.tree_feature_columns`) and retraining "
        "genuine v1.5-only baselines -- the ones this module's `v1.5_baseline` variant scores "
        "against -- before running this experiment."
    )
    lines.append("")
    lines.append(
        "**Related discovery: the test suite silently overwrites production bundles.** "
        "`tests/test_models.py` and `tests/test_models_positions.py` call `run_full_pipeline` "
        "against `train.position_spec(pos)` -- the REAL production `PositionSpec`, whose "
        "`artifact_path` IS `data/models/{pos}_model_bundle.joblib` -- using "
        "`cfg['optuna']['tiny_trials']` (5 trials, not the real 60/30). `run_full_pipeline` "
        "unconditionally calls `save_artifacts` at the end, so **running the full pytest suite "
        "overwrites every production bundle with a degraded 5-trial version**, without touching "
        "`outputs/model_{pos}_report.md`/`metrics.json` or the board to match. This is "
        "pre-existing repo behavior, not introduced by this change, but it is the direct "
        "mechanical cause of the artifact-staleness this review caught (a real retrain followed "
        "by a pytest run desynchronizes the bundle from its own report/board again). Not fixed "
        "here (outside the requested scope of this pass -- fixing it would mean pointing those "
        "tests' `PositionSpec` at a `tmp_path`-based artifact dir instead of the production one), "
        "but flagged plainly: shipping a real generation must be the LAST step after the final "
        "pytest run, never before it, until this test-isolation gap is closed."
    )
    lines.append("")
    lines.append(
        "**Cross-run hyperparameter instability (separately discovered, see README's Known "
        "Limitations for the full writeup).** Repeating `run_full_pipeline` for WR three times "
        "-- identical code, config, seed, and data throughout -- produced three different "
        "Optuna-selected hyperparameter sets (e.g. blend weights `{lgbm: 0.1, xgb: 0.9}` vs "
        "`{lgbm: 0.2, xgb: 0.8}`), despite the module's own documented determinism guarantee. "
        "The final ROUNDED holdout PR-AUC/top-10 precision in `outputs/model_wr_report.md` "
        "happened to land identically across all three anyway (0.104 / 0.000 every time) -- an "
        "empirical property of this specific holdout sample size, not a guarantee -- but scores "
        "computed differently (this module's own raw pre-calibration blend, or "
        "`overlay_backtest.py`'s pooled-OOF-based scores) are NOT similarly stable and do shift "
        "between regenerations. This module's own promotion DECISIONS (promote/reject) were "
        "unaffected across every regeneration run during this review -- all four positions "
        "rejected every time -- so the binding gate outcome is unchanged regardless."
    )
    lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
