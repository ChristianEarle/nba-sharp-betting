"""Vegas-feature ablation (v1.6): does the preseason team-line signal help?

A controlled A/B over the repo's own CV structure: for each position, fit
the SAME LightGBM classifier (identical fixed hyperparameters, identical
seed, identical early-stopping procedure) on the four expanding-window
validation folds (``src.models.cv``) and the one holdout split, twice --
once with every feature column except the ``vegas_*`` set (BASE), once
with them (VEGAS). The only difference between arms is the five columns,
so any metric gap is attributable to the features rather than to tuning
noise -- deliberately NOT a re-run of the full Optuna pipeline per arm,
where 60-trial search variance on ~1.2k-row/3-6%-positive samples would
swamp a small feature effect.

Hyperparameters are fixed at the midpoint of ``configs/model_wr.yaml``'s
search space (the same bounds every position tunes within), not tuned on
either arm. Holdout retrains (<=2023) fix ``n_estimators`` at the mean
best_iteration of that arm's four fold fits, mirroring
``src.models.train``'s no-early-stopping-at-holdout rule.

Interpretation guardrails, stated up front:

- Vegas coverage starts 2020, so fold train sets carry 0 (val 2020) to 3
  (val 2023) seasons of non-null vegas rows; early folds mostly measure
  "did 5 null columns hurt". The holdout split (train 2014-2023, 4 vegas
  seasons; eval 2024+2025, full coverage) is the fairest read.
- RB/TE/QB validation folds can have single-digit positives; per-fold
  PR-AUC swings there are noise, per the small-sample caveat in
  ``src.models.train``'s docstring. WR is the highest-powered read.

Writes ``outputs/ablation_vegas.md`` (gitignored, like every other
regenerable report) and prints the same tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import metrics as mt
from src.models.cv import holdout_split, validation_folds
from src.models.train import (
    fit_lgbm_classifier,
    fold_frames,
    load_config,
    load_modeling_frame,
    position_spec,
    tree_feature_columns,
    _model_best_iteration,
)
import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "outputs" / "ablation_vegas.md"

VEGAS_PREFIX = "vegas_"
SEED = 42

# Midpoint of configs/model_wr.yaml's lgbm search space (shared bounds
# across positions); n_estimators at the top of the range because early
# stopping against each fold's own validation season picks the effective
# count -- both arms get the identical procedure.
FIXED_PARAMS = {
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 50,
    "learning_rate": 0.05,
    "n_estimators": 600,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def run_arm(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Fold PR-AUCs + holdout PR-AUCs for one arm (one feature set)."""
    fold_rows = []
    best_iters = []
    for fold in validation_folds():
        train_df, val_df = fold_frames(df, fold.train_seasons, fold.val_season)
        model = fit_lgbm_classifier(train_df, val_df, feature_cols, FIXED_PARAMS, SEED)
        pred = model.predict_proba(val_df[feature_cols])[:, 1]
        fold_rows.append(
            {
                "val_season": fold.val_season,
                "n_val": len(val_df),
                "positives": int(val_df["breakout"].sum()),
                "pr_auc": mt.pr_auc(val_df["breakout"], pred),
            }
        )
        best_iters.append(_model_best_iteration(model, "lgbm", FIXED_PARAMS["n_estimators"]))

    split = holdout_split()
    train_df = df[df["season"].isin(split.train_seasons)]
    n_est = int(np.mean(best_iters))
    params = {**FIXED_PARAMS, "n_estimators": n_est}
    spw = float((train_df["breakout"] == 0).sum() / max(1, train_df["breakout"].sum()))
    final = lgb.LGBMClassifier(
        objective="binary",
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbosity=-1,
        scale_pos_weight=spw,
        **params,
    )
    final.fit(train_df[feature_cols], train_df["breakout"])

    holdout_rows = []
    for season in split.holdout_seasons:
        hd = df[df["season"] == season]
        pred = final.predict_proba(hd[feature_cols])[:, 1]
        holdout_rows.append(
            {
                "season": season,
                "n": len(hd),
                "positives": int(hd["breakout"].sum()),
                "pr_auc": mt.pr_auc(hd["breakout"], pred),
                "top10_precision": mt.top_k_precision(hd["breakout"].to_numpy(), pred, k=10),
            }
        )

    importances = pd.Series(final.feature_importances_, index=feature_cols)
    return {
        "folds": pd.DataFrame(fold_rows),
        "holdout": pd.DataFrame(holdout_rows),
        "holdout_n_estimators": n_est,
        "importances": importances,
    }


def run_position(pos: str) -> dict:
    spec = position_spec(pos)
    cfg = load_config(spec.config_path)
    df = load_modeling_frame(spec.features_path, spec.labels_path, cfg=cfg)

    all_cols = tree_feature_columns(df)
    vegas_cols = [c for c in all_cols if c.startswith(VEGAS_PREFIX)]
    base_cols = [c for c in all_cols if not c.startswith(VEGAS_PREFIX)]
    assert vegas_cols, (
        f"no vegas_* columns in features_{pos}.parquet -- run src.features.vegas first"
    )

    return {
        "pos": pos,
        "vegas_cols": vegas_cols,
        "base": run_arm(df, base_cols),
        "vegas": run_arm(df, base_cols + vegas_cols),
    }


def render(results: list[dict]) -> str:
    lines = ["# Vegas team-line feature ablation (v1.6)", ""]
    lines.append(
        "Same fixed-hyperparameter LGBM, same folds, same seed in both arms; "
        "the arms differ only in the five `vegas_*` columns. See "
        "`src/models/ablation_vegas.py`'s docstring for interpretation guardrails."
    )
    for r in results:
        pos = r["pos"].upper()
        base, veg = r["base"], r["vegas"]
        lines.append(f"\n## {pos}\n")
        lines.append("| val season | n | pos | PR-AUC base | PR-AUC +vegas | delta |")
        lines.append("|---|---|---|---|---|---|")
        for (_, b), (_, v) in zip(base["folds"].iterrows(), veg["folds"].iterrows()):
            lines.append(
                f"| {int(b.val_season)} | {int(b.n_val)} | {int(b.positives)} "
                f"| {b.pr_auc:.4f} | {v.pr_auc:.4f} | {v.pr_auc - b.pr_auc:+.4f} |"
            )
        mb, mv = base["folds"]["pr_auc"].mean(), veg["folds"]["pr_auc"].mean()
        lines.append(f"| **mean** | | | **{mb:.4f}** | **{mv:.4f}** | **{mv - mb:+.4f}** |")

        lines.append("\n| holdout | n | pos | PR-AUC base | PR-AUC +vegas | delta | top10 base | top10 +vegas |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for (_, b), (_, v) in zip(base["holdout"].iterrows(), veg["holdout"].iterrows()):
            lines.append(
                f"| {int(b.season)} | {int(b.n)} | {int(b.positives)} | {b.pr_auc:.4f} "
                f"| {v.pr_auc:.4f} | {v.pr_auc - b.pr_auc:+.4f} "
                f"| {b.top10_precision:.2f} | {v.top10_precision:.2f} |"
            )

        imp = veg["importances"]
        vegas_imp = imp[imp.index.str.startswith(VEGAS_PREFIX)].sort_values(ascending=False)
        total = imp.sum()
        lines.append("\nVegas-arm split importance (share of all splits):")
        for name, val in vegas_imp.items():
            lines.append(f"- `{name}`: {val} splits ({val / total:.1%})")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    results = [run_position(p) for p in ("wr", "rb", "te", "qb")]
    report = render(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
