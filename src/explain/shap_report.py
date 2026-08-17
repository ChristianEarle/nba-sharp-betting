"""SHAP explainability (Phase 5).

Two things this module does, both keyed off the frozen bundles Phase 4
saved (``data/models/{pos}_model_bundle.joblib``):

1. **Global beeswarm** (``global_shap_plot``): for each position, one
   ``outputs/shap_{pos}.png`` with a beeswarm panel per *tree* model in the
   classifier blend that carries a nonzero weight (LGBM and/or XGB — every
   position's frozen ``blend_weights`` gives the logistic head weight 0,
   see "Why the logistic head is skipped" below), computed on the pooled
   validation rows (``VALIDATION_SEASONS``, ``in_training_pool==1`` — the
   exact universe ``src.models.train`` tunes and calibrates against, not
   the holdout years and not 2026).
2. **Per-player attribution** (``player_shap_attribution`` /
   ``player_card``): the top-5 features driving one (season, gsis_id)
   row's score, blend-weighted across every nonzero-weight tree model the
   same way ``src.models.train.apply_blend`` combines their predicted
   probabilities — ``shap_value(blend) := sum_m weight_m * shap_value(m)``.
   Exposed as a CLI: ``uv run python -m src.explain.shap_report --why
   "Player Name" [--season 2026]``.

Why the logistic head is skipped
---------------------------------
``shap.TreeExplainer`` doesn't apply to a linear model the way it does to
LGBM/XGB (a ``LinearExplainer`` would be the right tool), and every
position's frozen blend gives the logistic head weight 0 as of this
write-up (see any ``outputs/model_{pos}_metrics.json``'s
``blend_weights``) — so its SHAP contribution would be multiplied by zero
either way. This module skips it entirely rather than building an unused
code path; if a future retrain ever gives the logistic head nonzero
weight, ``nonzero_tree_models`` will raise (or under-count) rather than
silently mis-attribute, and a ``LinearExplainer`` branch would need adding
here.

Margin space, not calibrated-probability space
-------------------------------------------------
``shap.TreeExplainer(model)`` (no background dataset, "tree_path_dependent"
perturbation — the fast, standard default) explains each tree model's own
raw margin/log-odds output, not the post-isotonic/Platt calibrated
probability the board reports. Blend-weighting each model's margin-space
SHAP vector the way the classifier blend itself is weighted is a
reasonable, cheap approximation of "why the blended score moved" — but the
calibrator on top is a monotone, *nonlinear* transform that SHAP values do
not pass through exactly. Every driver reported here answers "which
features push this player's score up or down and by how much, relative to
each other" (a ranking), not "this feature is worth exactly N percentage
points of calibrated probability." Documented rather than silently implied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models import train
from src.models.cv import VALIDATION_SEASONS

REPO_ROOT = train.REPO_ROOT
OUTPUTS_DIR = train.OUTPUTS_DIR

# Bundle keys that are actual fitted tree models (matches blend_weights' keys
# minus "logistic" — see module docstring for why logistic is excluded).
TREE_MODEL_KEYS = ("lgbm", "xgb")
TOP_N_DRIVERS = 5


# --------------------------------------------------------------------------
# Bundle + data loading
# --------------------------------------------------------------------------


def load_bundle(pos: str) -> dict:
    spec = train.position_spec(pos)
    if not spec.artifact_path.exists():
        raise FileNotFoundError(f"{spec.artifact_path} not found; run `python -m src.models.train {pos}` first")
    return joblib.load(spec.artifact_path)


def nonzero_tree_models(bundle: dict) -> dict[str, float]:
    """{model_key: weight} for every TREE_MODEL_KEYS entry with a nonzero blend weight."""
    weights = bundle["blend_weights"]
    out = {k: w for k, w in weights.items() if k in TREE_MODEL_KEYS and w > 0}
    if not out:
        raise ValueError(f"no tree model has nonzero blend weight in this bundle: {weights}")
    return out


def pooled_validation_frame(pos: str, bundle: dict) -> pd.DataFrame:
    """The exact universe Phase 4 tunes/calibrates against: in_training_pool==1 rows

    restricted to VALIDATION_SEASONS (2020-2023) — never the holdout years, never 2026.
    """
    spec = train.position_spec(pos)
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
    return df[df["season"].isin(VALIDATION_SEASONS)].reset_index(drop=True)


# --------------------------------------------------------------------------
# SHAP values
# --------------------------------------------------------------------------


def raw_shap_values(bundle: dict, model_key: str, X: pd.DataFrame) -> np.ndarray:
    """(n_rows, n_features) SHAP matrix for one tree model, positive-class / margin space.

    Defensively unwraps the list-of-arrays shape some shap/lightgbm version
    combinations return for a binary classifier (``[shap_class0, shap_class1]``)
    down to the positive-class array; current pinned versions return the
    plain array directly (verified against this repo's shap==0.51,
    lightgbm==4.x), but this guards against a future upgrade silently
    reversing that.
    """
    model = bundle[model_key]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]  # positive class
    return np.asarray(sv)


def blend_weighted_shap(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """(n_rows, n_features) SHAP matrix, blend-weighted across every nonzero-weight tree model.

    ``shap_value(blend) := sum_m weight_m * shap_value(m)`` — the same
    linear combination ``src.models.train.apply_blend`` uses on each
    model's predicted probability (see module docstring for the
    margin-space caveat).
    """
    weights = nonzero_tree_models(bundle)
    total = None
    for key, w in weights.items():
        sv = raw_shap_values(bundle, key, X) * w
        total = sv if total is None else total + sv
    return total


def global_mean_abs_importance(bundle: dict, X: pd.DataFrame, tree_cols: list[str], top_n: int = TOP_N_DRIVERS) -> pd.DataFrame:
    """Blend-weighted mean(|SHAP|) per feature, sorted descending — the "global top-N" table."""
    sv = blend_weighted_shap(bundle, X)
    mean_abs = np.abs(sv).mean(axis=0)
    out = pd.DataFrame({"feature": tree_cols, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    return out.head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------
# Global beeswarm (outputs/shap_{pos}.png)
# --------------------------------------------------------------------------


def global_shap_plot(pos: str, out_path: Path | None = None) -> pd.DataFrame:
    """Draw + save outputs/shap_{pos}.png (one beeswarm panel per nonzero-weight tree model).

    Returns the blend-weighted global top-5 importance table (also useful
    standalone, e.g. for a written report), so callers don't have to
    recompute SHAP values a second time to get it.
    """
    bundle = load_bundle(pos)
    tree_cols = bundle["tree_feature_cols"]
    val_df = pooled_validation_frame(pos, bundle)
    X = val_df[tree_cols]

    weights = nonzero_tree_models(bundle)
    n = len(weights)
    fig, axes = plt.subplots(n, 1, figsize=(9, 4.6 * n))
    axes = [axes] if n == 1 else list(axes)

    for ax, (key, w) in zip(axes, weights.items()):
        sv = raw_shap_values(bundle, key, X)
        plt.sca(ax)
        shap.summary_plot(sv, X, show=False, plot_size=None)
        ax.set_title(f"{pos.upper()} — {key} (blend weight {w:.2f}), pooled validation {VALIDATION_SEASONS[0]}-{VALIDATION_SEASONS[-1]}")
    fig.tight_layout()

    out_path = out_path or (OUTPUTS_DIR / f"shap_{pos}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    return global_mean_abs_importance(bundle, X, tree_cols)


# --------------------------------------------------------------------------
# Per-player attribution
# --------------------------------------------------------------------------


def player_shap_attribution(
    bundle: dict, row_df: pd.DataFrame, top_n: int = TOP_N_DRIVERS
) -> list[dict]:
    """Top-N blend-weighted SHAP drivers for the single row in ``row_df`` (a 1-row frame

    carrying every ``bundle["tree_feature_cols"]`` column). Each entry:
    {feature, shap_value (signed, margin space), raw_value (the feature's
    actual value for this row)}, sorted by |shap_value| descending.
    """
    tree_cols = bundle["tree_feature_cols"]
    X = row_df[tree_cols]
    sv = blend_weighted_shap(bundle, X)[0]
    order = np.argsort(-np.abs(sv))[:top_n]
    return [
        {"feature": tree_cols[i], "shap_value": float(sv[i]), "raw_value": row_df[tree_cols[i]].iloc[0]}
        for i in order
    ]


def compact_drivers(drivers: list[dict], n: int = 3) -> str:
    """"feature:+/-" for the top ``n`` drivers, e.g. "target_share_n1:+, age:-, new_hc:+" —

    the compact board column. Sign only (direction), not magnitude — the board's
    "rationale" line carries the plain-language version.
    """
    parts = []
    for d in drivers[:n]:
        sign = "+" if d["shap_value"] >= 0 else "-"
        parts.append(f"{d['feature']}:{sign}")
    return ", ".join(parts)


def regression_prediction(bundle: dict, row_df: pd.DataFrame) -> float:
    """Expected finish-rank delta: unweighted mean of lgbm_reg/xgb_reg's predictions.

    No blend-weight search ever ran for the regression head (``src.models.train``
    reports RMSE for both and stops there, unlike the classifier trio's
    grid-searched blend) — a plain average of the two is the simplest
    documented choice, not a tuned one, in keeping with this pipeline's
    "simple and stated, not tuned" overlay conventions elsewhere.
    """
    tree_cols = bundle["tree_feature_cols"]
    X = row_df[tree_cols]
    preds = [bundle["lgbm_reg"].predict(X)[0], bundle["xgb_reg"].predict(X)[0]]
    return float(np.mean(preds))


def classifier_probability(bundle: dict, row_df: pd.DataFrame) -> float:
    """Blend + calibrate one row through the frozen classifier trio (only nonzero-weight

    models are actually evaluated; see nonzero_tree_models). Mirrors
    src.models.train.holdout_predictions' scoring path exactly, one row at a time.
    """
    weights = bundle["blend_weights"]
    tree_cols = bundle["tree_feature_cols"]
    preds = {}
    if weights.get("lgbm", 0) > 0:
        preds["lgbm"] = bundle["lgbm"].predict_proba(row_df[tree_cols])[:, 1]
    if weights.get("xgb", 0) > 0:
        preds["xgb"] = bundle["xgb"].predict_proba(row_df[tree_cols])[:, 1]
    if weights.get("logistic", 0) > 0:
        model, imputer, scaler = bundle["logistic"]
        logit_cols = bundle["logistic_feature_cols"]
        X = scaler.transform(imputer.transform(row_df[logit_cols]))
        preds["logistic"] = model.predict_proba(X)[:, 1]
    nonzero_weights = {k: w for k, w in weights.items() if k in preds}
    blended = train.apply_blend(nonzero_weights, **preds)
    calibrated = train.apply_calibration(bundle["calibration_method"], bundle["calibrator"], blended)
    return float(calibrated[0])


# --------------------------------------------------------------------------
# --why CLI
# --------------------------------------------------------------------------


def _find_historical_row(name: str, season: int | None) -> tuple[str, pd.DataFrame] | None:
    """Search every position's features_{pos}.parquet for a name match; returns

    (position, one-row frame carrying tree_feature_cols + identity) or None.
    """
    for pos in train.POSITIONS:
        spec = train.position_spec(pos)
        if not spec.features_path.exists() or not spec.labels_path.exists():
            continue
        bundle = load_bundle(pos) if spec.artifact_path.exists() else None
        if bundle is None:
            continue
        df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
        matches = df[df["player_name"].str.lower() == name.lower()]
        if season is not None:
            matches = matches[matches["season"] == season]
        if matches.empty:
            continue
        row = matches.sort_values("season").tail(1)
        return pos, row
    return None


def _find_2026_row(name: str) -> tuple[str, pd.DataFrame, dict] | None:
    """Look up a 2026 row from src.inference.board_2026's saved feature matrices.

    Deferred import: src.inference.board_2026 imports this module for
    blend_weighted_shap/compact_drivers, so importing it back at module
    top-level here would be circular. Season-2026 lookups need that
    module's saved-matrix paths, so it's imported lazily, only when asked.
    """
    from src.inference import board_2026

    for pos in train.POSITIONS:
        path = board_2026.veteran_features_path(pos)
        if not path.exists():
            continue
        import polars as pl

        df = pl.read_parquet(path).to_pandas()
        matches = df[df["player_name"].str.lower() == name.lower()]
        if matches.empty:
            continue
        bundle = load_bundle(pos)
        return pos, matches.head(1), bundle
    return None


def print_player_card(name: str, season: int | None) -> int:
    if season == 2026:
        found = _find_2026_row(name)
        if found is None:
            print(f'"{name}" not found in any 2026 veteran feature matrix (rookies use src.models.rookie_heuristic, not this card).')
            return 1
        pos, row, bundle = found
    else:
        found = _find_historical_row(name, season)
        if found is None:
            print(f'"{name}" not found in any position\'s features_{{pos}}.parquet' + (f" for season {season}" if season else ""))
            return 1
        pos, row = found
        bundle = load_bundle(pos)

    prob = classifier_probability(bundle, row)
    rank_delta = regression_prediction(bundle, row)
    drivers = player_shap_attribution(bundle, row)

    r = row.iloc[0]
    print(f"\n{r['player_name']} ({pos.upper()}, {int(r['season'])})")
    print(f"  calibrated breakout probability: {prob:.3f}")
    print(f"  expected finish-rank delta:      {rank_delta:+.1f}")
    if "expectation_pos_rank" in row.columns:
        print(f"  preseason expectation pos rank:  {r['expectation_pos_rank']:.0f}" if pd.notna(r.get("expectation_pos_rank")) else "")
    print(f"  top-{len(drivers)} SHAP drivers (margin-space blend, see module docstring):")
    for d in drivers:
        sign = "pushes UP" if d["shap_value"] >= 0 else "pushes DOWN"
        print(f"    {d['feature']:<28} {sign:<11} shap={d['shap_value']:+.4f}  raw_value={d['raw_value']}")
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="SHAP explainability (Phase 5)")
    ap.add_argument("--why", metavar="PLAYER_NAME", help="print the full explanation card for one player")
    ap.add_argument("--season", type=int, default=None, help="season to look up (default: player's latest available)")
    ap.add_argument("--pos", nargs="*", default=list(train.POSITIONS), help="restrict global-plot generation to these positions")
    args = ap.parse_args()

    if args.why:
        return print_player_card(args.why, args.season)

    for pos in args.pos:
        spec = train.position_spec(pos)
        if not spec.artifact_path.exists():
            print(f"{pos}: no bundle at {spec.artifact_path}, skipping")
            continue
        top5 = global_shap_plot(pos)
        print(f"{pos.upper()}: wrote {OUTPUTS_DIR / f'shap_{pos}.png'}")
        print(top5.to_string(index=False))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
