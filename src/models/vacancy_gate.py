"""v2.4 -- team-change / vacancy capacity gates (this session's brief).

Four new roster-diff-derived features (``src.features.shared``, wired into
``src.features.{wr,rb,te,qb}`` -- see each module's v2.4 docstring paragraph) go through
TWO binding, pre-stated, mechanical gates before shipping in any position's actual model:

- ``qb_continuity`` (WR/TE/RB rows only -- meaningless for the thrower himself): fraction
  of the player's season-N team's N-1 pass attempts thrown by QBs still on that team's
  Week-1 season-N roster. A new-starting-QB situation reads low.
- ``vacated_td_share`` (every position's team context, including QB): share of the team's
  N-1 offensive TDs (rushing + receiving) belonging to players absent from the Week-1
  season-N roster.
- ``vacated_goal_line_carry_share`` (RB primarily, cheap enough to add to WR/TE too):
  departed players' share of the team's N-1 goal-line (yardline_100<=5) carries.
- ``max_single_vacated_target_share`` / ``max_single_vacated_carry_share`` (WR/TE/RB): the
  single largest departed player's own share -- star departure vs diffuse churn, a signal
  ``vacated_shares``' population SUM alone can't distinguish.

All four are purely roster-diff-derived (``season_roster_team`` vs N-1 usage) -- no manual
per-team lists anywhere, per the brief's explicit requirement.

Part A -- CLASSIFIER gate
---------------------------
Identical mechanism to ``src.models.efficiency_gate``'s v2.2 gate (frozen-hyperparameter
retrain, same "score both variants on the identical holdout rows" structure, same v1.7
seed-ensemble-aware promotion) -- this module does not re-implement any of that engine, it
calls straight into ``efficiency_gate``'s generic (``extra_cols``-parameterized) pieces:
``score_incumbent``, ``fit_and_score_efficiency_variant``, ``reconstruct_incumbent_score``,
``bundle_already_promoted``, ``promote_position``, ``add_to_excluded_features``, and the
now-parameterized ``metrics_table``/``gate_decision``/``write_gate_outputs`` (variant_name=
"plus_vacancy" instead of the v2.2 default "plus_efficiency"). Only ``NEW_VACANCY_COLUMNS``
(the candidate column list per position, including QB -- v2.2's gate never covered QB) and
this module's own report/output naming are new.

Gate rule (binding, identical to v2.2): KEEP iff pooled (2024+2025) holdout top-10
precision does NOT regress AND PR-AUC either improves or regresses by at most
``efficiency_gate.PR_AUC_TOLERANCE`` (0.01). REJECT -> ``configs/model_{pos}.yaml``'s
``excluded_features`` (the identical list the classifier's ``tree_feature_columns`` reads).

Part B -- QUANTILE gate (new this session, same spirit)
------------------------------------------------------------
No prior module gates the quantile head -- this is the exclusion mechanism's first use.
Frozen-hyperparameter analogue of Part A for ``src.models.quantile``: the tuned-once-at-
alpha=0.50 LightGBM hyperparameters (``bundle["params"]``) stay FROZEN; each alpha's model
is refit on <=2023 with ``tree_feature_cols + extra_cols``, its own ``n_estimators``
re-derived via that alpha's own early stopping (``quantile.alpha_oof``) -- exactly how
``efficiency_gate``'s classifier variant freezes tree-shape hyperparameters but still lets
n_estimators follow honestly from the new column set. The incumbent side needs no refit at
all: the shipped bundle's own already-fitted models score the holdout rows directly
(``quantile.score_quantiles``).

Gate rule (binding, new): KEEP iff holdout Spearman(q50, realized ppg) does not regress by
more than ``QUANTILE_SPEARMAN_TOLERANCE`` (0.01) AND pinball loss at q50 improves or stays
within ``QUANTILE_PINBALL_TOLERANCE`` (1%) of the incumbent's. Coverage (q10-q90) delta is
reported alongside but does not gate. REJECT -> ``configs/model_{pos}.yaml``'s
``excluded_features_quantile`` (v2.4's new key -- see ``quantile.tree_feature_columns``'s
docstring for why the quantile head needed its own exclusion list rather than sharing the
classifier's: the two gates can legitimately disagree on the same column).

Disposition (both gates)
---------------------------
- KEPT: the frozen-retrain-with-new-columns bundle becomes the position's shipped bundle;
  ``outputs/model_{pos}_report.md``/``metrics.json`` (classifier) or
  ``outputs/model_{pos}_quantile_report.md``/``metrics.json`` (quantile) are regenerated to
  describe it, banner-labeled so nobody mistakes either for a fresh full retrain.
- REJECTED: the columns stay in ``features_{pos}.parquet`` (nullable, auditable) but are
  added to the relevant exclusion list, so a future real retrain doesn't silently pick them
  up.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.models import efficiency_gate as eg
from src.models import metrics as mt
from src.models import quantile as qmod
from src.models import train

CLASSIFIER_OUT_PATH = train.OUTPUTS_DIR / "vacancy_gate.md"
QUANTILE_OUT_PATH = train.OUTPUTS_DIR / "vacancy_gate_quantile.md"

# Every v2.4 vacancy/continuity candidate column, by position -- mirrors
# src.features.shared.VACANCY_COLUMNS_WR_TE_RB / VACANCY_COLUMNS_QB exactly (imported
# lazily inside vacancy_columns_for to avoid a hard src.features import at module load
# time, matching efficiency_gate's own lazy-import-free style elsewhere in this repo).
NEW_VACANCY_COLUMNS: dict[str, tuple[str, ...]] = {
    "wr": (
        "qb_continuity",
        "vacated_td_share",
        "vacated_goal_line_carry_share",
        "max_single_vacated_target_share",
        "max_single_vacated_carry_share",
    ),
    "te": (
        "qb_continuity",
        "vacated_td_share",
        "vacated_goal_line_carry_share",
        "max_single_vacated_target_share",
        "max_single_vacated_carry_share",
    ),
    "rb": (
        "qb_continuity",
        "vacated_td_share",
        "vacated_goal_line_carry_share",
        "max_single_vacated_target_share",
        "max_single_vacated_carry_share",
    ),
    "qb": ("vacated_td_share",),
}

QUANTILE_SPEARMAN_TOLERANCE = 0.01
QUANTILE_PINBALL_TOLERANCE = 0.01  # 1%


def vacancy_columns_for(pos: str) -> list[str]:
    """The candidate v2.4 column list for one position -- NOT suffix-expanded (unlike

    ``efficiency_gate.new_columns_for``): these are already-final column names
    (``qb_continuity``, ``vacated_td_share``, ...), not ``BASE_METRICS`` entries that get
    an ``_n1``/``_yoy_delta`` pair. Both gate engines below (classifier via
    ``efficiency_gate``'s generic ``extra_cols``-taking functions, quantile via this
    module's own) treat ``extra_cols`` as opaque column names either way, so no other code
    needed to change to accept this different shape.
    """
    return list(NEW_VACANCY_COLUMNS.get(pos, ()))


_VACANCY_BANNER_TEMPLATE = """> **v2.4 vacancy gate -- frozen-hyperparameter retrain.** This report/bundle was
> regenerated by `src.models.vacancy_gate` (not a fresh `python -m src.models.train
> {pos}` Optuna run): every hyperparameter and blend weight below is FROZEN at the
> pre-v2.4 incumbent bundle's values, per seed -- the only thing that changed is the tree
> feature set, which now includes {extra_cols}. Promoted because pooled (2024+2025)
> holdout top-10 precision did not regress ({base_top10:.3f} -> {var_top10:.3f}) and
> PR-AUC {prauc_verb} ({base_prauc:.3f} -> {var_prauc:.3f}) -- see
> `outputs/vacancy_gate.md` for the full per-position gate table. The numbers in the rest
> of this report describe the SHIPPED bundle at `{artifact_path}` as of this promotion,
> computed the identical way every other position's report is (this position's own
> frozen-hyperparameter holdout retrain, scored once).
"""


# --------------------------------------------------------------------------
# Part A -- CLASSIFIER gate (delegates to efficiency_gate's generic engine)
# --------------------------------------------------------------------------


def run_vacancy_position(pos: str) -> dict | None:
    """The classifier vacancy gate for one position -- thin wrapper around

    ``efficiency_gate._run_gate_position`` with this module's own column list, variant
    name ("plus_vacancy"), and report banner. See this module's docstring's Part A.
    """
    return eg._run_gate_position(
        pos,
        extra_cols=vacancy_columns_for(pos),
        variant_name="plus_vacancy",
        banner_template=_VACANCY_BANNER_TEMPLATE,
        title_marker="(v2.4 vacancy gate, frozen-hyperparameter retrain)",
        metrics_key="v2_4_vacancy_gate",
        metrics_note="frozen-hyperparameter retrain via src.models.vacancy_gate -- see outputs/vacancy_gate.md",
    )


# --------------------------------------------------------------------------
# Part B -- QUANTILE gate (new this session)
# --------------------------------------------------------------------------


def fit_and_score_quantile_variant(
    bundle: dict, df: pd.DataFrame, extra_cols: list[str], folds: list, seed: int
) -> tuple[dict, dict]:
    """Refit every alpha's LightGBM at the FROZEN alpha=0.50-tuned hyperparameters

    (``bundle["params"]``) on <=2023 with ``tree_feature_cols + extra_cols``, that alpha's
    own ``n_estimators`` re-derived via its own early stopping (``quantile.alpha_oof`` --
    exactly how ``efficiency_gate``'s classifier variant freezes tree-shape but still lets
    n_estimators follow honestly from the new columns). Returns (holdout_metrics dict,
    {"tree_cols", "final_n_estimators", "models", "per_alpha_oof", "per_alpha_fold_metrics"}
    -- everything a KEEP decision's ``promote_quantile_position`` needs without refitting
    a third time).

    Idempotency guard (same as ``efficiency_gate.fit_and_score_efficiency_variant``): a
    bundle this gate already KEPT carries ``extra_cols`` in ``tree_feature_cols`` already --
    only columns not already present are appended, so re-running never hands LightGBM a
    duplicate-named feature.
    """
    tree_cols = list(bundle["tree_feature_cols"]) + [c for c in extra_cols if c not in bundle["tree_feature_cols"]]
    params = bundle["params"]

    per_alpha_oof: dict[float, pd.DataFrame] = {}
    per_alpha_fold_metrics: dict[float, list[dict]] = {}
    final_n_estimators: dict[float, int] = {}
    for a in qmod.ALPHAS:
        pooled, fold_metrics, best_iters = qmod.alpha_oof(df, tree_cols, folds, params, a, seed)
        per_alpha_oof[a] = pooled
        per_alpha_fold_metrics[a] = fold_metrics
        final_n_estimators[a] = int(round(np.mean(best_iters)))

    new_frozen = dict(bundle)
    new_frozen["tree_feature_cols"] = tree_cols
    new_frozen["final_n_estimators"] = final_n_estimators
    holdout_wide, holdout_models = qmod.holdout_predictions(df, seed, new_frozen)
    holdout_metrics = qmod.distribution_metrics(holdout_wide)

    fitted = {
        "tree_cols": tree_cols,
        "final_n_estimators": final_n_estimators,
        "models": holdout_models,
        "per_alpha_oof": per_alpha_oof,
        "per_alpha_fold_metrics": per_alpha_fold_metrics,
        "params": params,
    }
    return holdout_metrics, fitted


def quantile_bundle_already_promoted(bundle: dict, extra_cols: list[str]) -> bool:
    """True iff every one of this position's v2.4 columns is already in the quantile

    bundle's own ``tree_feature_cols`` -- i.e. a prior run of this gate already KEPT it.
    Same purpose as ``efficiency_gate.bundle_already_promoted``, quantile-bundle-shaped.
    """
    return bool(extra_cols) and all(c in bundle["tree_feature_cols"] for c in extra_cols)


def reconstruct_incumbent_quantile_metrics(bundle: dict, df: pd.DataFrame, extra_cols: list[str], folds: list, seed: int) -> dict:
    """The TRUE pre-promotion incumbent's holdout distribution metrics, reconstructed from

    a quantile bundle that has ALREADY been promoted -- the exact quantile-bundle-shaped
    analogue of ``efficiency_gate.reconstruct_incumbent_score`` (see that function's
    docstring for why re-running against an already-promoted bundle must never compare it
    against itself). Every hyperparameter/params value stays frozen across a promotion
    (only tree_feature_cols/final_n_estimators change), so refitting at the SAME frozen
    params over ``tree_feature_cols`` with ``extra_cols`` stripped back out is
    deterministically identical to what the original incumbent would have produced.
    """
    reduced_tree_cols = [c for c in bundle["tree_feature_cols"] if c not in extra_cols]
    reduced_frozen = dict(bundle)
    reduced_frozen["tree_feature_cols"] = reduced_tree_cols
    final_n_estimators = {}
    for a in qmod.ALPHAS:
        _, _, best_iters = qmod.alpha_oof(df, reduced_tree_cols, folds, bundle["params"], a, seed)
        final_n_estimators[a] = int(round(np.mean(best_iters)))
    reduced_frozen["final_n_estimators"] = final_n_estimators
    holdout_wide, _ = qmod.holdout_predictions(df, seed, reduced_frozen)
    return qmod.distribution_metrics(holdout_wide)


def quantile_gate_decision(incumbent_metrics: dict, variant_metrics: dict) -> tuple[bool, str]:
    """KEEP iff holdout Spearman(q50, realized) does not regress by more than

    QUANTILE_SPEARMAN_TOLERANCE AND pinball(q50) improves or stays within
    QUANTILE_PINBALL_TOLERANCE (relative) of the incumbent's. Coverage (q10-q90) reported
    for both but never gates -- the brief asks for it reported, not enforced (a shift in
    interval width/placement is not on its own evidence of a worse model the way a
    ranking/loss regression is). Returns (keep: bool, reason: str).
    """
    base_spearman = incumbent_metrics["spearman_q50_actual"]
    var_spearman = variant_metrics["spearman_q50_actual"]
    base_pinball50 = incumbent_metrics["pinball_by_alpha"]["0.50"]
    var_pinball50 = variant_metrics["pinball_by_alpha"]["0.50"]
    base_cov = incumbent_metrics["coverage_q10_q90"]
    var_cov = variant_metrics["coverage_q10_q90"]

    spearman_ok = var_spearman >= base_spearman - QUANTILE_SPEARMAN_TOLERANCE
    pinball_ok = var_pinball50 <= base_pinball50 * (1 + QUANTILE_PINBALL_TOLERANCE)
    keep = bool(spearman_ok and pinball_ok)

    reason = (
        f"Spearman(q50,actual) {base_spearman:.3f} -> {var_spearman:.3f} "
        f"({'ok, no regression beyond tolerance' if spearman_ok else f'REGRESSED by more than {QUANTILE_SPEARMAN_TOLERANCE} -- reject'}); "
        f"pinball(q50) {base_pinball50:.4f} -> {var_pinball50:.4f} "
        f"({'ok' if pinball_ok else f'REGRESSED by more than {QUANTILE_PINBALL_TOLERANCE * 100:.0f}% -- reject'}); "
        f"coverage(q10-q90) {base_cov:.3f} -> {var_cov:.3f} (reported only, not gating)"
    )
    return keep, ("KEEP: " if keep else "REJECT: ") + reason


def promote_quantile_position(qspec: qmod.QuantileSpec, bundle: dict, fitted: dict, pos: str, extra_cols: list[str], reason: str) -> dict:
    """Rebuild ``qspec.artifact_path`` with this position's new vacancy columns baked in

    at every incumbent-frozen hyperparameter, then regenerate
    ``outputs/model_{pos}_quantile_{report.md,metrics.json}`` via
    ``quantile.write_report``/``write_metrics_json`` (the SAME machinery a real
    ``run_full_pipeline`` call uses) with a v2.4 banner spliced in -- mirrors
    ``efficiency_gate.promote_position``/``write_gate_outputs``'s "never leave the
    report/metrics describing the pre-promotion bundle" discipline for the quantile
    artifacts too.
    """
    seed = bundle["seed"]
    cfg = bundle["cfg"]
    new_frozen = dict(bundle)
    new_frozen["tree_feature_cols"] = fitted["tree_cols"]
    new_frozen["final_n_estimators"] = fitted["final_n_estimators"]
    qmod.save_artifacts(qspec, new_frozen, fitted["models"], seed, cfg)

    val_wide = qmod._pool_alphas(fitted["per_alpha_oof"])
    val_metrics = qmod.distribution_metrics(val_wide)
    holdout_wide, _ = qmod.holdout_predictions(
        qmod.load_modeling_frame(qspec.features_path, qspec.labels_path, cfg=cfg), seed, new_frozen
    )
    holdout_metrics = qmod.distribution_metrics(holdout_wide)

    qmod.write_report(qspec, val_metrics, holdout_metrics, fitted["per_alpha_fold_metrics"], fitted["params"], seed, n_trials=0)
    report_md = qspec.report_path.read_text()
    title_line, _, rest = report_md.partition("\n")
    title_line = title_line.replace("(v2.0)", "(v2.4 vacancy gate, frozen-hyperparameter retrain)")
    rest = rest.replace(
        f"Tuned once at alpha=0.50 (0 Optuna trials, seed={seed}), reused for all 5 alphas.",
        f"Hyperparameters FROZEN at the pre-v2.4 alpha=0.50 tuning pass (seed={seed}), reused for all 5 alphas -- no new Optuna trials ran this gate.",
    )
    banner = (
        "\n> **v2.4 vacancy gate -- frozen-hyperparameter retrain.** This report/bundle was\n"
        "> regenerated by `src.models.vacancy_gate` (not a fresh Optuna tuning pass): the\n"
        f"> alpha=0.50-tuned hyperparameters below are FROZEN at the pre-v2.4 incumbent's\n"
        f"> values -- the only thing that changed is the tree feature set, which now\n"
        f"> includes {extra_cols}. {reason} See `outputs/vacancy_gate_quantile.md` for the\n"
        "> full per-position gate table.\n"
    )
    qspec.report_path.write_text(f"{title_line}\n{banner}\n{rest.lstrip(chr(10))}")

    qmod.write_metrics_json(qspec, val_metrics, holdout_metrics, fitted["params"], seed)
    metrics = json.loads(qspec.metrics_json_path.read_text())
    metrics["v2_4_vacancy_gate"] = {
        "promoted": True,
        "new_columns": extra_cols,
        "note": "frozen-hyperparameter retrain via src.models.vacancy_gate -- see outputs/vacancy_gate_quantile.md",
    }
    qspec.metrics_json_path.write_text(json.dumps(metrics, indent=2, default=str))
    return {"holdout_metrics": holdout_metrics, "val_metrics": val_metrics, "frozen": new_frozen}


def add_to_excluded_features_quantile(config_path, new_cols: list[str]) -> None:
    """A REJECTed position's new columns are added to configs/model_{pos}.yaml's

    ``excluded_features_quantile`` list (v2.4's new key -- see
    ``quantile.tree_feature_columns``'s docstring for why this is a SEPARATE list from
    ``excluded_features``, not the classifier gate's shared one). Idempotent: a column
    already listed is not duplicated. Creates the key if the config predates it (every
    real ``configs/model_{pos}.yaml`` already carries an empty ``[]`` placeholder as of
    v2.4, but this stays robust to a config that doesn't).
    """
    import re

    text = config_path.read_text()
    m = re.search(r"excluded_features_quantile:\s*\[([^\]]*)\]", text)
    if m is None:
        # No placeholder line present -- append a fresh one at end of file.
        current: list[str] = []
        merged = current + [c for c in new_cols if c not in current]
        text = text.rstrip("\n") + f"\nexcluded_features_quantile: [{', '.join(merged)}]\n"
        config_path.write_text(text)
        return
    current = [c.strip() for c in m.group(1).split(",") if c.strip()]
    merged = current + [c for c in new_cols if c not in current]
    new_line = f"excluded_features_quantile: [{', '.join(merged)}]"
    text = text[: m.start()] + new_line + text[m.end() :]
    config_path.write_text(text)


def run_vacancy_position_quantile(pos: str) -> dict | None:
    """The quantile vacancy gate for one position -- see this module's docstring's Part B."""
    qspec = qmod.quantile_spec(pos)
    extra_cols = vacancy_columns_for(pos)
    if not extra_cols:
        return None
    if not qspec.artifact_path.exists():
        print(f"  {pos}: no quantile bundle at {qspec.artifact_path}, skipping")
        return None
    bundle = joblib.load(qspec.artifact_path)
    cfg = bundle["cfg"]
    df = qmod.load_modeling_frame(qspec.features_path, qspec.labels_path, cfg=cfg)
    missing_cols = [c for c in extra_cols if c not in df.columns]
    if missing_cols:
        print(f"  {pos}: features_{pos}.parquet missing {missing_cols}; rebuild features first, skipping")
        return None

    seed = bundle["seed"]
    train_season_start = bundle.get("train_season_start", train.TRAIN_START_SEASON)
    folds = train.validation_folds(train_start_season=train_season_start)
    split = train.holdout_split(train_start_season=train_season_start)
    holdout_df = df[df["season"].isin(split.holdout_seasons)]

    if quantile_bundle_already_promoted(bundle, extra_cols):
        print(f"  {pos}: quantile bundle already carries {extra_cols} -- reconstructing the true pre-promotion incumbent")
        variant_wide = qmod.score_quantiles(bundle, holdout_df)
        variant_metrics = qmod.distribution_metrics(variant_wide)
        incumbent_metrics = reconstruct_incumbent_quantile_metrics(bundle, df, extra_cols, folds, seed)
        # No refit needed for the (already-current) variant side -- reuse the shipped
        # bundle's own fitted models/tree_cols/final_n_estimators as-is.
        fitted = {
            "tree_cols": list(bundle["tree_feature_cols"]),
            "final_n_estimators": dict(bundle["final_n_estimators"]),
            "models": bundle["models"],
            "params": bundle["params"],
        }
        # per_alpha_oof/per_alpha_fold_metrics are needed only by promote_quantile_position
        # (already-promoted bundles never re-promote here -- keep is re-derived below but a
        # positive decision on an already-promoted bundle is a no-op re-confirmation, not a
        # fresh promotion) so they're left absent; promote_quantile_position is only called
        # in the NOT-already-promoted branch.
        keep, reason = quantile_gate_decision(incumbent_metrics, variant_metrics)
        table = _quantile_metrics_table(pos, incumbent_metrics, variant_metrics)
        print(f"  {pos}: quantile gate re-check on already-promoted bundle -- {reason}")
        return {
            "position": pos, "extra_cols": extra_cols, "keep": keep, "reason": reason,
            "incumbent_metrics": incumbent_metrics, "variant_metrics": variant_metrics, "table": table,
            "already_promoted": True,
        }

    incumbent_wide = qmod.score_quantiles(bundle, holdout_df)
    incumbent_metrics = qmod.distribution_metrics(incumbent_wide)
    variant_metrics, fitted = fit_and_score_quantile_variant(bundle, df, extra_cols, folds, seed)

    keep, reason = quantile_gate_decision(incumbent_metrics, variant_metrics)
    table = _quantile_metrics_table(pos, incumbent_metrics, variant_metrics)

    if keep:
        promote_quantile_position(qspec, bundle, fitted, pos, extra_cols, reason)
        print(f"  {pos}: quantile KEPT -- bundle regenerated at {qspec.artifact_path}")
    else:
        add_to_excluded_features_quantile(train.position_spec(pos).config_path, extra_cols)
        print(f"  {pos}: quantile REJECTED -- {extra_cols} added to excluded_features_quantile")

    return {
        "position": pos, "extra_cols": extra_cols, "keep": keep, "reason": reason,
        "incumbent_metrics": incumbent_metrics, "variant_metrics": variant_metrics, "table": table,
        "already_promoted": False,
    }


def _quantile_metrics_table(pos: str, incumbent_metrics: dict, variant_metrics: dict) -> pd.DataFrame:
    rows = []
    for name, m in (("incumbent", incumbent_metrics), ("plus_vacancy", variant_metrics)):
        rows.append(
            {
                "position": pos, "variant": name, "n": m["n"],
                "spearman_q50_actual": m["spearman_q50_actual"],
                "pinball_q50": m["pinball_by_alpha"]["0.50"],
                "coverage_q10_q90": m["coverage_q10_q90"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def _null_rates(pos: str) -> dict[str, float]:
    spec = train.position_spec(pos)
    df = pd.read_parquet(spec.features_path)
    cols = vacancy_columns_for(pos)
    return {c: float(df[c].isna().mean()) for c in cols if c in df.columns}


def write_classifier_report(all_results: list[dict]) -> None:
    lines = ["# BreakoutLab -- Vacancy-Feature CLASSIFIER Capacity Gate (v2.4)", ""]
    lines.append(
        "Every hyperparameter/blend weight is FROZEN at the currently shipped bundle's "
        "per-seed values (no Optuna retuning) -- only the tree feature set (+ this "
        "position's new vacancy columns) and the n_estimators/calibrator that legitimately "
        "follow from it differ. Gate rule (binding): KEEP iff pooled (2024+2025) holdout "
        f"top-10 precision does not regress AND PR-AUC improves or regresses by at most {eg.PR_AUC_TOLERANCE}."
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
                f"{pooled.loc['incumbent','top10_precision']:.3f}", f"{pooled.loc['plus_vacancy','top10_precision']:.3f}",
                f"{pooled.loc['incumbent','pr_auc']:.3f}", f"{pooled.loc['plus_vacancy','pr_auc']:.3f}",
            ]
        )
    lines.append(eg._markdown_table(
        ["Position", "Decision", "Incumbent top-10", "+Vacancy top-10", "Incumbent PR-AUC", "+Vacancy PR-AUC"], rows
    ))
    lines.append("")

    for r in all_results:
        lines.append(f"## {r['label']}")
        lines.append("")
        lines.append(f"New columns: {r['extra_cols']}")
        lines.append("")
        null_rates = _null_rates(r["position"])
        lines.append(eg._markdown_table(["Column", "Null rate"], [[c, f"{v:.3f}"] for c, v in null_rates.items()]))
        lines.append("")
        table_rows = [
            [row["variant"], row["season"], row["n"], row["n_pos"], f"{row['top10_precision']:.3f}", f"{row['pr_auc']:.3f}"]
            for row in r["table"].to_dict("records")
        ]
        lines.append(eg._markdown_table(["Variant", "Season", "n", "n_pos", "Top-10 precision", "PR-AUC"], table_rows))
        lines.append("")
        lines.append(f"**Decision: {r['reason']}**")
        lines.append("")

    CLASSIFIER_OUT_PATH.write_text("\n".join(lines))


def write_quantile_report(all_results: list[dict]) -> None:
    lines = ["# BreakoutLab -- Vacancy-Feature QUANTILE Capacity Gate (v2.4)", ""]
    lines.append(
        "New this session -- the quantile head had no capacity gate before. Frozen "
        "alpha=0.50-tuned LightGBM hyperparameters throughout (no Optuna retuning); only "
        "the tree feature set (+ this position's new vacancy columns) and each alpha's own "
        "early-stopped n_estimators differ. Gate rule (binding): KEEP iff holdout "
        f"Spearman(q50, realized) does not regress by more than {QUANTILE_SPEARMAN_TOLERANCE} "
        f"AND pinball(q50) improves or stays within {QUANTILE_PINBALL_TOLERANCE*100:.0f}% of the "
        "incumbent's. Coverage(q10-q90) reported for both but does not gate."
    )
    lines.append("")
    lines.append("## Per-position decisions")
    lines.append("")
    rows = []
    for r in all_results:
        rows.append(
            [
                r["position"].upper(), "KEEP" if r["keep"] else "REJECT",
                f"{r['incumbent_metrics']['spearman_q50_actual']:.3f}", f"{r['variant_metrics']['spearman_q50_actual']:.3f}",
                f"{r['incumbent_metrics']['pinball_by_alpha']['0.50']:.4f}", f"{r['variant_metrics']['pinball_by_alpha']['0.50']:.4f}",
                f"{r['incumbent_metrics']['coverage_q10_q90']:.3f}", f"{r['variant_metrics']['coverage_q10_q90']:.3f}",
            ]
        )
    lines.append(eg._markdown_table(
        ["Position", "Decision", "Incumbent Spearman", "+Vacancy Spearman", "Incumbent pinball(q50)",
         "+Vacancy pinball(q50)", "Incumbent coverage", "+Vacancy coverage"],
        rows,
    ))
    lines.append("")

    for r in all_results:
        lines.append(f"## {r['position'].upper()}")
        lines.append("")
        lines.append(f"New columns: {r['extra_cols']}")
        lines.append("")
        null_rates = _null_rates(r["position"])
        lines.append(eg._markdown_table(["Column", "Null rate"], [[c, f"{v:.3f}"] for c, v in null_rates.items()]))
        lines.append("")
        lines.append(f"**Decision: {r['reason']}**")
        lines.append("")

    QUANTILE_OUT_PATH.write_text("\n".join(lines))


def main() -> int:
    classifier_results = []
    for pos in ("wr", "rb", "te", "qb"):
        print(f"vacancy classifier gate: {pos} ...")
        r = run_vacancy_position(pos)
        if r is not None:
            classifier_results.append(r)
    if classifier_results:
        write_classifier_report(classifier_results)
        print(f"wrote {CLASSIFIER_OUT_PATH}")

    quantile_results = []
    for pos in ("wr", "rb", "te", "qb"):
        print(f"vacancy quantile gate: {pos} ...")
        r = run_vacancy_position_quantile(pos)
        if r is not None:
            quantile_results.append(r)
    if quantile_results:
        write_quantile_report(quantile_results)
        print(f"wrote {QUANTILE_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
