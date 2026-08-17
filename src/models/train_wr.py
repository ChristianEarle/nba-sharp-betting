"""WR bindings of ``src.models.train`` (Phase 4).

WR was the original, hand-written Phase-4 module; every function it used
to define now lives in ``src.models.train``, generalized to take a
``PositionSpec`` instead of hardcoding WR's paths (see that module's
docstring for the full pipeline writeup — tuning, blending, calibration,
the one-shot holdout evaluation, determinism). This module is kept only
as a thin, WR-bound wrapper: same importable names
(``FEATURES_WR_PATH``, ``LABELS_PATH``, ``load_config``,
``load_modeling_frame``, ``run_full_pipeline``, ...) and the same
``python -m src.models.train_wr`` entry point, so nothing that already
depends on this module's shape (``tests/test_models.py`` in particular)
had to change.

For RB/TE/QB, use ``python -m src.models.train <rb|te|qb>`` directly, or
their own thin wrappers (``src.models.train_rb`` / ``train_te`` /
``train_qb``), which follow this identical pattern.
"""

from __future__ import annotations

from pathlib import Path

from src.models import train

SPEC = train.position_spec("wr")

REPO_ROOT = train.REPO_ROOT
PROCESSED_DIR = train.PROCESSED_DIR
MODELS_DIR = train.MODELS_DIR
OUTPUTS_DIR = train.OUTPUTS_DIR

FEATURES_WR_PATH = SPEC.features_path
LABELS_PATH = SPEC.labels_path
CONFIG_PATH = SPEC.config_path
REPORT_PATH = SPEC.report_path
METRICS_JSON_PATH = SPEC.metrics_json_path
ARTIFACT_PATH = SPEC.artifact_path

IDENTITY_COLS = train.IDENTITY_COLS
LABEL_JOIN_COLS = train.LABEL_JOIN_COLS
NON_FEATURE_COLS = train.NON_FEATURE_COLS
REPORT_ID_COLS = train.REPORT_ID_COLS
EARLY_STOPPING_ROUNDS = train.EARLY_STOPPING_ROUNDS

# Every fold-fitting / tuning / blending / calibration / reporting helper
# is position-agnostic already (it was never WR-specific) — re-exported
# unchanged for any caller that imported them from this module directly.
fit_lgbm_classifier = train.fit_lgbm_classifier
fit_xgb_classifier = train.fit_xgb_classifier
fit_logistic = train.fit_logistic
fit_lgbm_regressor = train.fit_lgbm_regressor
fit_xgb_regressor = train.fit_xgb_regressor
tree_feature_columns = train.tree_feature_columns
logistic_feature_columns = train.logistic_feature_columns
fold_frames = train.fold_frames
params_at_boundary = train.params_at_boundary
tune_classifier = train.tune_classifier
classifier_oof = train.classifier_oof
tune_logistic = train.tune_logistic
logistic_oof = train.logistic_oof
blend_grid_search = train.blend_grid_search
apply_blend = train.apply_blend
choose_calibration = train.choose_calibration
apply_calibration = train.apply_calibration
tune_regressor = train.tune_regressor
regressor_oof = train.regressor_oof
holdout_predictions = train.holdout_predictions
holdout_regression_eval = train.holdout_regression_eval
evaluate_scores = train.evaluate_scores
named_top10 = train.named_top10
generate_report = train.generate_report


def load_config(path: Path = CONFIG_PATH) -> dict:
    return train.load_config(path)


def load_modeling_frame(
    features_path: Path = FEATURES_WR_PATH,
    labels_path: Path = LABELS_PATH,
    cfg: dict | None = None,
) -> "pd.DataFrame":  # noqa: F821 - pandas import kept in src.models.train only
    return train.load_modeling_frame(features_path, labels_path, cfg=cfg)


def run_full_pipeline(
    cfg: dict | None = None,
    classifier_trials: int | None = None,
    regression_trials: int | None = None,
    seed: int | None = None,
    output_root=None,
) -> dict:
    return train.run_full_pipeline(
        SPEC,
        cfg=cfg,
        classifier_trials=classifier_trials,
        regression_trials=regression_trials,
        seed=seed,
        output_root=output_root,
    )


def save_artifacts(frozen: dict, seed: int, cfg: dict) -> None:
    train.save_artifacts(SPEC, frozen, seed, cfg)


def write_outputs(result: dict, pytest_summary: str | None = None) -> None:
    train.write_outputs(SPEC, result, pytest_summary=pytest_summary)


def main() -> int:
    return train.main("wr")


if __name__ == "__main__":
    raise SystemExit(main())
