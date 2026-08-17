"""TE bindings of ``src.models.train`` (Phase 4). See ``src.models.train_wr``

for the pattern this follows (thin, position-bound wrapper; all real logic
lives in ``src.models.train``).
"""

from __future__ import annotations

from pathlib import Path

from src.models import train

SPEC = train.position_spec("te")

FEATURES_TE_PATH = SPEC.features_path
LABELS_PATH = SPEC.labels_path
CONFIG_PATH = SPEC.config_path
REPORT_PATH = SPEC.report_path
METRICS_JSON_PATH = SPEC.metrics_json_path
ARTIFACT_PATH = SPEC.artifact_path


def load_config(path: Path = CONFIG_PATH) -> dict:
    return train.load_config(path)


def load_modeling_frame(
    features_path: Path = FEATURES_TE_PATH,
    labels_path: Path = LABELS_PATH,
    cfg: dict | None = None,
):
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


def main() -> int:
    return train.main("te")


if __name__ == "__main__":
    raise SystemExit(main())
