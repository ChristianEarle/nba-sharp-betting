"""v1.7 real retrain driver: run_full_pipeline + write_outputs for one position, using the
real (non-tiny) config, WITHOUT re-running the full pytest suite per position (the pytest
summary text embedded in each report is generated once, up front, and passed in) --
train.main()'s per-position pytest call is redundant 4x over when retraining every position
in the same session; this calls the identical run_full_pipeline/write_outputs/save_artifacts
path `python -m src.models.train_{pos}` uses, just factoring the shared pytest run out.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import train


def main() -> int:
    pos = sys.argv[1]
    summary_path = sys.argv[2] if len(sys.argv) > 2 else None
    pytest_summary = Path(summary_path).read_text() if summary_path else None

    spec = train.position_spec(pos)
    print(f"{spec.label_position} model training | v1.7 real retrain (config: {spec.config_path})", flush=True)
    cfg = train.load_config(spec.config_path)
    t0 = time.time()
    result = train.run_full_pipeline(spec, cfg=cfg)
    train.write_outputs(spec, result, pytest_summary=pytest_summary)
    dt = time.time() - t0
    print(f"{spec.label_position}: done in {dt:.0f}s", flush=True)
    print(f"wrote {spec.report_path}", flush=True)
    print(f"wrote {spec.metrics_json_path}", flush=True)
    print(f"wrote {spec.artifact_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
