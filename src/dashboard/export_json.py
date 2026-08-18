"""JSON data-contract export for the ui/ React app.

Reuses ``src.dashboard.build``'s payload assembly (``assemble_payload`` and
``_sanitize``) READ-ONLY -- this module imports from ``src.dashboard.build``
but never edits it. Where ``src.dashboard.build.build_dashboard`` embeds the
sanitized payload inside a hand-rolled HTML page, this module instead writes
the identical sanitized payload straight to a standalone JSON file
(``outputs/board_payload.json`` by default) for the Vite/React UI in ``ui/``
to ``fetch()`` at runtime.

Same determinism contract as the HTML build: no wall-clock timestamps in the
payload (see ``src.dashboard.build``'s module docstring), every float
rounded to 3dp, ``allow_nan=False`` so a stray NaN/Inf fails loudly instead
of round-tripping as invalid JSON, and ``sort_keys=True`` so dict key order
(not guaranteed stable across polars/pandas groupings) doesn't make two
back-to-back builds byte-different. ``separators=(",", ":")`` matches
``build.py``'s own compact (pretty=false) encoding.

CLI:
    uv run python -m src.dashboard.export_json [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dashboard.build import OUTPUTS_DIR, _sanitize, assemble_payload

DEFAULT_OUT_PATH = OUTPUTS_DIR / "board_payload.json"


def export_payload(out_path: Path = DEFAULT_OUT_PATH) -> tuple[Path, dict]:
    """Assembles the dashboard payload and writes it to ``out_path`` as JSON.

    Returns (out_path, report) -- ``report`` is ``assemble_payload``'s own
    report dict (printed by ``main``, never embedded in the JSON itself),
    with ``json_bytes`` added.
    """
    payload, report = assemble_payload()
    clean = _sanitize(payload)
    payload_json = json.dumps(clean, allow_nan=False, separators=(",", ":"), sort_keys=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload_json, encoding="utf-8")

    report = dict(report)
    report["json_bytes"] = len(payload_json.encode("utf-8"))
    report["total_board_rows"] = len(payload["board"])
    return out_path, report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"output JSON path (default: {DEFAULT_OUT_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path, report = export_payload(args.out)
    print(f"wrote {path} ({report['json_bytes']:,} bytes)")
    print(f"board rows: {report['total_board_rows']}")
    print(f"players with feature profiles: {report['payload_players_features']}")
    print(f"players with ECR drift series: {report['payload_players_ecr']}")
    print(f"positions view counts: {report['positions_view_counts']}")
    if report.get("key_collision_warnings"):
        print(f"WARNING: {len(report['key_collision_warnings'])} player-key collision(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
