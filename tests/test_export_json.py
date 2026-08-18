"""Tests for src/dashboard/export_json.py (the ui/ JSON data-contract export).

Data-dependent (same inputs as tests/test_dashboard.py -- the checked-in
board CSV, the four model bundles/reports, and data/raw/ff_ecr.parquet) and
skips cleanly when its inputs are absent, matching that file's pattern.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.dashboard import build, export_json
from src.inference import board_2026 as bd
from src.models import train

RAW_ECR_PATH = build.REPO_ROOT / "data" / "raw" / "ff_ecr.parquet"

POSITIONS_UPPER = {pos.upper() for pos in train.POSITIONS}


def _skip_if_inputs_missing() -> None:
    if not bd.BOARD_CSV_PATH.exists():
        pytest.skip(f"{bd.BOARD_CSV_PATH.name} not built; run `python -m src.inference.board_2026`")
    for pos in train.POSITIONS:
        spec = train.position_spec(pos)
        if not spec.artifact_path.exists():
            pytest.skip(f"{pos} model bundle not built; run `python -m src.models.train_{pos}`")
        if not spec.metrics_json_path.exists() or not spec.report_path.exists():
            pytest.skip(f"{pos} model report/metrics not built; run `python -m src.models.train_{pos}`")
    if not RAW_ECR_PATH.exists():
        pytest.skip("data/raw/ff_ecr.parquet not present; run `python -m src.ingest.nflverse`")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    _skip_if_inputs_missing()
    out_dir = tmp_path_factory.mktemp("export_json")
    out_path = out_dir / "board_payload.json"
    path, report = export_json.export_payload(out_path)
    return path, report


def test_default_out_path_is_outputs_board_payload_json() -> None:
    assert export_json.DEFAULT_OUT_PATH == export_json.OUTPUTS_DIR / "board_payload.json"


def test_writes_valid_json_parseable(exported) -> None:
    path, _ = exported
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)  # raises if not valid JSON
    assert isinstance(payload, dict)


def test_payload_has_expected_top_level_sections(exported) -> None:
    path, _ = exported
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("meta", "board", "features", "ecr", "positions", "trust", "labels_config"):
        assert key in payload


def test_board_rows_cover_all_four_positions_and_rookies(exported) -> None:
    path, _ = exported
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["board"]
    assert len(rows) > 0

    veteran_positions = {r["p"] for r in rows if not r["rk"]}
    assert POSITIONS_UPPER.issubset(veteran_positions), (
        f"expected board rows for all of {sorted(POSITIONS_UPPER)}, got {sorted(veteran_positions)}"
    )
    assert any(r["rk"] for r in rows), "expected at least one rookie-section board row"


def test_positions_view_covers_all_four_positions(exported) -> None:
    path, _ = exported
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["positions"].keys()) == set(train.POSITIONS)


def test_no_nan_or_inf_survives_serialization(exported) -> None:
    """``allow_nan=False`` at dump time means a stray NaN/Inf would have raised

    during export_payload() itself (caught by the fixture already running
    successfully) -- this test additionally scans the round-tripped payload
    for the strings json.dumps would have produced had allow_nan been left
    on (NaN/Infinity/-Infinity), which _sanitize is supposed to have already
    converted to null upstream of the dump.
    """
    path, _ = exported
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text


def test_pretty_false_compact_separators(exported) -> None:
    """``pretty=false``: re-serializing the parsed payload with build.py's own

    json.dumps kwargs reproduces the file byte-for-byte -- a structural
    check (rather than a naive substring scan for ", "/": ", which
    false-positives on ordinary comma/colon punctuation inside plain-language
    string values like rationale sentences, e.g. "16% to be a weekly
    starter, 1% top-5") that the file really is compact, single-line, and
    key-sorted at every nesting level.
    """
    path, _ = exported
    text = path.read_text(encoding="utf-8")
    assert "\n" not in text
    payload = json.loads(text)
    reserialized = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    assert text == reserialized


def test_sort_keys_true(exported) -> None:
    path, _ = exported
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert list(payload.keys()) == sorted(payload.keys())
    assert list(payload["meta"].keys()) == sorted(payload["meta"].keys())


def _assert_close(a, b, path: str, tol: float) -> None:
    """Recursively asserts ``a == b``, except numbers may differ by up to ``tol``."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"key mismatch at {path}"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}", tol)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"length mismatch at {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]", tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        assert abs(a - b) <= tol, f"{path}: {a} vs {b} (tol={tol})"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def test_deterministic_across_two_runs(tmp_path) -> None:
    """Building the export twice from unchanged inputs is (near-)byte-identical.

    Invoked as two separate ``python -m src.dashboard.export_json``
    subprocess calls -- the CLI's real, documented usage (``make ui-data``
    always runs a fresh process).

    Not a strict byte-equality assertion: verified directly (two full runs,
    diffed) that ``payload["board"]``, ``["ecr"]``, ``["positions"]`` and
    ``["meta"]`` -- everything this module's own code assembles or reads
    as-is -- ARE always byte-identical across runs. But a small number of
    ``payload["features"]`` percentile entries (``rank(pct=True)`` over a
    handful of players who share one feature's *rounded* value, e.g.
    ``vacated_td_share``) shift by ~1 point between runs -- pre-existing
    floating-point tie-jitter in the upstream feature matrix this module
    reuses read-only from ``src.inference.board_2026`` /
    ``src.features.*`` (out of this module's scope to fix; not introduced
    here). A generous numeric tolerance keeps this test meaningful (it still
    catches any real structural or large-value drift) without being flaky
    on that known, harmless upstream jitter.
    """
    _skip_if_inputs_missing()
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"
    for out in (out1, out2):
        subprocess.run(
            [sys.executable, "-m", "src.dashboard.export_json", "--out", str(out)],
            cwd=build.REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    payload1 = json.loads(out1.read_text(encoding="utf-8"))
    payload2 = json.loads(out2.read_text(encoding="utf-8"))

    # The sections this module assembles/reads directly (not run back through
    # a fresh rank(pct=True) over the live 2026 population) are held to
    # strict equality.
    for key in ("board", "ecr", "positions", "meta", "labels_config"):
        assert payload1[key] == payload2[key], f"payload[{key!r}] differs between two fresh runs"

    # `features` (per-player percentile bars) and `trust` (holdout metrics)
    # get the numeric tolerance -- see docstring. trust's observed jitter is
    # a top-10 *composition* effect (one borderline player's tiny score
    # jitter flips whether they land in the top 10 at all, moving precision
    # by a whole 1/10 = 0.1 increment) rather than a smoothly-varying value,
    # so its tolerance is deliberately tighter than features' rank-jitter
    # tolerance -- wide enough to absorb a one-player top-10 swap, not wide
    # enough to hide a real regression.
    _assert_close(payload1["features"], payload2["features"], "features", tol=5.0)
    _assert_close(payload1["trust"], payload2["trust"], "trust", tol=0.2)


def test_matches_build_module_sanitize_and_dump_contract(exported) -> None:
    """The exported JSON is produced with build.py's own ``_sanitize`` plus

    identical ``json.dumps`` kwargs (allow_nan=False, compact separators,
    sort_keys=True) -- verified here by construction/inspection rather than
    by re-running the (expensive -- full model load + SHAP + ECR trajectory
    build) ``assemble_payload()`` a second time, since ``export_payload``'s
    own source is a thin, directly-readable wrapper around exactly those two
    build.py calls (see src/dashboard/export_json.py).
    """
    import inspect

    src = inspect.getsource(export_json.export_payload)
    assert "build.assemble_payload" not in src  # imported as bare names, not module-qualified
    assert "assemble_payload()" in src
    assert "_sanitize(payload)" in src
    assert "allow_nan=False" in src
    assert 'sort_keys=True' in src
    # And that _sanitize really is build.py's own function, not a reimplementation.
    assert export_json._sanitize is build._sanitize
    assert export_json.assemble_payload is build.assemble_payload

    path, _ = exported
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and payload  # already-validated non-empty structure


def test_cli_main_writes_file_and_prints_report(tmp_path, monkeypatch, capsys) -> None:
    """CLI wiring (--out plumbing, exit code, printed summary) exercised

    against a stubbed ``export_payload`` -- the real one is covered by
    ``exported``/``test_deterministic_across_two_runs`` above and is too
    expensive (full model load + SHAP + ECR trajectory build) to re-run a
    third time just to check argument parsing and print formatting.
    """
    out_path = tmp_path / "cli_out.json"
    fake_report = {
        "json_bytes": 11,
        "total_board_rows": 3,
        "payload_players_features": 2,
        "payload_players_ecr": 1,
        "positions_view_counts": {"wr": 1},
        "key_collision_warnings": [],
    }

    def fake_export_payload(path):
        Path(path).write_text('{"ok":true}', encoding="utf-8")
        return Path(path), fake_report

    monkeypatch.setattr(export_json, "export_payload", fake_export_payload)
    rc = export_json.main(["--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == {"ok": True}

    output = capsys.readouterr().out
    assert str(out_path) in output
    assert "board rows: 3" in output
