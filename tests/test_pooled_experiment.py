"""Tests for the v2.1 Deliverable 2 pooled-position classifier experiment

(``src/models/pooled_experiment.py``). Data-dependent (reads the gitignored
``data/processed/features_{pos}/labels.parquet``) and skips cleanly when
absent, matching every other pipeline test file's pattern in this repo.
Every pipeline run in this file uses a tiny 2-trial, 1-seed config
(``output_root`` redirects the written report/metrics away from
``outputs/`` -- the real Deliverable-2 numbers live there and must never be
clobbered by a test run, same discipline as
``tests/test_models_positions.py``). The full run is __main__-only:
``python -m src.models.pooled_experiment`` / ``make pooled-experiment``.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from src.models import pooled_experiment as pe
from src.models import train

_REQUIRED_PATHS = [train.position_spec(p).features_path for p in train.POSITIONS] + [
    train.position_spec(p).labels_path for p in train.POSITIONS
]


def _skip_if_raw_missing() -> None:
    if not all(p.exists() for p in _REQUIRED_PATHS):
        pytest.skip("features/labels parquet not built; run the src.features.* + src.labels.build chain")


@lru_cache(maxsize=1)
def _pooled_frame():
    _skip_if_raw_missing()
    return pe.load_pooled_frame()


def test_load_pooled_frame_totals_match_the_brief() -> None:
    """The brief's own verified data facts: ~166 positives total, 27-48 per position."""
    pooled_df, tree_cols, logit_cols, cfg_by_pos = _pooled_frame()
    assert int(pooled_df["breakout"].sum()) == 166
    for pos in train.POSITIONS:
        label_pos = pos.upper()
        sub = pooled_df[pooled_df["src_position"] == label_pos]
        n_pos = int(sub["breakout"].sum())
        assert 27 <= n_pos <= 48, f"{label_pos}: {n_pos} positives outside the brief's stated [27, 48] range"


def test_load_pooled_frame_no_position_leak_into_wrong_bucket() -> None:
    pooled_df, *_ = _pooled_frame()
    for pos in train.POSITIONS:
        label_pos = pos.upper()
        sub = pooled_df[pooled_df["src_position"] == label_pos]
        # every OTHER position's one-hot column must be 0 for this position's own rows
        for other in train.POSITIONS:
            if other == pos:
                assert (sub[f"position_{pos}"] == 1.0).all()
            else:
                assert (sub[f"position_{other}"] == 0.0).all()


def test_load_pooled_frame_no_duplicate_season_gsis_id() -> None:
    pooled_df, *_ = _pooled_frame()
    dupes = pooled_df.groupby(["season", "gsis_id"]).size()
    assert (dupes <= 1).all(), "a (season, gsis_id) row appears under more than one position"


def test_tree_cols_include_position_onehot_and_no_string_columns() -> None:
    pooled_df, tree_cols, logit_cols, cfg_by_pos = _pooled_frame()
    for c in pe.POSITION_ONEHOT_COLS:
        assert c in tree_cols
    for c in tree_cols:
        assert pooled_df[c].dtype.kind in "fib", f"{c}: non-numeric dtype {pooled_df[c].dtype} would break LightGBM"


@pytest.fixture(scope="module")
def tiny_result(tmp_path_factory):
    _skip_if_raw_missing()
    out = tmp_path_factory.mktemp("pooled_experiment_tiny")
    return pe.run_experiment(classifier_trials=2, seeds=[42], base_seed=42, output_root=out)


def test_tiny_run_has_one_decision_per_position(tiny_result) -> None:
    positions = {d["position"] for d in tiny_result["decisions"]}
    assert positions == {"WR", "RB", "TE", "QB"}


def test_tiny_run_promote_is_boolean_and_conservative_on_ties(tiny_result) -> None:
    for d in tiny_result["decisions"]:
        assert isinstance(d["promote"], bool)
        # Gate contract: promote only on a STRICT win; equal or NaN either side -> False.
        if d["pooled_holdout_top10_precision"] <= d["incumbent_top10_precision"]:
            assert d["promote"] is False


@pytest.mark.parametrize("pos", ("WR", "RB", "TE", "QB"))
def test_tiny_run_chosen_arm_valid(tiny_result, pos: str) -> None:
    d = next(d for d in tiny_result["decisions"] if d["position"] == pos)
    assert d["chosen_arm"] in ("pooled", "pruned_pooled")


def test_tiny_run_writes_to_output_root_not_production(tmp_path) -> None:
    """A tiny run with output_root set must never touch the production outputs/ files --

    the same non-clobbering guarantee ``train.run_full_pipeline``'s ``output_root`` gives.
    """
    _skip_if_raw_missing()
    before = pe.POOLED_REPORT_MD_PATH.read_bytes() if pe.POOLED_REPORT_MD_PATH.exists() else None
    out = tmp_path / "redirected"
    pe.run_experiment(classifier_trials=2, seeds=[42], base_seed=42, output_root=out)
    assert (out / pe.POOLED_REPORT_MD_PATH.name).exists()
    assert (out / pe.POOLED_METRICS_JSON_PATH.name).exists()
    after = pe.POOLED_REPORT_MD_PATH.read_bytes() if pe.POOLED_REPORT_MD_PATH.exists() else None
    assert before == after, "production outputs/pooled_experiment.md was modified by a tiny test run"


def test_pruned_feature_count_at_most_top_n(tiny_result) -> None:
    assert len(tiny_result["tree_cols_pruned"]) <= pe.PRUNED_TOP_N


def test_val_pr_auc_in_unit_interval(tiny_result) -> None:
    for arm in ("pooled", "pruned_pooled"):
        for pos, val in tiny_result["val_pr_auc"][arm].items():
            if val == val:  # not NaN
                assert 0.0 <= val <= 1.0, f"{arm}/{pos}: val PR-AUC {val} outside [0, 1]"


def test_holdout_top10_precision_in_unit_interval(tiny_result) -> None:
    for arm in ("pooled", "pruned_pooled"):
        for pos, info in tiny_result["holdout_top10"][arm].items():
            for key in ("top10_precision_all_holdout", "top10_precision_eligible_holdout"):
                val = info[key]
                if val == val:  # not NaN
                    assert 0.0 <= val <= 1.0, f"{arm}/{pos}/{key}: {val} outside [0, 1]"


def test_determinism_two_tiny_runs_identical_decisions(tmp_path) -> None:
    _skip_if_raw_missing()
    r1 = pe.run_experiment(classifier_trials=2, seeds=[42], base_seed=42, output_root=tmp_path / "a")
    r2 = pe.run_experiment(classifier_trials=2, seeds=[42], base_seed=42, output_root=tmp_path / "b")
    for d1, d2 in zip(
        sorted(r1["decisions"], key=lambda d: d["position"]), sorted(r2["decisions"], key=lambda d: d["position"])
    ):
        assert d1["chosen_arm"] == d2["chosen_arm"]
        assert d1["promote"] == d2["promote"]
        assert d1["pooled_holdout_top10_precision"] == pytest.approx(d2["pooled_holdout_top10_precision"], abs=1e-9)
