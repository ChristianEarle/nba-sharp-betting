"""Tests for Phase 6 (2026 inference) and Phase 6/7's board deliverable.

Every test here is data-dependent (a saved model bundle, a built 2026
feature matrix, or the board CSV) and skips cleanly when its input is
absent, matching every other data-dependent test file's pattern in this
repo (``_skip_if_*`` helpers). Run ``python -m src.inference.board_2026``
(after the usual ``src.ingest.nflverse`` / ``src.labels.build`` /
``src.features.*`` / ``src.models.train_*`` chain) to produce everything
these tests read.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.inference import board_2026
from src.models import train

# --------------------------------------------------------------------------
# Golden-player regression test
# --------------------------------------------------------------------------

# Verified directly against the currently-saved bundles before encoding
# (see the session's final report): each of these is a real breakout==1
# holdout row (2024 or 2025) that scores strictly above its position+season
# median calibrated probability when scored from the stored
# features_{pos}.parquet row through the saved bundle -- not cherry-picked
# for a huge score, just a clear, reproducible margin over the median.
GOLDEN_PLAYERS = [
    ("wr", 2025, "Chris Olave"),
    ("te", 2025, "Kyle Pitts"),
    ("rb", 2024, "Chuba Hubbard"),
    ("qb", 2024, "Baker Mayfield"),
]


def _score_holdout(pos: str) -> pd.DataFrame | None:
    spec = train.position_spec(pos)
    if not (spec.artifact_path.exists() and spec.features_path.exists() and spec.labels_path.exists()):
        return None
    bundle = joblib.load(spec.artifact_path)
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
    holdout = df[df["season"].isin([2024, 2025])].copy()
    if holdout.empty:
        return None

    tree_cols = bundle["tree_feature_cols"]
    weights = bundle["blend_weights"]
    preds = {}
    if weights.get("lgbm", 0) > 0:
        preds["lgbm"] = bundle["lgbm"].predict_proba(holdout[tree_cols])[:, 1]
    if weights.get("xgb", 0) > 0:
        preds["xgb"] = bundle["xgb"].predict_proba(holdout[tree_cols])[:, 1]
    if weights.get("logistic", 0) > 0:
        model, imputer, scaler = bundle["logistic"]
        X = scaler.transform(imputer.transform(holdout[bundle["logistic_feature_cols"]]))
        preds["logistic"] = model.predict_proba(X)[:, 1]
    nonzero_weights = {k: w for k, w in weights.items() if k in preds}
    blended = train.apply_blend(nonzero_weights, **preds)
    holdout["prob"] = train.apply_calibration(bundle["calibration_method"], bundle["calibrator"], blended)
    return holdout


@pytest.mark.parametrize("pos,season,player_name", GOLDEN_PLAYERS)
def test_golden_breakout_scores_above_position_season_median(pos: str, season: int, player_name: str) -> None:
    holdout = _score_holdout(pos)
    if holdout is None:
        pytest.skip(f"{pos}: bundle or feature/label data not built; run `python -m src.models.train {pos}`")

    row = holdout[(holdout["season"] == season) & (holdout["player_name"] == player_name)]
    if row.empty:
        pytest.skip(f"{player_name} {season} not in {pos} holdout rows (bundle/feature data has drifted)")

    median = holdout.loc[holdout["season"] == season, "prob"].median()
    assert row["prob"].iloc[0] > median, (
        f"{player_name} {season} ({pos}): prob={row['prob'].iloc[0]:.4f} did not beat "
        f"the {season} position median ({median:.4f})"
    )


# --------------------------------------------------------------------------
# Board schema test
# --------------------------------------------------------------------------


def _board_df() -> pd.DataFrame:
    if not board_2026.BOARD_CSV_PATH.exists():
        pytest.skip(f"{board_2026.BOARD_CSV_PATH.name} not built; run `python -m src.inference.board_2026`")
    return pd.read_csv(board_2026.BOARD_CSV_PATH)


REQUIRED_BOARD_COLUMNS = [
    "player_name",
    "pos",
    "team",
    "probability",
    "probability_calibrated_raw",
    "tier",
    "base_rate_multiple",
    "raw_score",
    "probability_saturated",
    "expected_rank_delta",
    "consensus_ecr_pos_rank",
    "adp",
    "sleeper_adp_pos_rank",
    "adp_gap",
    "implied_pts",
    "edge",
    "availability",
    "shap_top3",
    "rationale",
    "section",
    "probability_heuristic",
    "broke_out_last_season",
    # v2.4: low-octane-offense risk flag (see board_2026.compute_low_octane_flags).
    "low_octane_offense",
    "low_octane_source",
]


def test_board_required_columns_present() -> None:
    df = _board_df()
    missing = [c for c in REQUIRED_BOARD_COLUMNS if c not in df.columns]
    assert not missing, f"breakout_board_2026.csv missing columns: {missing}"


def test_board_probabilities_in_unit_interval() -> None:
    """`probability_heuristic` (rookies) is a plain, uncapped [0, 1] probability.

    `probability` (veterans, v1.7) is the base-rate-renormalized display value:
    strictly positive, capped at ``board_2026.PROB_DISPLAY_CAP`` (0.97) -- see
    board_2026's module docstring ("v1.7: base-rate renormalization") for why a
    per-position rescale replaces the old fixed [0.01, 0.95] clamp.
    `probability_calibrated_raw` (pre-rescale, Platt-calibrated) still carries the
    old clamp range, since that column is untouched by the position-level rescale.
    """
    df = _board_df()

    heuristic_vals = df["probability_heuristic"].dropna()
    assert (heuristic_vals >= 0).all() and (heuristic_vals <= 1).all(), "probability_heuristic: values outside [0, 1]"

    prob_vals = df["probability"].dropna()
    assert not prob_vals.empty
    assert (prob_vals > 0).all() and (prob_vals <= board_2026.PROB_DISPLAY_CAP + 1e-9).all(), (
        f"probability: values outside (0, {board_2026.PROB_DISPLAY_CAP}]"
    )

    raw_vals = df["probability_calibrated_raw"].dropna()
    assert not raw_vals.empty
    assert (raw_vals >= board_2026.PROB_DISPLAY_LO - 1e-9).all() and (raw_vals <= board_2026.PROB_DISPLAY_HI + 1e-9).all(), (
        f"probability_calibrated_raw: values outside the display clamp [{board_2026.PROB_DISPLAY_LO}, {board_2026.PROB_DISPLAY_HI}]"
    )


def test_board_displayed_ranking_matches_raw_score_ranking() -> None:
    """v1.7: Platt calibration is strictly monotone and the per-position renormalization

    scale is a single positive constant, so within each position the displayed
    `probability` must rank identically to `raw_score` (the pre-calibration ensemble
    blend score) -- see board_2026's module docstring.
    """
    df = _board_df()
    veterans = df[df["section"] == "veteran"]
    for pos, group in veterans.groupby("pos"):
        by_prob = group.sort_values(["probability", "raw_score"], ascending=[False, False])["player_name"].tolist()
        by_raw = group.sort_values("raw_score", ascending=False)["player_name"].tolist()
        assert by_prob == by_raw, f"{pos}: displayed probability ranking does not match raw_score ranking"


def test_board_saturated_rows_flagged_and_ordered_by_raw_score() -> None:
    """Every probability_saturated==1 row hit the pre-rescale display clamp boundary

    (0.01 or 0.95, on `probability_calibrated_raw`), and within a tied raw-calibrated
    bucket, raw_score gives a strict-enough ordering to break the tie.
    """
    df = _board_df()
    veterans = df[df["section"] == "veteran"]
    saturated = veterans[veterans["probability_saturated"] == 1]
    if saturated.empty:
        pytest.skip("no saturated rows on this build's board (Platt didn't hit a terminal clamp bucket)")

    assert set(saturated["probability_calibrated_raw"].round(2).unique()) <= {board_2026.PROB_DISPLAY_LO, board_2026.PROB_DISPLAY_HI}

    for (pos, prob), group in veterans.groupby(["pos", "probability_calibrated_raw"]):
        if len(group) > 1:
            assert group["raw_score"].nunique() > 1, (
                f"{pos} probability_calibrated_raw={prob}: {len(group)} tied rows share an identical raw_score too -- "
                "ordering among them is arbitrary"
            )


def test_board_has_zero_saturated_rows_v1_7() -> None:
    """v1.7: src.inference.board_2026 scores off train.fit_platt (a sigmoid, which can only

    output an exact 0.0/1.0 in the limit -- effectively never at real floating-point raw
    scores), so the board's probability_saturated count should be exactly 0.
    """
    df = _board_df()
    veterans = df[df["section"] == "veteran"]
    if veterans.empty:
        pytest.skip("no veteran rows on the board")
    n_saturated = int(veterans["probability_saturated"].sum())
    assert n_saturated == 0, f"expected 0 saturated veteran rows with Platt calibration, got {n_saturated}"


def test_board_displayed_probability_sum_near_historical_mean_breakouts() -> None:
    """v1.7 renormalization check: each position's sum(displayed probability) should be

    within +/-20% of that position's historical mean per-season breakout count -- the
    whole point of the rescale (see board_2026.attach_renormalized_probability).
    """
    df = _board_df()
    veterans = df[df["section"] == "veteran"]
    if veterans.empty:
        pytest.skip("no veteran rows on the board")
    mean_breakouts = board_2026.historical_mean_breakouts_by_position()
    for pos, group in veterans.groupby("pos"):
        target = mean_breakouts.get(pos)
        if target is None or target <= 0:
            continue
        total = group["probability"].sum()
        lo, hi = target * 0.8, target * 1.2
        assert lo <= total <= hi, f"{pos}: sum(displayed probability)={total:.2f} not within +/-20% of historical mean {target:.2f}"


def test_board_rookies_are_flagged_and_separate_from_veterans() -> None:
    df = _board_df()
    sections = set(df["section"].unique())
    assert "veteran" in sections
    rookie_sections = [s for s in sections if "rookie" in s]
    assert rookie_sections, "no rookie section found on the board"

    veterans = df[df["section"] == "veteran"]
    rookies = df[df["section"].isin(rookie_sections)]
    # Veterans are scored on `probability` (calibrated), never `probability_heuristic`;
    # rookies are the reverse -- the two scales must never be mixed on one column.
    assert veterans["probability"].notna().any()
    assert rookies["probability_heuristic"].notna().any()
    assert veterans["probability_heuristic"].isna().all(), "veterans must not carry a heuristic probability"
    assert rookies["probability"].isna().all(), "rookies must not carry a calibrated veteran probability"


def test_board_availability_column_exists() -> None:
    df = _board_df()
    assert "availability" in df.columns


# --------------------------------------------------------------------------
# 2026 feature-matrix sanity
# --------------------------------------------------------------------------

# Approximate expected row counts per position's 2026 veteran (non-rookie)
# population -- rosters.parquet season==2026, years_exp>=1. QB's real
# active-roster population is smaller than the other three (roughly one
# start/backup/third string per team, ~32-100), so it gets its own,
# narrower bound rather than sharing WR/RB/TE's range.
ROW_COUNT_BOUNDS = {
    "qb": (50, 150),
    "rb": (100, 250),
    "wr": (150, 350),
    "te": (80, 200),
}


def _feature_matrix(pos: str) -> pl.DataFrame:
    path = board_2026.veteran_features_path(pos)
    if not path.exists():
        pytest.skip(f"{path.name} not built; run `python -m src.inference.board_2026`")
    return pl.read_parquet(path)


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_2026_feature_matrix_row_count_sane(pos: str) -> None:
    df = _feature_matrix(pos)
    lo, hi = ROW_COUNT_BOUNDS[pos]
    assert lo <= df.height <= hi, f"{pos}: {df.height} 2026 veteran rows outside sanity bounds [{lo}, {hi}]"


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_2026_feature_matrix_no_duplicate_keys(pos: str) -> None:
    df = _feature_matrix(pos)
    dupes = df.group_by(["season", "gsis_id"]).len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"{pos}: duplicate (season, gsis_id) rows in the 2026 feature matrix: {dupes}"


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_2026_new_oc_is_all_null(pos: str) -> None:
    """configs/coaching_changes.csv has no 2026 rows yet (documented limitation) --

    new_oc must read null for every 2026 row, not a guessed 0.
    """
    df = _feature_matrix(pos)
    assert df.get_column("new_oc").null_count() == df.height, f"{pos}: new_oc should be all-null for 2026"


@pytest.mark.parametrize("pos", ("wr", "rb", "te", "qb"))
def test_2026_new_hc_is_not_all_null(pos: str) -> None:
    """schedules.parquet's 2026 Week-1 home/away_coach columns are fully populated,

    so new_hc must be derivable for at least some 2026 rows.
    """
    df = _feature_matrix(pos)
    assert df.get_column("new_hc").null_count() < df.height, f"{pos}: new_hc is entirely null for 2026"


# --------------------------------------------------------------------------
# v2.1 Deliverable 3: driver-chip honesty fix
# --------------------------------------------------------------------------


def test_shap_top3_binary_chips_never_self_contradict() -> None:
    """team_change/new_hc chips must reflect the feature's actual VALUE, not just its

    name + SHAP sign -- the pre-v2.1 bug ("New team +" rendering for team_change=0
    players). A single player's binary-feature state is a single fact, so a chip
    string can never claim both mutually-exclusive states for the same feature
    (e.g. both "Stayed put" and "New team", or both coach-continuity states).
    """
    df = _board_df()
    if "shap_top3" not in df.columns:
        pytest.skip("shap_top3 not in board CSV")
    s = df["shap_top3"].astype(str)
    for lo, hi in board_2026.BINARY_FEATURE_STATES.values():
        contradictions = df[s.str.contains(lo, na=False, regex=False) & s.str.contains(hi, na=False, regex=False)]
        assert contradictions.empty, f"a shap_top3 row claims both {lo!r} and {hi!r} at once"


def test_shap_top3_no_bare_article_after_elite_thin() -> None:
    """Grammar regression guard: 'Elite a ...'/'Thin a ...' (a leading article surviving

    the Elite/Thin/Rising/Falling prefix) must never appear -- honest_state_label strips
    it (see src.inference.board_2026.honest_state_label).
    """
    df = _board_df()
    if "shap_top3" not in df.columns:
        pytest.skip("shap_top3 not in board CSV")
    bad = df["shap_top3"].astype(str).str.contains(
        r"(?:Elite|Thin|Rising|Falling) (?:a|an|the) ", regex=True, na=False
    )
    assert not bad.any(), f"{int(bad.sum())} shap_top3 rows have a leading-article grammar bug"


def test_humanize_feature_full_key_takes_priority_over_stripped_base() -> None:
    """Finding 2 regression guard: `_FEATURE_LABELS` is keyed by the FULL feature name
    (almost always the "_n1" form). `_humanize_feature` must try that exact key BEFORE
    falling back to the suffix-stripped base -- looking up the stripped base first made
    every suffixed key in the map dead (e.g. "rec_yards_pg_n1" never matched, always
    falling through to the raw "rec yards pg" code).
    """
    assert board_2026._humanize_feature("rec_yards_pg_n1") == board_2026._FEATURE_LABELS["rec_yards_pg_n1"]
    assert board_2026._humanize_feature("target_share_n1") == board_2026._FEATURE_LABELS["target_share_n1"]
    assert "_" not in board_2026._humanize_feature("rec_yards_pg_n1")


def test_every_bundle_feature_has_a_plain_language_label() -> None:
    """Every feature that can appear in any position's SHAP top list must humanize to a
    plain-language phrase, never a raw code fragment -- audits the full tree + logistic
    feature-column universe across all four position bundles.
    """
    import re as _re

    positions = ("qb", "rb", "te", "wr")
    specs = {p: train.position_spec(p) for p in positions}
    if not all(spec.artifact_path.exists() for spec in specs.values()):
        pytest.skip("model bundles not built; run `python -m src.models.train <pos>`")

    all_feats: set[str] = set()
    for pos, spec in specs.items():
        bundle = joblib.load(spec.artifact_path)
        all_feats |= set(bundle["tree_feature_cols"])
        all_feats |= set(bundle.get("logistic_feature_cols", []))

    raw_fragment = _re.compile(r"\b(pg|n1|yoy)\b")
    offenders = []
    for feat in sorted(all_feats):
        label = board_2026._humanize_feature(feat)
        if "_" in label or raw_fragment.search(label):
            offenders.append((feat, label))
    assert not offenders, f"features without a plain-language label: {offenders}"


def test_shap_top3_chips_never_contain_raw_feature_codes() -> None:
    """Regenerated-board audit: zero shap_top3 chips contain an underscore or a raw
    "pg"/"n1"/"yoy" code fragment -- the shipped-CSV symptom of Finding 2's bug.
    """
    import re as _re

    df = _board_df()
    if "shap_top3" not in df.columns:
        pytest.skip("shap_top3 not in board CSV")
    s = df["shap_top3"].astype(str)
    has_underscore = s.str.contains("_", na=False, regex=False)
    has_raw_fragment = s.str.contains(r"\b(?:pg|n1|yoy)\b", na=False, regex=True)
    bad = df[has_underscore | has_raw_fragment]
    assert bad.empty, f"{len(bad)} shap_top3 rows contain raw feature-code fragments: {bad['shap_top3'].tolist()[:5]}"


def test_honest_state_label_binary_reflects_value() -> None:
    lo, hi = board_2026.BINARY_FEATURE_STATES["team_change"]
    assert board_2026.honest_state_label("team_change", 0, None) == lo
    assert board_2026.honest_state_label("team_change", 1, None) == hi
    assert board_2026.honest_state_label("new_hc", 1, None) == board_2026.BINARY_FEATURE_STATES["new_hc"][1]
    assert board_2026.honest_state_label("new_hc", 0, None) == board_2026.BINARY_FEATURE_STATES["new_hc"][0]


def test_honest_state_label_continuous_reflects_value_vs_median() -> None:
    assert board_2026.honest_state_label("target_share_n1", 0.9, 0.2).startswith("Elite")
    assert board_2026.honest_state_label("target_share_n1", 0.05, 0.2).startswith("Thin")
    # lower-is-better feature: a below-median age is "Elite", not "Thin".
    assert board_2026.honest_state_label("age", 22, 27).startswith("Elite")
    assert board_2026.honest_state_label("age", 32, 27).startswith("Thin")
    # missing median -> no state claim invented, falls back to the plain label.
    assert board_2026.honest_state_label("target_share_n1", 0.9, None) == board_2026._humanize_feature(
        "target_share_n1"
    )


def test_honest_state_label_trend_compares_to_zero_not_median() -> None:
    """Finding 3: '_yoy_delta' trend features compare the player's OWN delta to ZERO,
    never to the position median. A below-median-but-positive delta must still read
    'Rising', and any negative delta must read 'Falling' -- even when the population
    median delta is itself negative (the bug: comparing to a negative median could
    mislabel an actual decliner as 'Rising').
    """
    # Named decliner case: median delta among the population is NEGATIVE (-0.05, i.e.
    # most players' target share fell), but this specific player's own delta is also
    # negative (-0.02) -- a real decline. Old (median-relative) logic would have called
    # this "Rising" since -0.02 >= -0.05; the fixed zero-relative logic must call it
    # "Falling".
    label = board_2026.honest_state_label("target_share_yoy_delta", -0.02, -0.05)
    assert label.startswith("Falling"), f"a real decliner (-0.02) must read 'Falling', got {label!r}"

    # A positive delta is always "Rising", even below a higher positive median.
    label_up = board_2026.honest_state_label("target_share_yoy_delta", 0.01, 0.05)
    assert label_up.startswith("Rising"), f"a positive delta (0.01) must read 'Rising', got {label_up!r}"

    # Missing median must NOT suppress the trend label -- zero-comparison needs no median.
    label_no_median = board_2026.honest_state_label("target_share_yoy_delta", 0.03, None)
    assert label_no_median.startswith("Rising")


def test_honest_state_label_level_features_still_use_median() -> None:
    """Elite/Thin (vs the position median) must still apply to plain level features
    (non-trend, non-binary) -- only trend features move to the zero comparison.
    """
    assert board_2026.honest_state_label("target_share_n1", 0.9, 0.2).startswith("Elite")
    assert board_2026.honest_state_label("target_share_n1", 0.05, 0.2).startswith("Thin")


def test_honest_state_label_returning_competition_custom_states() -> None:
    """Finding 4: returning_incumbent_share and backfield_committee_count are
    LOWER_IS_BETTER and render with bespoke 'Light/Heavy returning competition' state
    text (with a directional arrow), not the generic Elite/Thin prefix (which reads
    backwards for a "competition" concept).
    """
    assert "returning_incumbent_share" in board_2026.LOWER_IS_BETTER
    assert "backfield_committee_count" in board_2026.LOWER_IS_BETTER

    # A below-median (light) committee count is the GOOD state.
    light = board_2026.honest_state_label("backfield_committee_count_n1", 1, 3)
    assert light.startswith("Light returning competition")
    assert "Elite" not in light and "Thin" not in light

    # An above-median (heavy) committee count is the BAD state.
    heavy = board_2026.honest_state_label("backfield_committee_count_n1", 4, 3)
    assert heavy.startswith("Heavy returning competition")
    assert "Elite" not in heavy and "Thin" not in heavy


# --------------------------------------------------------------------------
# v2.1 addendum: "broke out last season" transparency badge
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Finding 9: score_veterans_batch must hoist the seed-invariant logistic head
# --------------------------------------------------------------------------


class _CountingLogit:
    """Fake logistic model that counts predict_proba calls -- the loop-hoist regression

    guard: the logistic head is SEED-INVARIANT (fit once, shared across every seed), so
    it must be called at most ONCE per score_veterans_batch call, never once per seed.
    """

    def __init__(self) -> None:
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        n = len(X)
        return np.column_stack([np.zeros(n), np.full(n, 0.5)])


class _ConstModel:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.zeros(n), np.full(n, 0.6)])

    def predict(self, X):
        return np.zeros(len(X))


class _IdentityTransform:
    def transform(self, X):
        return np.asarray(X)


class _IdentityCalibrator:
    def predict(self, score):
        return score


def test_score_veterans_batch_calls_logistic_head_once_regardless_of_seed_count() -> None:
    counting_logit = _CountingLogit()
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0], "player_name": ["A", "B", "C"], "gsis_id": ["1", "2", "3"]})
    bundle = {
        "tree_feature_cols": ["f1", "f2"],
        "logistic_feature_cols": ["f1"],
        "logistic": (counting_logit, _IdentityTransform(), _IdentityTransform()),
        # THREE seeds, each with a nonzero logistic blend weight -- pre-fix, this would
        # call predict_proba three times (once per seed) instead of once, shared.
        "per_seed_models": {
            1: {"lgbm": _ConstModel(), "xgb": _ConstModel()},
            2: {"lgbm": _ConstModel(), "xgb": _ConstModel()},
            3: {"lgbm": _ConstModel(), "xgb": _ConstModel()},
        },
        "per_seed": {
            1: {"blend_weights": {"lgbm": 0.3, "xgb": 0.3, "logistic": 0.4}},
            2: {"blend_weights": {"lgbm": 0.2, "xgb": 0.4, "logistic": 0.4}},
            3: {"blend_weights": {"lgbm": 0.4, "xgb": 0.2, "logistic": 0.4}},
        },
        "active_calibration_method": "isotonic",
        "active_calibrator": _IdentityCalibrator(),
        "lgbm_reg": _ConstModel(),
        "xgb_reg": _ConstModel(),
    }

    out = board_2026.score_veterans_batch(bundle, df)

    assert counting_logit.calls == 1, f"logistic head called {counting_logit.calls} times for 3 seeds -- must be exactly 1 (seed-invariant)"
    assert len(out) == len(df)
    assert np.isfinite(out["raw_score"]).all()


def test_score_veterans_batch_skips_logistic_head_when_every_seed_weight_is_zero() -> None:
    counting_logit = _CountingLogit()
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [4.0, 5.0], "player_name": ["A", "B"], "gsis_id": ["1", "2"]})
    bundle = {
        "tree_feature_cols": ["f1", "f2"],
        "logistic_feature_cols": ["f1"],
        "logistic": (counting_logit, _IdentityTransform(), _IdentityTransform()),
        "per_seed_models": {1: {"lgbm": _ConstModel(), "xgb": _ConstModel()}},
        "per_seed": {1: {"blend_weights": {"lgbm": 0.5, "xgb": 0.5, "logistic": 0.0}}},
        "active_calibration_method": "isotonic",
        "active_calibrator": _IdentityCalibrator(),
        "lgbm_reg": _ConstModel(),
        "xgb_reg": _ConstModel(),
    }
    board_2026.score_veterans_batch(bundle, df)
    assert counting_logit.calls == 0, "logistic head must not be called when no seed's blend weight uses it"


def test_broke_out_last_season_matches_labels() -> None:
    """Every board row flagged broke_out_last_season=True must have a real

    breakout==1 row in labels.parquet's season==2025 -- and every 2025 breakout
    player who is also on the 2026 board must be flagged (both directions).
    """
    if not train.LABELS_PATH.exists():
        pytest.skip("labels.parquet not built")
    df = _board_df()
    if "broke_out_last_season" not in df.columns:
        pytest.skip("broke_out_last_season not in board CSV")

    labels = pl.read_parquet(train.LABELS_PATH)
    broke_out_names = set(
        labels.filter((pl.col("season") == 2025) & (pl.col("breakout") == 1))
        .get_column("player_name")
        .to_list()
    )
    flagged = set(df[df["broke_out_last_season"] == True]["player_name"])  # noqa: E712
    assert flagged <= broke_out_names, f"flagged players not actually 2025 breakouts: {flagged - broke_out_names}"

    on_board_veterans = set(df[df["section"] == "veteran"]["player_name"])
    should_be_flagged = broke_out_names & on_board_veterans
    assert should_be_flagged <= flagged, f"2025 breakouts on the board but not flagged: {should_be_flagged - flagged}"


def test_broke_out_last_season_is_boolean() -> None:
    df = _board_df()
    if "broke_out_last_season" not in df.columns:
        pytest.skip("broke_out_last_season not in board CSV")
    assert df["broke_out_last_season"].isin([True, False]).all()


def test_broke_out_last_season_rookies_never_flagged() -> None:
    df = _board_df()
    if "broke_out_last_season" not in df.columns:
        pytest.skip("broke_out_last_season not in board CSV")
    rookies = df[df["section"] == "rookie (heuristic)"]
    if rookies.empty:
        pytest.skip("no rookie rows on the board")
    assert not rookies["broke_out_last_season"].any(), "a 2026 rookie cannot have broken out in 2025"


# --------------------------------------------------------------------------
# v2.4: low-octane-offense risk flag (board_2026.compute_low_octane_flags)
# --------------------------------------------------------------------------


def _skip_if_schedules_missing() -> None:
    if not board_2026.sh.PATHS["schedules"].exists():
        pytest.skip("schedules.parquet not built")


def test_compute_low_octane_flags_flags_exactly_five_teams_source_b() -> None:
    """Source B (schedules-derived actual PPG) fallback -- the current repo state, since

    no season-2026 odds_api snapshot exists yet (see the module docstring). Exactly
    LOW_OCTANE_N (5) teams must read low_octane_offense==1, and every other resolvable
    team must read 0 (never null/missing after the fallback), source labeled "est.".
    """
    _skip_if_schedules_missing()
    schedules = pl.read_parquet(board_2026.sh.PATHS["schedules"])
    out = board_2026.compute_low_octane_flags(schedules)
    assert (out["low_octane_source"] == "est.").all(), "Source B fallback must label every row 'est.', never 'Vegas'"
    assert int(out["low_octane_offense"].sum()) == board_2026.LOW_OCTANE_N
    assert set(out["low_octane_offense"].unique()) <= {0, 1}


def test_compute_low_octane_flags_source_a_used_when_2026_vegas_lines_present(monkeypatch, tmp_path) -> None:
    """Source A (Vegas preseason team lines) takes priority over Source B whenever the

    repo actually has season-``SEASON`` team_lines rows for at least LOW_OCTANE_N teams --
    synthesized here since the real repo has none yet (see the module docstring's "current
    repo state" note). Exactly LOW_OCTANE_N teams must be flagged, source labeled "Vegas".
    """
    _skip_if_schedules_missing()
    from src.ingest import vegas as vg_ingest

    # 8 REAL team full names (vegas.TEAM_NAME_TO_CODE only accepts real NFL franchise
    # names, never raises on an unmapped one), synthetic spreads/totals engineered so
    # implied_ppg is strictly increasing team-to-team -- the bottom 5 (by construction)
    # are the first 5 in this list.
    teams = [
        "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
        "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    ]
    codes = [vg_ingest.TEAM_NAME_TO_CODE[t] for t in teams]
    rows = []
    for i, t in enumerate(teams):
        rows.append(
            {
                "season": board_2026.SEASON, "snapshot": "2026-08-01", "event_id": f"evt{i}",
                "commence_time": "2026-09-07T00:00:00Z", "home_team": t, "away_team": teams[(i + 1) % len(teams)],
                "home_ml": -110, "away_ml": -110, "home_spread": 0.0, "total": 30.0 + i,
            }
        )
    fake_team_lines = pd.DataFrame(rows)
    fake_path = tmp_path / "team_lines.parquet"
    fake_team_lines.to_parquet(fake_path)

    monkeypatch.setattr(vg_ingest, "TEAM_LINES_PATH", fake_path)
    schedules = pl.read_parquet(board_2026.sh.PATHS["schedules"])
    out = board_2026.compute_low_octane_flags(schedules)

    assert (out["low_octane_source"] == "Vegas").all()
    assert int(out["low_octane_offense"].sum()) == board_2026.LOW_OCTANE_N
    assert set(out["team"]) == set(codes)

    # Self-consistency (rather than hand-computing the round-robin implied_ppg average):
    # the flagged set must be exactly the bottom LOW_OCTANE_N teams by the SAME
    # build_vegas_team implied_ppg this function itself sorts on.
    vt = vg_ingest.build_vegas_team(pl.from_pandas(fake_team_lines)).filter(pl.col("season") == board_2026.SEASON)
    expected_bottom = set(vt.sort("implied_ppg").head(board_2026.LOW_OCTANE_N).get_column("team").to_list())
    flagged = set(out.loc[out["low_octane_offense"] == 1, "team"])
    assert flagged == expected_bottom, (flagged, expected_bottom)


def test_low_octane_offense_flag_covers_exactly_five_teams_on_board() -> None:
    """Every veteran board row's own low_octane_offense/low_octane_source, joined onto

    the real board -- exactly LOW_OCTANE_N distinct TEAMS (not rows -- a flagged team can
    have many players) should read low_octane_offense==1, and the source label must be
    either "Vegas" or "est." (never "unknown" -- schedules.parquet always resolves every
    real NFL team's actual-PPG fallback).
    """
    df = _board_df()
    veterans = df[df["section"] == "veteran"]
    if veterans.empty or "low_octane_offense" not in veterans.columns:
        pytest.skip("no veteran rows / low_octane_offense not on the board CSV")
    flagged_teams = set(veterans.loc[veterans["low_octane_offense"] == 1, "team"].dropna())
    assert len(flagged_teams) == board_2026.LOW_OCTANE_N, flagged_teams
    assert set(veterans["low_octane_source"].dropna().unique()) <= {"Vegas", "est.", "unknown"}


# --------------------------------------------------------------------------
# v2.4: peer-rank-only display -- the "typical finish" framing is dropped from every
# user-facing sentence/table (board_2026.build_projection_sentence / write_board),
# while predicted_finish_rank/value_gap_typical stay on the CSV for analysts.
# --------------------------------------------------------------------------


def test_projection_sentence_never_mentions_typical_finish() -> None:
    row = pd.Series(
        {
            "expected_ppg": 15.7, "floor_ppg": 12.4, "ceiling_ppg": 20.4, "pos": "RB",
            "value_gap": 6.0, "expectation_pos_rank": 8, "cohort_rank": 2, "season": 2026,
            "predicted_finish_rank": 10, "p_startable": 0.74, "p_elite": 0.18,
        }
    )
    sentence = board_2026.build_projection_sentence(row)
    assert isinstance(sentence, str)
    assert "historically finishes" not in sentence
    assert "typical" not in sentence.lower()
    assert "Projects RB2 of 2026" in sentence


def test_board_csv_keeps_predicted_finish_rank_and_value_gap_typical_as_analyst_columns() -> None:
    """Peer-rank-only display is a presentation change, not a data removal --

    predicted_finish_rank/value_gap_typical must still be on the board CSV.
    """
    df = _board_df()
    assert "predicted_finish_rank" in df.columns
    assert "value_gap_typical" in df.columns
    assert "cohort_rank" in df.columns
    assert "value_gap" in df.columns


def test_board_md_never_shows_typical_finish_column_or_label() -> None:
    """No TABLE (a `|`-delimited row -- header or data) may carry a "Typical finish" or

    "Value gap (cohort)" column; the module's own explanatory prose is allowed to
    reference those retired column names by name when describing the change (see
    write_board's peer-rank-only note), so this checks table rows specifically, not
    every line in the file.
    """
    if not board_2026.BOARD_MD_PATH.exists():
        pytest.skip(f"{board_2026.BOARD_MD_PATH.name} not built; run `python -m src.inference.board_2026`")
    table_lines = [line for line in board_2026.BOARD_MD_PATH.read_text().splitlines() if line.startswith("|")]
    assert table_lines, "no markdown table rows found at all -- board.md structure changed?"
    assert not any("Typical finish" in line for line in table_lines)
    assert not any("Value gap (cohort)" in line for line in table_lines)
    # No player-detail sentence (the "say" line embedded in the Rationale/Projection
    # sentence table cells) may carry the retired typical-finish clause either.
    assert not any("historically finishes" in line for line in table_lines)
