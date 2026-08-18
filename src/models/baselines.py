"""Non-model baselines for WR breakout scoring (Deliverable 3).

Scored on the identical modeling universe as the trained model
(``src.models.train_wr``'s ``in_training_pool == 1`` rows) with the
identical metrics (``src.models.metrics``). Every baseline returns a
"score" Series where *higher = more likely breakout*, so baselines and the
model's calibrated probability sort the same direction and plug into the
same per-season top-k evaluation.

None of these are fit on data — they are fixed, documented formulas, not
tuned models. ``age_adjusted_adp``'s constants live in
``configs/model_wr.yaml``'s ``baselines.age_adjusted_adp`` block, per the
brief, rather than hardcoded here.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def adp_knows_best(df: pd.DataFrame) -> pd.Series:
    """score = -expectation_pos_rank: pure market order. Implies no breakout signal at all —

    the market's own preseason rank, inverted so a lower (better) rank scores higher.
    """
    return -df["expectation_pos_rank"]


def prior_season_ppg_rank(df: pd.DataFrame) -> pd.Series:
    """score = -(within-season rank of ppr_ppg_n1), 1 = highest N-1 ppg in that season.

    Ties broken by row order (``method="first"``, deterministic). A null
    ``ppr_ppg_n1`` (no qualifying prior season) ranks worst within its
    season via ``na_option="bottom"`` rather than being dropped.
    """
    rank = df.groupby("season")["ppr_ppg_n1"].rank(method="first", ascending=False, na_option="bottom")
    return -rank


def age_adjusted_adp(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """score = -expectation_pos_rank + age_bonus*1[age<=age_threshold] + year_in_league_bonus*1[year_in_league in eligible].

    ``cfg`` is ``configs/model_wr.yaml``'s ``baselines.age_adjusted_adp``
    block: fixed, documented constants, not tuned on this data.
    """
    age_bonus = cfg["age_bonus"] * (df["age"] <= cfg["age_threshold"]).astype(float)
    yil_bonus = cfg["year_in_league_bonus"] * df["year_in_league"].isin(cfg["year_in_league_eligible"]).astype(float)
    return -df["expectation_pos_rank"] + age_bonus + yil_bonus


BASELINE_NAMES = ("adp_knows_best", "prior_season_ppg_rank", "age_adjusted_adp")

# --------------------------------------------------------------------------
# v1.7 "fair baselines" (Step 4 of the model-fix brief): two more, evaluated
# alongside the three above on validation folds + the single holdout pass.
# Both are new functions rather than additions to BASELINE_NAMES/
# compute_all_baselines above -- those two are consumed by
# tests/test_models.py and tests/test_models_positions.py exactly as a
# static, no-CV-needed "score every row from cfg alone" contract, which
# ``napkin_logistic`` (a real per-fold fit) cannot honor. ``src.models.train``
# wires both of these into its own report/metrics-JSON generation, where it
# already has ``folds``/``holdout_split`` on hand.
# --------------------------------------------------------------------------


def eligible_prior_ppg(df: pd.DataFrame, adp_worse_than: float) -> pd.Series:
    """score: rank label-ELIGIBLE players (``expectation_pos_rank >= adp_worse_than`` --

    the same eligibility gate ``src.labels.build`` applies to define a breakout at all,
    see ``configs/labels.yaml``) by ``ppr_ppg_n1`` descending, within season; every
    INeligible player scores below every eligible player that season.

    Built by masking ``ppr_ppg_n1`` to NaN for ineligible rows before ranking: pandas'
    ``rank(..., na_option="bottom")`` already places every NaN after every real value
    within its group, in original-order among themselves -- which is exactly "ineligible
    players below all eligible ones, deterministic tiebreak" with no separate branch
    needed. An eligible player with a genuinely null ``ppr_ppg_n1`` gets the identical
    bottom-of-group treatment as ``prior_season_ppg_rank`` already gives it.
    """
    eligible = df["expectation_pos_rank"] >= adp_worse_than
    masked_ppg = df["ppr_ppg_n1"].where(eligible)
    rank = masked_ppg.groupby(df["season"]).rank(method="first", ascending=False, na_option="bottom")
    return -rank


def napkin_logistic_oof(df: pd.DataFrame, folds) -> pd.DataFrame:
    """Fold-by-fold L2 logistic on 3 features (expectation_pos_rank, ppr_ppg_n1, age),

    median-imputed (fit on train only, per ``src.models.cv.Fold``) -- pooled OOF
    predictions, same expanding-window discipline every other OOF computation in this
    pipeline uses (``src.models.train.classifier_oof``/``logistic_oof``). Returns
    (season, gsis_id, score); score is *not* named "pred" to avoid an accidental
    ``blend_grid_search``-style merge collision with the real classifier trio's OOF frames.
    """
    feats = ["expectation_pos_rank", "ppr_ppg_n1", "age"]
    rows = []
    for fold in folds:
        train_df = df[df["season"].isin(fold.train_seasons)]
        val_df = df[df["season"] == fold.val_season]
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(train_df[feats])
        X_val = imputer.transform(val_df[feats])
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, train_df["breakout"])
        score = model.predict_proba(X_val)[:, 1]
        rows.append(pd.DataFrame({"season": val_df["season"].to_numpy(), "gsis_id": val_df["gsis_id"].to_numpy(), "score": score}))
    return pd.concat(rows, ignore_index=True)


def napkin_logistic_holdout(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.Series:
    """Same 3-feature L2 logistic, fit once on ``train_df`` (<=2023) and scored on

    ``holdout_df`` (2024+2025) -- the holdout analogue of ``napkin_logistic_oof``,
    matching how every other model/baseline in this pipeline handles the ONE holdout pass
    (fit on <=2023, predict, never refit per holdout year).
    """
    feats = ["expectation_pos_rank", "ppr_ppg_n1", "age"]
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[feats])
    X_holdout = imputer.transform(holdout_df[feats])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_df["breakout"])
    score = model.predict_proba(X_holdout)[:, 1]
    return pd.Series(score, index=holdout_df.index)


def compute_all_baselines(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Every baseline's score Series, keyed by name, aligned to df's index."""
    age_cfg = cfg["baselines"]["age_adjusted_adp"]
    return {
        "adp_knows_best": adp_knows_best(df),
        "prior_season_ppg_rank": prior_season_ppg_rank(df),
        "age_adjusted_adp": age_adjusted_adp(df, age_cfg),
    }
