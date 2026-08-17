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


def compute_all_baselines(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Every baseline's score Series, keyed by name, aligned to df's index."""
    age_cfg = cfg["baselines"]["age_adjusted_adp"]
    return {
        "adp_knows_best": adp_knows_best(df),
        "prior_season_ppg_rank": prior_season_ppg_rank(df),
        "age_adjusted_adp": age_adjusted_adp(df, age_cfg),
    }
