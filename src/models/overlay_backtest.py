"""Overlay backtest: did the ADP-discount overlay (and a Vegas-informed variant) beat the
model alone, 2023-2025? (v1.5 Phase C, Deliverable 2)

Three scores, compared against the SAME modeling universe and predictions
every other report in this repo uses -- nothing here retrains or retunes
anything:

- **model**: the bundle's own calibrated probability. 2023 comes from the
  bundle's persisted pooled OOF validation predictions (2023 is one of the
  four ``src.models.cv.VALIDATION_SEASONS`` folds -- v1.5 bundles ship
  ``pooled_oof_val``, see ``src.models.train.run_full_pipeline``), run
  through the bundle's ``smoothed_calibrator``. 2024/2025 come straight
  from the bundle's holdout-retrained models via
  ``src.inference.board_2026.score_veterans_batch`` -- the exact function
  the real 2026 board scores with -- so "holdout predictions reproducible
  from bundles" means literally that: no retraining happens in this file.
- **overlay**: ``probability * log1p(expectation_pos_rank)`` -- the
  board's own documented formula (see
  ``src.inference.board_2026``'s "Overlay formula" section), reproduced
  here verbatim, not reinvented.
- **overlay_vegas**: ``probability * log1p(expectation_pos_rank) *
  clip(1 + z(implied_ppg), 0.5, 1.5)``, where ``z(implied_ppg)`` is the
  player's season-N team's implied points-per-game
  (``data/processed/vegas_team.parquet``, from ``src.ingest.vegas``)
  z-scored **across that season's priced teams** (team-level, computed
  once per season and joined onto every player on that team -- not
  z-scored within the position's scored-row population, which would let
  a position with fewer teams represented skew the mean/std). Restricted
  to ``has_vegas == 1`` rows (null otherwise, per the brief) -- in
  practice this is nearly the full universe for 2020-2025, since every
  team ends up listed at least once per season even in the thin
  2020-2022 partial-coverage years (see ``src.ingest.vegas``'s module
  docstring).

Metrics: top-10 precision (``src.models.metrics.top_k_precision``) and
Spearman rank correlation of each score against ``finish_rank_delta``
(rows with a null finish_rank_delta -- fewer than
``configs/labels.yaml``'s min_games that season -- excluded from the
correlation, same convention every other report in this repo uses),
reported per (position, season) AND pooled across 2023-2025, exactly
mirroring ``src.models.train.evaluate_scores``'s "pooled" row.

Small-sample honesty
----------------------
Holdout positive counts run 7-13 per position (see
``src.models.train``'s module docstring) -- a top-10-precision delta of
one player is entirely plausible noise at this sample size. Every verdict
line states the pooled positive count and flags explicitly when it's too
small to distinguish a real ranking improvement from chance, rather than
letting a bare "X beats Y" number imply more confidence than the sample
supports.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.inference import board_2026 as board
from src.ingest import vegas
from src.models import metrics as mt
from src.models import train

REPO_ROOT = train.REPO_ROOT
OUTPUTS_DIR = train.OUTPUTS_DIR
OUT_PATH = OUTPUTS_DIR / "overlay_backtest.md"

BACKTEST_SEASONS: tuple[int, ...] = (2023,) + train.HOLDOUT_SEASONS  # (2023, 2024, 2025)
SCORE_COLS: dict[str, str] = {"model": "probability", "overlay": "overlay_score", "overlay_vegas": "overlay_vegas_score"}

# Below this pooled positive count, a top-10-precision/Spearman delta between
# two scores is flagged as not distinguishable from noise (see module
# docstring's "Small-sample honesty" section) -- a documented threshold, not
# a statistical test; the report states the raw n_pos alongside it either way.
MIN_POS_FOR_CONFIDENT_DELTA = 15


# --------------------------------------------------------------------------
# Scored universe: model probability for 2023 (OOF) + 2024/2025 (holdout)
# --------------------------------------------------------------------------


def build_scored_frame(spec: train.PositionSpec, bundle: dict) -> pd.DataFrame:
    """(season, gsis_id) scored rows for BACKTEST_SEASONS with a `probability` column --

    2023 from the bundle's pooled OOF validation predictions (smoothed-calibrated), 2024/2025
    from the bundle's holdout-retrained models scored via board_2026.score_veterans_batch
    (identical function the real board uses -- no retraining anywhere in this file).
    """
    cfg = bundle["cfg"]
    df = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=cfg)
    df = df[df["season"].isin(BACKTEST_SEASONS)].copy()

    oof = bundle["pooled_oof_val"]
    oof_2023 = oof[oof["season"] == 2023][["season", "gsis_id", "pred_blend"]].copy()
    calibrated = train.apply_calibration(
        bundle["smoothed_calibration_method"], bundle["smoothed_calibrator"], oof_2023["pred_blend"].to_numpy()
    )
    oof_2023["probability"] = np.clip(calibrated, board.PROB_DISPLAY_LO, board.PROB_DISPLAY_HI)

    val_2023 = df[df["season"] == 2023].merge(
        oof_2023[["season", "gsis_id", "probability"]], on=["season", "gsis_id"], how="inner"
    )
    assert len(val_2023) == len(df[df["season"] == 2023]), (
        f"{spec.position}: pooled_oof_val is missing rows for 2023's in_training_pool universe"
    )

    holdout_df = df[df["season"].isin(train.HOLDOUT_SEASONS)].copy()
    holdout_scored = board.score_veterans_batch(bundle, holdout_df) if len(holdout_df) else holdout_df.assign(probability=[])

    keep = ["season", "gsis_id", "player_name", "team", "breakout", "finish_rank_delta", "expectation_pos_rank", "probability"]
    both = pd.concat([val_2023[keep], holdout_scored[keep]], ignore_index=True)
    both["position"] = spec.label_position
    return both


# --------------------------------------------------------------------------
# Vegas z-score + overlays
# --------------------------------------------------------------------------


def load_vegas_team_with_zscore() -> pd.DataFrame:
    """vegas_team.parquet -> pandas, + z_implied_ppg (team-level, z-scored within season
    across every team vegas_team.parquet priced that season)."""
    if not vegas.VEGAS_TEAM_PATH.exists():
        vegas.run_build_vegas_team()
    vt = pl.read_parquet(vegas.VEGAS_TEAM_PATH).to_pandas()
    vt["z_implied_ppg"] = vt.groupby("season")["implied_ppg"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    return vt


def attach_overlays(df: pd.DataFrame, vegas_team: pd.DataFrame) -> pd.DataFrame:
    """discount = log1p(expectation_pos_rank) (the board's own formula); overlay_score =
    probability * discount; overlay_vegas_score additionally multiplies by
    clip(1 + z(implied_ppg), 0.5, 1.5) for has_vegas==1 rows, null otherwise.
    """
    out = df.merge(vegas_team[["season", "team", "implied_ppg", "has_vegas", "z_implied_ppg"]], on=["season", "team"], how="left")
    out["has_vegas"] = out["has_vegas"].fillna(0).astype(int)
    out["discount"] = np.log1p(out["expectation_pos_rank"])
    out["overlay_score"] = out["probability"] * out["discount"]
    multiplier = (1.0 + out["z_implied_ppg"]).clip(lower=0.5, upper=1.5)
    out["overlay_vegas_score"] = np.where(out["has_vegas"] == 1, out["probability"] * out["discount"] * multiplier, np.nan)
    return out


# --------------------------------------------------------------------------
# Metrics: top-10 precision + Spearman vs finish_rank_delta, per season + pooled
# --------------------------------------------------------------------------


def _spearman(score: pd.Series, target: pd.Series) -> float:
    mask = score.notna() & target.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(score[mask].corr(target[mask], method="spearman"))


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season|'pooled', score name): n, n_pos, top10_precision, spearman."""
    rows = []
    for season in list(BACKTEST_SEASONS) + ["pooled"]:
        mask = (df["season"] == season) if season != "pooled" else pd.Series(True, index=df.index)
        sub = df.loc[mask]
        for name, col in SCORE_COLS.items():
            valid = sub[sub[col].notna()]
            n = int(len(valid))
            n_pos = int(valid["breakout"].sum()) if n else 0
            top10 = mt.top_k_precision(valid["breakout"], valid[col], 10) if n else float("nan")
            rho = _spearman(valid[col], valid["finish_rank_delta"])
            rows.append({"season": season, "score": name, "n": n, "n_pos": n_pos, "top10_precision": top10, "spearman": rho})
    return pd.DataFrame(rows)


def verdict_line(position: str, table: pd.DataFrame) -> str:
    pooled = table[table["season"] == "pooled"].set_index("score")
    n_pos = int(pooled.loc["model", "n_pos"])

    def _fmt_delta(metric: str, a: str, b: str) -> str:
        va, vb = pooled.loc[a, metric], pooled.loc[b, metric]
        if pd.isna(va) or pd.isna(vb):
            return f"{metric}: n/a"
        return f"{metric} {a}={va:.3f} vs {b}={vb:.3f} (delta {va - vb:+.3f})"

    top10_overlay = _fmt_delta("top10_precision", "overlay", "model")
    top10_vegas = _fmt_delta("top10_precision", "overlay_vegas", "model")
    sp_overlay = _fmt_delta("spearman", "overlay", "model")
    sp_vegas = _fmt_delta("spearman", "overlay_vegas", "model")

    ranking = sorted(
        ("model", "overlay", "overlay_vegas"),
        key=lambda s: (pooled.loc[s, "top10_precision"] if pd.notna(pooled.loc[s, "top10_precision"]) else -1),
        reverse=True,
    )
    caveat = (
        f"pooled n_pos={n_pos} < {MIN_POS_FOR_CONFIDENT_DELTA} -- a one-player top-10 swing "
        "is well within noise at this sample size, so this ranking is directional, not a "
        "statistically supported win."
        if n_pos < MIN_POS_FOR_CONFIDENT_DELTA
        else f"pooled n_pos={n_pos} -- still small, but past the module's documented flag threshold."
    )
    return (
        f"**{position} verdict:** pooled top-10-precision order is {' > '.join(ranking)}. "
        f"{top10_overlay}; {top10_vegas}. Spearman: {sp_overlay}; {sp_vegas}. {caveat}"
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def table_to_markdown(table: pd.DataFrame) -> str:
    headers = ["Season", "Score", "n", "n_pos", "Top-10 precision", "Spearman vs finish_rank_delta"]
    rows = [[r["season"], r["score"], r["n"], r["n_pos"], r["top10_precision"], r["spearman"]] for _, r in table.iterrows()]
    return _markdown_table(headers, rows)


def run_position(pos: str) -> tuple[pd.DataFrame, str] | None:
    spec = train.position_spec(pos)
    if not spec.artifact_path.exists():
        print(f"  {pos}: no bundle at {spec.artifact_path}, skipping")
        return None
    bundle = joblib.load(spec.artifact_path)
    vegas_team = load_vegas_team_with_zscore()

    scored = build_scored_frame(spec, bundle)
    scored = attach_overlays(scored, vegas_team)
    table = score_table(scored)
    verdict = verdict_line(spec.label_position, table)
    return table, verdict


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BreakoutLab -- Overlay Backtest (v1.5 Phase C)",
        "",
        "model = bundle's own calibrated probability (2023 pooled OOF, 2024/2025 holdout-retrained "
        "-- see module docstring). overlay = probability * log1p(expectation_pos_rank) (the board's "
        "documented formula). overlay_vegas = overlay * clip(1 + z(implied_ppg), 0.5, 1.5), "
        "has_vegas rows only.",
        "",
    ]
    all_verdicts = []
    for pos in train.POSITIONS:
        result = run_position(pos)
        if result is None:
            continue
        table, verdict = result
        lines.append(f"## {train.position_spec(pos).label_position}")
        lines.append("")
        lines.append(table_to_markdown(table))
        lines.append("")
        lines.append(verdict)
        lines.append("")
        all_verdicts.append(verdict)
        print(verdict)

    OUT_PATH.write_text("\n".join(lines))
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
