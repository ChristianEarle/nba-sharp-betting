"""v2.0 derived quantities from the quantile head's per-player PPR-ppg distribution

(``src.models.quantile``) -- rank-to-ppg threshold curves, per-player finish-rank /
outcome-ladder probabilities, honesty renormalization, and the Deliverable-3
comparison gate against the v1.7 classifier. Nothing here trains a model; every
function takes an already-scored (q10..q90) frame and turns it into board-ready
columns.

Rank-to-ppg thresholds
------------------------
For a position and a finish rank K, "the ppg needed to finish rank K" is computed
per season directly off ``labels.parquet`` (the K-th ranked qualifying player's
``ppr_ppg`` that season, among players with a non-null ``finish_pos_rank``), then
the MEDIAN of that value across every season 2014-2025 is taken as the raw
threshold for that (position, K). A monotone-decreasing (isotonic) smoothing pass
turns the resulting noisy step sequence into a full curve for every
K=1..``MAX_K`` -- "the threshold can only get easier as K gets worse" is a
first-principles property of a finish-rank curve, not an assumption being
imposed for convenience. Persisted to ``data/processed/rank_thresholds.parquet``.

Per-player probabilities
--------------------------
A player's 5 non-crossing quantiles ((0.10,q10), (0.25,q25), ..., (0.90,q90)) are
treated as 5 points on his personal ppg CDF: P(ppg <= q_p) = p. ``P(finish <= K)``
for ANY K is then ``P(ppg >= T_pos(K))`` = ``1 - CDF(T_pos(K))``, where
``CDF(threshold)`` is read off those 5 points by linear interpolation between them,
and linearly EXTRAPOLATED (using the nearest segment's slope) for a threshold
outside [q10, q90] -- clamped to [0.02, 0.98] so no player is ever displayed as a
mathematical certainty or impossibility from 5 fitted points. ``predicted_finish_rank``
is the inverse of the same curve: the K where T_pos(K) crosses the player's own q50,
linearly interpolated and clamped to [1, MAX_K].

Outcome ladder (elite / startable / useful / bust)
-----------------------------------------------------
Four cumulative probabilities, all read off the identical per-player CDF above, no
new training:
  - ``p_elite``      = P(finish <= 5)
  - ``p_startable``  = P(finish <= position's startable K: QB12/RB15/WR18/TE8)
  - ``p_useful``     = P(finish <= 2x startable K: QB24/RB30/WR36/TE16 -- flex/bench value)
Each is independently rescaled per position (see "Honesty normalization" below), then
a fourth, MUTUALLY EXCLUSIVE decomposition is derived for display (a stacked bar, not
a new model output): elite, starter-not-elite (p_startable - p_elite), useful-not-
startable (p_useful - p_startable), and bust (1 - p_useful) -- these four sum to
exactly 1 for every player by construction once the cumulative values are correctly
ordered (see below).

Honesty normalization (v1.7 pattern, generalized)
----------------------------------------------------
Exactly ``src.inference.board_2026.attach_renormalized_probability``'s mechanism
(a per-position linear rescale so the DISPLAYED sum matches a known target, capped at
0.97) applied three times, independently, over ALL scored veterans at a position (not
just eligible ones -- unlike v1.7's classifier probability, p_startable here is a
property of every player, eligible or not): p_elite sums to 5 (exactly 5 players finish
top-5 every season, by definition), p_startable sums to the position's startable count K,
p_useful sums to 2K. Each rescale is a single positive per-position constant, so it can
only shrink/grow every player's value by the same factor and cannot change within-
position ranking.

Scaling each of the three curves independently is not guaranteed to preserve
p_elite <= p_startable <= p_useful for every individual player even though each curve's
OWN aggregate sum comes out exactly right -- so after scaling, ordering is enforced by
clipping UPWARD cumulatively (p_startable = max(p_startable, p_elite), then
p_useful = max(p_useful, p_startable), each still capped at 0.97): a player's floor never
drops, higher-inclusion probabilities are only ever pulled up to match, never down. This
is a display-layer patch on top of three otherwise-independent rescales, not a claim that
any curve's aggregate target is still hit to the decimal after the clip -- verified to
stay within +/-1% in practice (tests/test_quantile.py) because violations are rare and
small.

Eligibility & value gap
--------------------------
``breakout_eligible`` reuses the exact label-eligibility gate
(``configs/labels.yaml``'s ``adp_worse_than``, per position) -- a player capped one slot
behind the deepest market signal (``adp_source == "capped"``) gets a large
``expectation_pos_rank`` and therefore counts as eligible automatically, no special case
needed. ``value_gap = expectation_pos_rank - predicted_finish_rank``; positive means the
market has him going LATER (worse rank number) than the model expects him to finish
(better rank number) -- undervalued.

Games-played caveat
----------------------
Every quantity here is a PER-GAME (ppg) projection. Injury/availability risk is NOT
modeled anywhere in this pipeline -- a player projected for 14.0 ppg who plays 6 games
is not the same fantasy asset as one who plays 17, and nothing here distinguishes them.
This is a known, deliberate limitation (see README), not an oversight.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression

from src.labels.build import load_labels_config
from src.models import metrics as mt
from src.models import quantile as qm
from src.models import train

REPO_ROOT = train.REPO_ROOT
PROCESSED_DIR = train.PROCESSED_DIR
OUTPUTS_DIR = train.OUTPUTS_DIR

RANK_THRESHOLDS_PATH = PROCESSED_DIR / "rank_thresholds.parquet"
COMPARISON_GATE_JSON_PATH = OUTPUTS_DIR / "comparison_gate.json"
COMPARISON_GATE_MD_PATH = OUTPUTS_DIR / "comparison_gate.md"

MAX_K = 60
DISPLAY_CAP = 0.97
CDF_LO, CDF_HI = 0.02, 0.98
ELITE_K = 5

HOLDOUT_SEASONS = (2024, 2025)


# --------------------------------------------------------------------------
# Rank-to-ppg threshold curve (Deliverable 2)
# --------------------------------------------------------------------------


def raw_thresholds_by_season(labels: pl.DataFrame) -> pl.DataFrame:
    """(season, position, K, ppg) for every player-season with a real finish_pos_rank --

    K is just that player's own finish_pos_rank; "the K-th ranked player's ppg that
    season" falls out of grouping by (season, position, K) with exactly one row each
    (finish_pos_rank is a within-season-position ordinal, so no two players in the same
    season/position ever share a K).
    """
    q = labels.filter(pl.col("finish_pos_rank").is_not_null())
    return q.select(pl.col("season"), pl.col("position"), pl.col("finish_pos_rank").alias("K"), pl.col("ppr_ppg"))


def median_thresholds(raw: pl.DataFrame, max_k: int = MAX_K) -> pd.DataFrame:
    """(position, K) -> median ppg across every 2014-2025 season that reached that K."""
    out = (
        raw.filter(pl.col("K") <= max_k)
        .group_by(["position", "K"])
        .agg(pl.col("ppr_ppg").median().alias("ppg"), pl.len().alias("n_seasons"))
        .sort(["position", "K"])
    )
    return out.to_pandas()


def smooth_threshold_curve(median_df: pd.DataFrame, max_k: int = MAX_K) -> pd.DataFrame:
    """Isotonic (monotone DECREASING in K, out_of_bounds="clip") smoothing of the raw

    per-K medians into a full K=1..max_k curve per position -- see module docstring.
    """
    rows = []
    for pos, g in median_df.groupby("position"):
        g = g.sort_values("K")
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(g["K"].to_numpy(dtype=float), g["ppg"].to_numpy(dtype=float))
        full_k = np.arange(1, max_k + 1, dtype=float)
        smoothed = iso.predict(full_k)
        for k, v in zip(full_k, smoothed):
            rows.append({"position": pos, "K": int(k), "threshold_ppg": float(v)})
    return pd.DataFrame(rows).sort_values(["position", "K"]).reset_index(drop=True)


def build_rank_thresholds(labels_path: Path = train.LABELS_PATH, out_path: Path = RANK_THRESHOLDS_PATH) -> pd.DataFrame:
    labels = pl.read_parquet(labels_path)
    raw = raw_thresholds_by_season(labels)
    median_df = median_thresholds(raw)
    smoothed = smooth_threshold_curve(median_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    smoothed.to_parquet(out_path, index=False)
    return smoothed


def load_or_build_thresholds(out_path: Path = RANK_THRESHOLDS_PATH) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_parquet(out_path)
    return build_rank_thresholds(out_path=out_path)


def threshold_lookup(thresholds: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{position: (K ascending 1..MAX_K, threshold_ppg matching, descending)}."""
    out = {}
    for pos, g in thresholds.groupby("position"):
        g = g.sort_values("K")
        out[pos] = (g["K"].to_numpy(dtype=float), g["threshold_ppg"].to_numpy(dtype=float))
    return out


# --------------------------------------------------------------------------
# Per-player finish-rank inversion + outcome-ladder CDF
# --------------------------------------------------------------------------


def predicted_finish_rank(q50: np.ndarray, position: str, lookup: dict) -> np.ndarray:
    """K where T_pos(K) crosses q50 -- linear interpolation, clamped to [1, MAX_K]."""
    k_arr, thr_arr = lookup[position]
    thr_asc = thr_arr[::-1]  # threshold ascending <-> K descending (60 -> 1)
    k_desc = k_arr[::-1]
    return np.clip(np.interp(q50, thr_asc, k_desc), 1.0, float(MAX_K))


def _cdf_at(threshold: np.ndarray, quantile_points: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """P(ppg <= threshold) per row, via linear interpolation of that row's (ppg, alpha)

    points, linearly EXTRAPOLATED beyond the outermost points using the nearest
    segment's slope, clamped to [CDF_LO, CDF_HI]. ``quantile_points`` is (n, 5) -- one
    row per player, ascending (non-crossing already enforced upstream).
    """
    threshold = np.asarray(threshold, dtype=float)
    n = quantile_points.shape[0]
    out = np.empty(n, dtype=float)
    for i in range(n):
        x = quantile_points[i]
        t = threshold[i]
        if t <= x[0]:
            dx = x[1] - x[0]
            slope = (alphas[1] - alphas[0]) / dx if dx > 1e-9 else 0.0
            cdf = alphas[0] + slope * (t - x[0])
        elif t >= x[-1]:
            dx = x[-1] - x[-2]
            slope = (alphas[-1] - alphas[-2]) / dx if dx > 1e-9 else 0.0
            cdf = alphas[-1] + slope * (t - x[-1])
        else:
            cdf = float(np.interp(t, x, alphas))
        out[i] = min(max(cdf, CDF_LO), CDF_HI)
    return out


def p_finish_le_k(K: float, position: str, quantile_points: np.ndarray, alphas: np.ndarray, lookup: dict) -> np.ndarray:
    """P(finish <= K) = 1 - CDF(T_pos(K)) for every row -- see module docstring."""
    k_arr, thr_arr = lookup[position]
    threshold = float(np.interp(K, k_arr, thr_arr))
    cdf = _cdf_at(np.full(quantile_points.shape[0], threshold), quantile_points, alphas)
    return 1.0 - cdf


# --------------------------------------------------------------------------
# Per-position projections (called once per position, before the multi-
# position honesty-renormalization pass -- same two-stage shape as
# board_2026.attach_renormalized_probability).
# --------------------------------------------------------------------------


def compute_projections_for_position(scored: pd.DataFrame, label_position: str, thresholds: pd.DataFrame, labels_cfg: dict) -> pd.DataFrame:
    """``scored`` must already carry q10..q90 (``src.models.quantile.score_quantiles``)

    and ``expectation_pos_rank``. Attaches floor/expected/ceiling ppg,
    predicted_finish_rank, the three RAW (pre-honesty-normalization) cumulative
    ladder probabilities, breakout_eligible, value_gap, and each position's
    startable_k/useful_k (carried through so the multi-position renormalization pass
    doesn't need to re-derive them).
    """
    lookup = threshold_lookup(thresholds)
    alphas = np.array(sorted(qm.ALPHAS))
    qcols = [qm.ALPHA_COL[a] for a in alphas]
    quantile_points = scored[qcols].to_numpy(dtype=float)

    out = scored.copy()
    out["floor_ppg"] = out[qm.ALPHA_COL[0.10]]
    out["expected_ppg"] = out[qm.ALPHA_COL[0.50]]
    out["ceiling_ppg"] = out[qm.ALPHA_COL[0.90]]
    out["predicted_finish_rank"] = predicted_finish_rank(out[qm.ALPHA_COL[0.50]].to_numpy(dtype=float), label_position, lookup)

    startable_k = int(labels_cfg["thresholds"][label_position]["finish_top"])
    useful_k = 2 * startable_k
    out["startable_k"] = startable_k
    out["useful_k"] = useful_k
    out["p_elite_raw"] = p_finish_le_k(ELITE_K, label_position, quantile_points, alphas, lookup)
    out["p_startable_raw"] = p_finish_le_k(startable_k, label_position, quantile_points, alphas, lookup)
    out["p_useful_raw"] = p_finish_le_k(useful_k, label_position, quantile_points, alphas, lookup)

    adp_worse_than = float(labels_cfg["thresholds"][label_position]["adp_worse_than"])
    out["breakout_eligible"] = out["expectation_pos_rank"] >= adp_worse_than
    out["value_gap"] = out["expectation_pos_rank"] - out["predicted_finish_rank"]
    return out


# --------------------------------------------------------------------------
# Honesty renormalization + outcome-ladder segments (multi-position pass)
# --------------------------------------------------------------------------


def attach_ladder(board: pd.DataFrame, pos_col: str = "pos") -> pd.DataFrame:
    """Independently rescale p_elite_raw/p_startable_raw/p_useful_raw per position so

    each sums, over every scored veteran at that position, to its target (5, K, 2K --
    ``startable_k``/``useful_k`` columns), cap at DISPLAY_CAP, then enforce
    p_elite <= p_startable <= p_useful per player by clipping upward cumulatively -- see
    module docstring's "Honesty normalization" section. Adds p_elite/p_startable/p_useful
    (displayed, cumulative) and seg_elite/seg_starter/seg_useful/seg_bust (mutually
    exclusive, sum to 1 per row).
    """
    board = board.copy()
    for c in ("p_elite", "p_startable", "p_useful", "renorm_scale_elite", "renorm_scale_startable", "renorm_scale_useful"):
        board[c] = np.nan

    for pos in board[pos_col].dropna().unique():
        mask = (board[pos_col] == pos).to_numpy()
        if mask.sum() == 0:
            continue
        k = float(board.loc[mask, "startable_k"].iloc[0])
        specs = {"elite": (5.0, "p_elite_raw"), "startable": (k, "p_startable_raw"), "useful": (2.0 * k, "p_useful_raw")}
        scaled = {}
        for name, (target, raw_col) in specs.items():
            raw = board.loc[mask, raw_col].to_numpy(dtype=float)
            total = float(raw.sum())
            scale = (target / total) if total > 0 else 1.0
            scaled[name] = np.minimum(DISPLAY_CAP, raw * scale)
            board.loc[mask, f"renorm_scale_{name}"] = scale

        elite = scaled["elite"]
        startable = np.minimum(DISPLAY_CAP, np.maximum(scaled["startable"], elite))
        useful = np.minimum(DISPLAY_CAP, np.maximum(scaled["useful"], startable))

        board.loc[mask, "p_elite"] = elite
        board.loc[mask, "p_startable"] = startable
        board.loc[mask, "p_useful"] = useful

    board["seg_elite"] = board["p_elite"]
    board["seg_starter"] = board["p_startable"] - board["p_elite"]
    board["seg_useful"] = board["p_useful"] - board["p_startable"]
    board["seg_bust"] = 1.0 - board["p_useful"]
    return board


# --------------------------------------------------------------------------
# Deliverable 3: comparison gate vs the v1.7 classifier
# --------------------------------------------------------------------------


def _classifier_pooled_top10(metrics_json: dict) -> float:
    for row in metrics_json.get("holdout_metrics", []):
        if row.get("model") == "model" and row.get("season") == "pooled":
            return float(row.get("top10_precision", float("nan")))
    return float("nan")


def comparison_gate() -> pd.DataFrame:
    """On the SAME holdout rows (season in {2024,2025}, ``in_training_pool==1`` -- the

    identical population ``train.holdout_predictions`` scores) the v1.7 classifier was
    judged on: rank ELIGIBLE players by quantile-derived RAW p_startable (pre-honesty-
    renormalization -- a per-position-season positive rescale cannot change
    within-position ranking, so raw and renormalized give an identical top-10; see
    module docstring) and compute top-10 precision against breakout==1, pooled across
    both holdout seasons -- exactly how ``train.evaluate_scores`` computes the
    classifier's own "model"/"pooled"/"top10_precision". DECISION (pre-stated, binding):
    quantile becomes the board's primary engine for a position iff its precision is
    >= the classifier's; ties favor quantile (richer output wins ties). Written to
    ``outputs/comparison_gate.{json,md}``.
    """
    labels_cfg = load_labels_config()
    thresholds = load_or_build_thresholds()
    rows = []
    for pos in train.POSITIONS:
        cspec = train.position_spec(pos)
        qspec = qm.quantile_spec(pos)
        if not (cspec.artifact_path.exists() and qspec.artifact_path.exists() and cspec.metrics_json_path.exists()):
            continue
        cbundle = joblib.load(cspec.artifact_path)
        qbundle = joblib.load(qspec.artifact_path)

        cdf = train.load_modeling_frame(cspec.features_path, cspec.labels_path, cfg=cbundle["cfg"])
        holdout = cdf[cdf["season"].isin(HOLDOUT_SEASONS)].reset_index(drop=True)
        clf_top10 = _classifier_pooled_top10(json.loads(cspec.metrics_json_path.read_text()))

        if holdout.empty:
            rows.append(
                {"position": cspec.label_position, "quantile_top10_precision": float("nan"), "classifier_top10_precision": clf_top10, "n_eligible_holdout": 0, "primary_engine": "classifier"}
            )
            continue

        scored = qm.score_quantiles(qbundle, holdout)
        proj = compute_projections_for_position(scored, cspec.label_position, thresholds, labels_cfg)
        eligible = proj[proj["breakout_eligible"]].reset_index(drop=True)

        if len(eligible) == 0:
            q_top10 = float("nan")
        else:
            idx = mt.top_k_indices(eligible["p_startable_raw"].to_numpy(), 10)
            q_top10 = float(eligible.iloc[idx]["breakout"].to_numpy().mean())

        decision = "quantile" if (not np.isnan(q_top10) and (np.isnan(clf_top10) or q_top10 >= clf_top10)) else "classifier"
        rows.append(
            {
                "position": cspec.label_position,
                "quantile_top10_precision": q_top10,
                "classifier_top10_precision": clf_top10,
                "n_eligible_holdout": int(len(eligible)),
                "primary_engine": decision,
            }
        )
    table = pd.DataFrame(rows)
    _write_gate_outputs(table)
    return table


def _write_gate_outputs(table: pd.DataFrame) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_GATE_JSON_PATH.write_text(json.dumps(table.to_dict(orient="records"), indent=2, default=str))
    lines = [
        "# Deliverable 3 -- comparison gate: quantile p_startable vs v1.7 classifier",
        "",
        "Pre-stated, binding decision rule: the quantile head becomes a position's primary board",
        "engine iff its top-10 precision (ranking ELIGIBLE holdout players by raw p_startable) is",
        ">= the v1.7 classifier's own holdout top-10 precision, on the identical 2024+2025 holdout",
        "rows. Ties favor quantile.",
        "",
        "| Position | Quantile top-10 precision | Classifier top-10 precision | n eligible (holdout) | Primary engine |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['position']} | {r['quantile_top10_precision']:.3f} | {r['classifier_top10_precision']:.3f} | "
            f"{r['n_eligible_holdout']} | **{r['primary_engine']}** |"
        )
    COMPARISON_GATE_MD_PATH.write_text("\n".join(lines) + "\n")


def load_engine_decision() -> dict[str, str]:
    """{label_position: "quantile"|"classifier"} -- falls back to "classifier" for every

    position if the gate hasn't been run yet (safe default: keep shipping the proven
    v1.7 board until the gate says otherwise).
    """
    if not COMPARISON_GATE_JSON_PATH.exists():
        return {p: "classifier" for p in ("QB", "RB", "WR", "TE")}
    rows = json.loads(COMPARISON_GATE_JSON_PATH.read_text())
    decision = {r["position"]: r["primary_engine"] for r in rows}
    for p in ("QB", "RB", "WR", "TE"):
        decision.setdefault(p, "classifier")
    return decision


def main() -> int:
    print("building rank_thresholds.parquet")
    thresholds = build_rank_thresholds()
    print(f"  wrote {RANK_THRESHOLDS_PATH} ({len(thresholds)} rows)")
    print("running Deliverable-3 comparison gate")
    table = comparison_gate()
    print(table.to_string(index=False))
    print(f"  wrote {COMPARISON_GATE_JSON_PATH}")
    print(f"  wrote {COMPARISON_GATE_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
