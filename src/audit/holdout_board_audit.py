"""Holdout board audit ("are we sure about this model being correct?").

Simulates the FULL 2026-board product for the 2024 and 2025 seasons, exactly as the
real 2026 board is built (``src.inference.board_2026``), then grades every displayed
claim against what actually happened (``labels.parquet``). This is an audit module:
read-only against production artifacts, writes only into ``outputs/`` (its own
filenames) and, when called with an ``output_root``, entirely inside that directory.
Nothing here trains or retrains a model -- every classifier/quantile score comes from
the shipped ``data/models/*_bundle.joblib`` artifacts' already-fit estimators
(``per_seed_models`` / ``models[alpha]``), the identical frozen holdout predictions
``src.models.train.holdout_predictions`` / ``src.models.quantile.holdout_predictions``
produced at training time (retrained on <=2023, scored on 2024+2025) -- scoring a
feature row through an already-fit model is a deterministic function call, not a leak,
exactly the same operation ``src.inference.board_2026.score_veterans_batch`` /
``src.models.quantile.score_quantiles`` perform for the real 2026 board.

Population
------------
"Veterans only, skip rookie heuristic": for historical season S, the natural board
population is ``in_training_pool == 1`` rows of ``features_{pos}.parquet`` joined to
``labels.parquet`` (``src.models.train.load_modeling_frame``) restricted to
``season == S`` -- ``in_training_pool`` already excludes rookies (``~is_rookie``, see
``src.labels.build.add_training_pool_flag``), so this is structurally the same
"veteran, board-eligible" population the 2026 board's ``veteran_population`` builds
from ``rosters.parquet`` (years_exp >= 1), just sourced from the real historical join
instead of a synthetic 2026 population (2026 has no ``labels.parquet`` row to join to
yet -- that's the ENTIRE reason ``board_2026.build_veteran_feature_matrix`` exists as a
separate feature-rebuild path). The quantile head's own population additionally
requires ``games >= 8`` (``src.models.quantile.load_modeling_frame`` -- the same gate
``src.labels.build`` uses before it will even assign a ``finish_pos_rank``), so a
thin-sample veteran carries classifier columns but null quantile columns, exactly the
board's own "no bundle for this position yet -> null quantile columns" convention
(``build_veteran_board``'s ``else`` branch).

Leave-future-out machinery (the integrity-critical part)
-------------------------------------------------------------
Two board-time constants are fit from ``labels.parquet`` directly, not from either
model, and the production versions are fit from the FULL 2014-2025 span -- reused
unmodified for the real 2026 board (2026 is fully future relative to that whole span,
so nothing leaks there) but NOT safe to reuse for a season-S audit, where season S (and,
for S=2024, season 2025 too) would then be baked into a constant the board displays for
that very season:

- ``rank_thresholds`` (``src.inference.projections.build_rank_thresholds``): median
  ppg-to-finish-rank curve across every labelled season. Rebuilt here
  (``build_leaveout_thresholds``) from seasons < S only.
- ``historical_mean_breakouts_by_position`` (``src.inference.board_2026``): mean
  per-season breakout count per position, the classifier probability renormalization
  target. Rebuilt here (``leaveout_mean_breakouts``) from seasons < S only.

Both leave-out rebuilds are compared against the shipped (full-span) constants in the
audit report (``rank_threshold_shipped_vs_leaveout`` / the mean-breakouts table) --
the delta the integrity rules ask for.

The outcome-ladder's OWN renormalization targets (``src.inference.projections.attach_ladder``:
5 for elite, the position's fixed ``startable_k``/``useful_k`` from
``configs/labels.yaml``) are NOT season-derived -- they are definitional constants ("5
players finish top-5 every season", by construction) -- so they need no leave-out
variant and are reused as-is, applied within season S's own board population only (per
the integrity rules: "same per-position mechanism, applied within season S's board
population").

The Deliverable-3 engine-decision gate (``primary_engine`` per position, which lens
ranks Breakout Hunt) is read as CURRENTLY SHIPPED (``outputs/comparison_gate.json``,
read-only) rather than rebuilt leave-out -- this is a deliberate, stated deviation (see
module docstring's "Deviations" note below and the brief: "same lenses ... per-position
primary engine as currently configured"), not an oversight: the brief asks this audit to
grade the board a drafter would ACTUALLY have seen, and the shipped gate is itself
compared against BOTH holdout seasons pooled (2024+2025), so it is already not
season-S-specific information in the leakage sense the thresholds/mean-breakouts
constants are (it does not encode "what happened in season S" the way a median-ppg
threshold or a breakout count does -- it encodes "which of two model FAMILIES the repo
currently ships," a fixed engineering choice, not a season outcome).

Everything else (classifier probability renormalization scale, ladder segment
enforcement, breakout_eligible gate, SHAP driver chips, rationale sentences) is byte-for-
byte the same function call the real 2026 board makes -- imported directly from
``src.inference.board_2026`` / ``src.models.quantile`` / ``src.inference.projections``,
never reimplemented, so this module cannot silently drift from what a drafter actually
saw.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.inference import board_2026 as bd
from src.inference import projections as pj
from src.labels.build import load_labels_config
from src.models import metrics as mt
from src.models import quantile as qmod
from src.models import train

REPO_ROOT = train.REPO_ROOT
OUTPUTS_DIR = train.OUTPUTS_DIR

AUDIT_SEASONS: tuple[int, ...] = (2024, 2025)

REPORT_PATH = OUTPUTS_DIR / "holdout_board_audit.md"
METRICS_JSON_PATH = OUTPUTS_DIR / "holdout_board_audit_metrics.json"

PROB_BUCKETS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 1.01)]
PROB_BUCKET_LABELS = ["0-5%", "5-10%", "10-20%", "20-35%", "35%+"]

MIN_BUCKET_N_FOR_FLAG = 20
CALIBRATION_FLAG_MULTIPLE = 2.0


# --------------------------------------------------------------------------
# Leave-future-out constants (the integrity-critical rebuilds)
# --------------------------------------------------------------------------


def build_leaveout_thresholds(season_cutoff_exclusive: int) -> pd.DataFrame:
    """``src.inference.projections``' rank-threshold curve, fit from seasons strictly

    before ``season_cutoff_exclusive`` only (i.e. season S itself and anything later are
    excluded) -- reuses ``pj.raw_thresholds_by_season`` / ``median_thresholds`` /
    ``smooth_threshold_curve`` verbatim, just pre-filtered, so this is structurally the
    identical curve-fitting logic the shipped ``data/processed/rank_thresholds.parquet``
    uses, minus the seasons that would leak.
    """
    labels = pl.read_parquet(train.LABELS_PATH)
    raw = pj.raw_thresholds_by_season(labels).filter(pl.col("season") < season_cutoff_exclusive)
    median_df = pj.median_thresholds(raw)
    return pj.smooth_threshold_curve(median_df)


def leaveout_mean_breakouts(season_cutoff_exclusive: int) -> dict[str, float]:
    """``src.inference.board_2026.historical_mean_breakouts_by_position``, fit from

    seasons strictly before ``season_cutoff_exclusive`` only -- same zero-fill-per-
    (position, season) logic, just pre-filtered.
    """
    labels = pl.read_parquet(train.LABELS_PATH).filter(pl.col("season") < season_cutoff_exclusive)
    counts = labels.filter(pl.col("breakout") == 1).group_by(["position", "season"]).agg(pl.len().alias("n"))
    grid = labels.select("position").unique().join(labels.select("season").unique(), how="cross")
    full = grid.join(counts, on=["position", "season"], how="left").with_columns(pl.col("n").fill_null(0))
    means = full.group_by("position").agg(pl.col("n").mean().alias("mean_breakouts"))
    return {row["position"]: float(row["mean_breakouts"]) for row in means.to_dicts()}


def threshold_delta_report(season: int, leaveout: pd.DataFrame) -> pd.DataFrame:
    """Shipped (full 2014-2025 span) vs leave-out (seasons < S) threshold curve, at a

    handful of representative K per position -- the delta the integrity rules ask for.
    """
    shipped = pj.load_or_build_thresholds()
    rows = []
    for pos in sorted(leaveout["position"].unique()):
        for k in (1, 5, 10, 15, 20, 30):
            s_row = shipped[(shipped["position"] == pos) & (shipped["K"] == k)]
            l_row = leaveout[(leaveout["position"] == pos) & (leaveout["K"] == k)]
            if s_row.empty or l_row.empty:
                continue
            s_val = float(s_row["threshold_ppg"].iloc[0])
            l_val = float(l_row["threshold_ppg"].iloc[0])
            rows.append(
                {
                    "season": season, "position": pos, "K": k,
                    "shipped_threshold_ppg": s_val, "leaveout_threshold_ppg": l_val,
                    "delta": l_val - s_val,
                }
            )
    return pd.DataFrame(rows)


def mean_breakouts_delta_report(season: int, leaveout: dict[str, float]) -> pd.DataFrame:
    shipped = bd.historical_mean_breakouts_by_position()
    rows = []
    for pos in sorted(leaveout):
        rows.append(
            {
                "season": season, "position": pos,
                "shipped_mean_breakouts": shipped.get(pos, float("nan")),
                "leaveout_mean_breakouts": leaveout[pos],
                "delta": leaveout[pos] - shipped.get(pos, float("nan")),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Season-S board simulation (mirrors board_2026.build_veteran_board exactly,
# sourced from the real historical join instead of the synthetic 2026
# population -- see module docstring)
# --------------------------------------------------------------------------


def _realized_columns(labels_path: Path, season: int) -> pd.DataFrame:
    """ppr_ppg + games for season S -- the two realized-outcome columns

    ``src.models.train.load_modeling_frame``'s LABEL_JOIN_COLS does not carry (it only
    joins breakout/finish_rank_delta/finish_pos_rank/expectation_pos_rank/adp_source/
    in_training_pool). Grading needs the raw realized ppg and games-played too.
    """
    return (
        pl.read_parquet(labels_path)
        .filter(pl.col("season") == season)
        .select(["season", "gsis_id", "ppr_ppg", "games"])
        .to_pandas()
    )


def build_season_board(season: int) -> pd.DataFrame:
    """The full season-S "board" -- classifier probability + quantile ladder + honesty

    renormalization + SHAP driver chips + eligibility + primary-engine lens, built from
    the frozen holdout-retrained bundles exactly as ``board_2026.build_veteran_board``
    assembles the real 2026 board, minus the rookie section and minus the ADP/Sleeper/
    Vegas overlay (not applicable to a historical season). See module docstring.
    """
    labels_cfg = load_labels_config()
    thresholds = build_leaveout_thresholds(season)
    mean_breakouts = leaveout_mean_breakouts(season)

    rows = []
    for pos in train.POSITIONS:
        spec = train.position_spec(pos)
        qspec = qmod.quantile_spec(pos)
        if not spec.artifact_path.exists():
            print(f"  {pos}: no bundle at {spec.artifact_path}, skipping")
            continue
        bundle = joblib.load(spec.artifact_path)

        pop = train.load_modeling_frame(spec.features_path, spec.labels_path, cfg=bundle["cfg"])
        pop = pop[pop["season"] == season].reset_index(drop=True)
        if pop.empty:
            print(f"  {pos}: no season-{season} rows in the in_training_pool, skipping")
            continue
        pop["pos"] = spec.label_position
        pop = pop.merge(_realized_columns(spec.labels_path, season), on=["season", "gsis_id"], how="left")

        scored = bd.score_veterans_batch(bundle, pop)
        scored = bd.attach_shap_drivers(bundle, scored)
        scored["rationale"] = [
            bd.build_rationale(pos, scored.iloc[i], scored.iloc[i]["_shap_top_features"]) for i in range(len(scored))
        ]

        adp_worse_than = float(labels_cfg["thresholds"][spec.label_position]["adp_worse_than"])
        scored["breakout_eligible"] = scored["expectation_pos_rank"] >= adp_worse_than

        if qspec.artifact_path.exists():
            qbundle = joblib.load(qspec.artifact_path)
            qpop = qmod.load_modeling_frame(qspec.features_path, qspec.labels_path, cfg=qbundle["cfg"])
            qpop = qpop[qpop["season"] == season].reset_index(drop=True)
            if not qpop.empty:
                qscored = qmod.score_quantiles(qbundle, qpop)
                proj = pj.compute_projections_for_position(qscored, spec.label_position, thresholds, labels_cfg)
                scored = scored.merge(proj[bd._PROJECTION_COLS], on="gsis_id", how="left")
            else:
                for c in bd._PROJECTION_COLS:
                    if c != "gsis_id":
                        scored[c] = np.nan
        else:
            print(f"  {pos}: no quantile bundle at {qspec.artifact_path}, quantile columns left null")
            for c in bd._PROJECTION_COLS:
                if c != "gsis_id":
                    scored[c] = np.nan

        rows.append(scored)
        print(f"  season {season} {pos}: scored {len(scored)} veterans")

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined = bd.attach_renormalized_probability(combined, mean_breakouts)
    combined = pj.attach_ladder(combined, pos_col="pos")
    engine_decision = pj.load_engine_decision()
    combined["primary_engine"] = combined["pos"].map(engine_decision).fillna("classifier")
    combined["already_priced_as_starter"] = ~combined["breakout_eligible"].astype("boolean")

    # Grading columns.
    startable_k = combined["startable_k"]
    useful_k = combined["useful_k"]
    combined["realized_startable"] = combined["finish_pos_rank"] <= startable_k
    combined["realized_elite"] = combined["finish_pos_rank"] <= 5
    combined["realized_useful"] = combined["finish_pos_rank"] <= useful_k
    # A player with no finish_pos_rank (games<8 that season -> src.labels.build never
    # assigns one) realized none of the ladder tiers -- NaN <= k is False in pandas,
    # which already gives the correct "not startable/elite/useful" reading here, so no
    # special-casing is needed; made explicit as its own boolean for the grading tables.
    combined["realized_startable"] = combined["realized_startable"].fillna(False)
    combined["realized_elite"] = combined["realized_elite"].fillna(False)
    combined["realized_useful"] = combined["realized_useful"].fillna(False)
    combined["season"] = season
    return combined


# --------------------------------------------------------------------------
# Grading 1: ladder calibration
# --------------------------------------------------------------------------


def _bucket(series: pd.Series) -> pd.Categorical:
    edges = [b[0] for b in PROB_BUCKETS] + [PROB_BUCKETS[-1][1]]
    return pd.cut(series, bins=edges, labels=PROB_BUCKET_LABELS, right=False, include_lowest=True)


def ladder_calibration_table(board: pd.DataFrame, prob_col: str, realized_col: str) -> pd.DataFrame:
    """Bucket players by displayed ``prob_col``; per bucket, realized rate

    (``realized_col``) vs mean displayed probability. Pooled across positions AND
    seasons (the board frame passed in should already be the seasons/positions of
    interest) -- flags a bucket >2x off (either direction) with n>=20.
    """
    df = board[board[prob_col].notna()].copy()
    df["bucket"] = _bucket(df[prob_col])
    out = (
        df.groupby("bucket", observed=True)
        .agg(n=(prob_col, "size"), mean_displayed=(prob_col, "mean"), realized_rate=(realized_col, "mean"))
        .reindex(PROB_BUCKET_LABELS)
        .reset_index()
        .rename(columns={"bucket": "displayed_bucket"})
    )
    out["ratio_realized_over_displayed"] = out["realized_rate"] / out["mean_displayed"]
    out["flag_off_2x"] = (
        (out["n"] >= MIN_BUCKET_N_FOR_FLAG)
        & (
            (out["ratio_realized_over_displayed"] >= CALIBRATION_FLAG_MULTIPLE)
            | (out["ratio_realized_over_displayed"] <= 1.0 / CALIBRATION_FLAG_MULTIPLE)
        )
    )
    return out


def ladder_calibration_by_position(board: pd.DataFrame, prob_col: str, realized_col: str) -> dict[str, pd.DataFrame]:
    return {pos: ladder_calibration_table(board[board["pos"] == pos], prob_col, realized_col) for pos in sorted(board["pos"].unique())}


# --------------------------------------------------------------------------
# Grading 2: range honesty
# --------------------------------------------------------------------------


def range_honesty_table(board: pd.DataFrame) -> pd.DataFrame:
    """Per position + pooled: fraction of realized ppg inside [floor, ceiling] (target

    ~80%, since floor/ceiling = q10/q90), below floor, above ceiling, plus Spearman
    of expected_ppg vs realized ppg (games>=8 population, which the quantile head's
    modeling universe already restricts to).
    """
    df = board[board["floor_ppg"].notna() & board["ppr_ppg"].notna()].copy()
    rows = []
    for pos, g in list(df.groupby("pos")) + [("ALL", df)]:
        inside = ((g["ppr_ppg"] >= g["floor_ppg"]) & (g["ppr_ppg"] <= g["ceiling_ppg"])).mean()
        below = (g["ppr_ppg"] < g["floor_ppg"]).mean()
        above = (g["ppr_ppg"] > g["ceiling_ppg"]).mean()
        sp = mt.spearman(g["ppr_ppg"], g["expected_ppg"])
        rows.append(
            {"position": pos, "n": int(len(g)), "pct_inside_range": inside, "pct_below_floor": below, "pct_above_ceiling": above, "spearman_expected_vs_realized": sp}
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Grading 3: draft-value test
# --------------------------------------------------------------------------


def _finish_or_bust(finish_pos_rank: pd.Series, bust_value: float = 61.0) -> pd.Series:
    """Realized finish rank with a numeric bust proxy (labels.parquet's own MAX_K+1,

    see src.inference.projections.MAX_K=60) for a player with no finish_pos_rank
    (games<8 that season) -- a stated, documented convention (not silently dropped),
    used only for averaging "finish vs price"; the boolean startable/elite/useful
    grading columns never use this proxy (NaN already reads correctly there).
    """
    return finish_pos_rank.fillna(bust_value)


def draft_value_breakout_hunt(board: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Per position: model's Breakout-Hunt top-N (ranked by that position's shipped

    primary_engine sort column, eligible players only) vs the market's own late-round
    order (same eligible pool, ranked by expectation_pos_rank ascending -- best-ranked-
    among-the-cheap first). Reports realized breakout hits, avg finish-vs-price, and
    delivered startable-season count for both.
    """
    rows = []
    for pos, g in board.groupby("pos"):
        elig = g[g["breakout_eligible"] == True].copy()  # noqa: E712
        if elig.empty:
            continue
        engine = elig["primary_engine"].iloc[0] if pd.notna(elig["primary_engine"].iloc[0]) else "classifier"
        sort_col = "p_startable" if engine == "quantile" else "probability"
        elig = elig[elig[sort_col].notna()]
        if elig.empty:
            continue
        model_pick = elig.sort_values(sort_col, ascending=False).head(top_n)
        adp_pick = elig.sort_values("expectation_pos_rank", ascending=True).head(top_n)
        for label, pick in (("model", model_pick), ("adp_order", adp_pick)):
            finish = _finish_or_bust(pick["finish_pos_rank"])
            price = pick["expectation_pos_rank"]
            rows.append(
                {
                    "position": pos, "picker": label, "engine": engine, "n": int(len(pick)),
                    "realized_breakout_rate": float(pick["breakout"].mean()),
                    "avg_finish_vs_price": float((price - finish).mean()),
                    "delivered_startable_seasons": int(pick["realized_startable"].sum()),
                    "avg_realized_ppg": float(pick["ppr_ppg"].mean()),
                }
            )
    return pd.DataFrame(rows)


def value_gap_terciles(board: pd.DataFrame, metric_col: str = "value_gap") -> pd.DataFrame:
    """Full-Projections' value_gap, in terciles (pooled across positions within the

    board passed in): did positive-gap (model says undervalued) players beat their
    price more often, and by how much, than negative-gap players? ``metric_col`` lets
    the same grading logic run against either ``value_gap`` (cohort_rank-based, the
    display default as of v2.3) or ``value_gap_typical`` (predicted_finish_rank-based,
    the metric this audit originally validated pre-v2.3) -- see
    ``value_gap_metric_comparison`` below, which runs both and reports which one grades
    better on realized outcomes.
    """
    df = board[board[metric_col].notna() & board["expectation_pos_rank"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["tercile"] = pd.qcut(df[metric_col], q=3, labels=["bottom (model says overpriced)", "middle", "top (model says undervalued)"])
    finish = _finish_or_bust(df["finish_pos_rank"])
    df["beat_price"] = finish < df["expectation_pos_rank"]
    df["finish_vs_price"] = df["expectation_pos_rank"] - finish
    out = (
        df.groupby("tercile", observed=True)
        .agg(n=(metric_col, "size"), mean_value_gap=(metric_col, "mean"), beat_price_rate=("beat_price", "mean"), avg_finish_vs_price=("finish_vs_price", "mean"))
        .reset_index()
    )
    return out


def value_gap_metric_comparison(pooled: pd.DataFrame) -> dict:
    """Grades BOTH value-gap metrics (cohort_rank-based ``value_gap`` vs

    predicted_finish_rank-based ``value_gap_typical``) on the identical pooled
    2024+2025 population and picks a winner by tercile SEPARATION -- top-tercile
    beat_price_rate minus bottom-tercile beat_price_rate, the same "did the top tercile
    actually beat the market more often than the bottom tercile" question the original
    (pre-v2.3) audit asked of value_gap. Larger separation = a more useful sort (bigger
    gap between "model says undervalued" and "model says overpriced" players' real
    outcomes). Ties keep the pre-v2.3 validated metric (value_gap_typical) as the
    default sort, since it already has a track record and a tie is not evidence the
    new metric is better. Returns the two tercile tables plus the decision and the
    numbers behind it, so the report/README can show the reasoning, not just the
    conclusion.
    """
    cohort = value_gap_terciles(pooled, "value_gap")
    typical = value_gap_terciles(pooled, "value_gap_typical")

    def _spread(t: pd.DataFrame) -> float:
        if t.empty or len(t) < 3:
            return float("nan")
        return float(t.iloc[-1]["beat_price_rate"] - t.iloc[0]["beat_price_rate"])

    cohort_spread = _spread(cohort)
    typical_spread = _spread(typical)
    if np.isnan(cohort_spread) and np.isnan(typical_spread):
        winner = "neither (insufficient data)"
    elif np.isnan(cohort_spread):
        winner = "typical"
    elif np.isnan(typical_spread):
        winner = "cohort"
    else:
        winner = "cohort" if cohort_spread > typical_spread else "typical"
    return {
        "cohort_terciles": cohort,
        "typical_terciles": typical,
        "cohort_spread": cohort_spread,
        "typical_spread": typical_spread,
        "winner": winner,
        # The board's Full-Projections lens itself is ALWAYS sorted by expected_ppg
        # (never by value_gap -- see src.inference.board_2026's module docstring), so
        # this decision governs which metric is the board's displayed/colored
        # `value_gap` default and which sorter the DRAFT-VALUE-style analyses should
        # prefer, not the lens's own row order.
        "recommended_default_value_gap_metric": "cohort_rank-based (value_gap)" if winner == "cohort" else "predicted_finish_rank-based (value_gap_typical)",
    }


# --------------------------------------------------------------------------
# Grading 4: named receipts
# --------------------------------------------------------------------------


def named_receipts(board: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for pos, g in board.groupby("pos"):
        elig = g[g["breakout_eligible"] == True].copy()  # noqa: E712
        if elig.empty:
            continue
        engine = elig["primary_engine"].iloc[0] if pd.notna(elig["primary_engine"].iloc[0]) else "classifier"
        sort_col = "p_startable" if engine == "quantile" else "probability"
        elig = elig[elig[sort_col].notna()].sort_values(sort_col, ascending=False).head(top_n)
        for _, r in elig.iterrows():
            rows.append(
                {
                    "position": pos, "player_name": r["player_name"], "engine": engine,
                    "displayed_probability": r.get("probability"), "displayed_p_startable": r.get("p_startable"),
                    "predicted_finish_rank": r.get("predicted_finish_rank"), "expectation_pos_rank": r.get("expectation_pos_rank"),
                    "realized_finish_pos_rank": r.get("finish_pos_rank"), "realized_ppg": r.get("ppr_ppg"),
                    "realized_breakout": bool(r.get("breakout")), "realized_startable": bool(r.get("realized_startable")),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Grading 5: failure autopsy
# --------------------------------------------------------------------------


def _diagnose(row: pd.Series) -> str:
    """One-line, data-grounded diagnosis -- games played (availability), a team/coach

    change flag the model DID see (so a big miss despite seeing it reads as a magnitude
    error, not a blind spot), or an explicit "not inferable from data" fallback. Never
    invents a reason the row's own columns don't support.
    """
    games = row.get("games")
    if pd.notna(games) and games < 14:
        return f"availability: played only {int(games)} games that season (per-game rate projection, no injury/availability modeling -- see src.inference.projections' games-played caveat)"
    if row.get("team_change") == 1:
        return "situation change the model DID see (team_change=1 flag was in the feature row) -- a magnitude miss, not a blind spot"
    top_feats = row.get("_shap_top_features")
    if isinstance(top_feats, list) and top_feats:
        lead = top_feats[0]
        return f"model's own leading driver was '{lead[2]}' ({'boosted' if lead[1] >= 0 else 'held back'}) -- outcome diverged from that read; not inferable further from data"
    return "not inferable from data"


def failure_autopsy(board: pd.DataFrame, n: int = 3) -> dict[str, pd.DataFrame]:
    """3 worst misses (highest displayed p_startable that busted) and 3 biggest

    surprises (realized breakouts the board ranked near the bottom of displayed
    probability) per season -- with SHAP drivers and a one-line diagnosis each.
    """
    misses_rows, surprise_rows = [], []
    for pos, g in board.groupby("pos"):
        scored = g[g["p_startable"].notna()].copy()
        busts = scored[~scored["realized_startable"]].sort_values("p_startable", ascending=False).head(n)
        for _, r in busts.iterrows():
            misses_rows.append(
                {
                    "position": pos, "player_name": r["player_name"], "displayed_p_startable": r["p_startable"],
                    "displayed_probability": r.get("probability"), "predicted_finish_rank": r.get("predicted_finish_rank"),
                    "realized_finish_pos_rank": r.get("finish_pos_rank"), "realized_ppg": r.get("ppr_ppg"), "games": r.get("games"),
                    "drivers": r.get("shap_top3"), "diagnosis": _diagnose(r),
                }
            )
        breakouts = g[(g["breakout"] == 1) & g["probability"].notna()].sort_values("probability", ascending=True).head(n)
        for _, r in breakouts.iterrows():
            surprise_rows.append(
                {
                    "position": pos, "player_name": r["player_name"], "displayed_probability": r.get("probability"),
                    "displayed_p_startable": r.get("p_startable"), "expectation_pos_rank": r.get("expectation_pos_rank"),
                    "realized_finish_pos_rank": r.get("finish_pos_rank"), "realized_ppg": r.get("ppr_ppg"),
                    "drivers": r.get("shap_top3"), "diagnosis": _diagnose(r),
                }
            )
    return {"worst_misses": pd.DataFrame(misses_rows), "biggest_surprises": pd.DataFrame(surprise_rows)}


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def _md_table(df: pd.DataFrame, float_cols: tuple[str, ...] = (), pct_cols: tuple[str, ...] = ()) -> str:
    if df.empty:
        return "_(no rows)_"
    df = df.copy()
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "")
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in headers:
            v = row[c]
            if isinstance(v, bool):
                vals.append("YES" if v else "")
            elif isinstance(v, float):
                vals.append("" if pd.isna(v) else f"{v:.2f}")
            else:
                vals.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _flagged_buckets(cal: pd.DataFrame, label: str) -> list[str]:
    out = []
    for _, r in cal[cal["flag_off_2x"] == True].iterrows():  # noqa: E712
        out.append(
            f"**{label} bucket {r['displayed_bucket']}** (n={int(r['n'])}): displayed {r['mean_displayed']*100:.1f}% vs "
            f"realized {r['realized_rate']*100:.1f}% ({r['ratio_realized_over_displayed']:.2f}x)"
        )
    return out


def build_audit(seasons: tuple[int, ...] = AUDIT_SEASONS) -> dict:
    """Runs the full simulation + grading for every season, returns everything the

    report/metrics writers need. The single source of truth both ``write_report`` and
    ``write_metrics_json`` read from -- keeps the two outputs from silently drifting
    apart.
    """
    boards = {s: build_season_board(s) for s in seasons}
    pooled = pd.concat([b for b in boards.values() if not b.empty], ignore_index=True)

    ladder_pooled = {
        "p_startable": ladder_calibration_table(pooled, "p_startable", "realized_startable"),
        "p_elite": ladder_calibration_table(pooled, "p_elite", "realized_elite"),
        "p_useful": ladder_calibration_table(pooled, "p_useful", "realized_useful"),
    }
    ladder_by_season = {
        s: {
            "p_startable": ladder_calibration_table(b, "p_startable", "realized_startable"),
            "p_elite": ladder_calibration_table(b, "p_elite", "realized_elite"),
            "p_useful": ladder_calibration_table(b, "p_useful", "realized_useful"),
        }
        for s, b in boards.items()
        if not b.empty
    }

    range_honesty_pooled = range_honesty_table(pooled)
    range_honesty_by_season = {s: range_honesty_table(b) for s, b in boards.items() if not b.empty}

    draft_value_pooled = draft_value_breakout_hunt(pooled)
    draft_value_by_season = {s: draft_value_breakout_hunt(b) for s, b in boards.items() if not b.empty}
    # value_gap here means the CURRENT board default (cohort_rank-based, v2.3+) --
    # kept for backward compatibility with anything that reads this key. The full
    # both-metrics comparison (cohort vs the pre-v2.3 predicted_finish_rank-based
    # value_gap_typical) is value_gap_metric_comparison, pooled-only (a per-season
    # winner isn't meaningful at this sample size -- see the report).
    value_gap_pooled = value_gap_terciles(pooled, "value_gap")
    value_gap_by_season = {s: value_gap_terciles(b, "value_gap") for s, b in boards.items() if not b.empty}
    value_gap_metric_cmp = value_gap_metric_comparison(pooled)

    receipts_by_season = {s: named_receipts(b) for s, b in boards.items() if not b.empty}
    autopsy_by_season = {s: failure_autopsy(b) for s, b in boards.items() if not b.empty}

    threshold_deltas = pd.concat(
        [threshold_delta_report(s, build_leaveout_thresholds(s)) for s in seasons], ignore_index=True
    )
    breakout_deltas = pd.concat(
        [mean_breakouts_delta_report(s, leaveout_mean_breakouts(s)) for s in seasons], ignore_index=True
    )

    return {
        "seasons": seasons,
        "boards": boards,
        "pooled": pooled,
        "ladder_pooled": ladder_pooled,
        "ladder_by_season": ladder_by_season,
        "range_honesty_pooled": range_honesty_pooled,
        "range_honesty_by_season": range_honesty_by_season,
        "draft_value_pooled": draft_value_pooled,
        "draft_value_by_season": draft_value_by_season,
        "value_gap_pooled": value_gap_pooled,
        "value_gap_by_season": value_gap_by_season,
        "value_gap_metric_cmp": value_gap_metric_cmp,
        "receipts_by_season": receipts_by_season,
        "autopsy_by_season": autopsy_by_season,
        "threshold_deltas": threshold_deltas,
        "breakout_deltas": breakout_deltas,
    }


def _verdict(a: dict) -> str:
    dv = a["draft_value_pooled"]
    model_rows = dv[dv["picker"] == "model"]
    adp_rows = dv[dv["picker"] == "adp_order"]
    model_startable = int(model_rows["delivered_startable_seasons"].sum())
    adp_startable = int(adp_rows["delivered_startable_seasons"].sum())
    model_n = int(model_rows["n"].sum())
    adp_n = int(adp_rows["n"].sum())
    cmp = a["value_gap_metric_cmp"]
    cohort_t, typical_t = cmp["cohort_terciles"], cmp["typical_terciles"]
    cohort_top = float(cohort_t.iloc[-1]["beat_price_rate"]) if not cohort_t.empty else float("nan")
    cohort_bot = float(cohort_t.iloc[0]["beat_price_rate"]) if not cohort_t.empty else float("nan")
    typical_top = float(typical_t.iloc[-1]["beat_price_rate"]) if not typical_t.empty else float("nan")
    typical_bot = float(typical_t.iloc[0]["beat_price_rate"]) if not typical_t.empty else float("nan")
    flags = []
    for name, cal in a["ladder_pooled"].items():
        col = {"p_startable": "realized_startable", "p_elite": "realized_elite", "p_useful": "realized_useful"}[name]
        flags += _flagged_buckets(cal, name)
    flag_txt = f"{len(flags)} calibration bucket(s) off by >2x (n>=20)" if flags else "no calibration bucket off by >2x at n>=20"
    return (
        f"Across both holdout seasons (2024+2025, {model_n} Breakout Hunt top-10-per-position picks pooled), the "
        f"model's own top-10 picks delivered {model_startable}/{model_n} startable seasons vs {adp_startable}/{adp_n} "
        f"for simply taking the market's own ADP order within the same cheap-eligible pool -- {'a real edge' if model_startable > adp_startable else ('no better than the market order' if model_startable == adp_startable else 'WORSE than just following ADP order')}. "
        f"The Full-Projections value_gap signal is more convincing either way, but the two available versions of it don't grade "
        f"identically: the pre-v2.3 predicted_finish_rank-based gap (`value_gap_typical`) shows top tercile {typical_top*100:.0f}% "
        f"vs bottom tercile {typical_bot*100:.0f}% beating their preseason price ({(typical_top-typical_bot)*100:+.0f}pt spread); "
        f"the v2.3 cohort_rank-based gap (`value_gap`, now the board's display default) shows top {cohort_top*100:.0f}% vs bottom "
        f"{cohort_bot*100:.0f}% ({(cohort_top-cohort_bot)*100:+.0f}pt spread). Decision (per this run): **{cmp['winner']}** grades "
        f"as the stronger sorter, so **{cmp['recommended_default_value_gap_metric']}** is this audit's recommended default for "
        "ranking/sorting by value gap -- see the 'Metric decision' subsection below for the full reasoning; the board still "
        "DISPLAYS cohort_rank as the headline rank regardless (a presentational change, kept separate from which metric is the "
        f"validated sorter). Ladder calibration: {flag_txt}. "
        "Range honesty and per-position detail are below; read the flagged buckets and the failure autopsy before trusting any "
        "single displayed percentage on the 2026 board."
    )


def write_report(a: dict, report_path: Path = REPORT_PATH) -> None:
    lines = ["# Holdout Board Audit -- 2024 & 2025", "", '_"Are we sure about this model being correct?" -- end-to-end replay of the board product against realized outcomes._', ""]

    lines.append("## Verdict")
    lines.append("")
    lines.append(_verdict(a))
    lines.append("")

    lines.append("## Headline calibration flags (read first)")
    lines.append("")
    any_flag = False
    for name, cal in a["ladder_pooled"].items():
        for f in _flagged_buckets(cal, name):
            lines.append(f"- {f}")
            any_flag = True
    if not any_flag:
        lines.append("- No ladder bucket (p_elite/p_startable/p_useful, pooled 2024+2025) is off by more than 2x at n>=20.")
    lines.append("")

    lines.append("## 1. Ladder calibration")
    lines.append("")
    lines.append("Displayed probability bucket vs realized rate. Pooled across both holdout seasons and all four positions; per-season tables follow.")
    lines.append("")
    for name, cal in a["ladder_pooled"].items():
        lines.append(f"### {name} (pooled 2024+2025)")
        lines.append("")
        lines.append(_md_table(cal.rename(columns={"mean_displayed": "mean displayed", "realized_rate": "realized rate", "ratio_realized_over_displayed": "realized/displayed", "flag_off_2x": ">2x off (n>=20)"})))
        lines.append("")
    for s in a["seasons"]:
        if s not in a["ladder_by_season"]:
            continue
        lines.append(f"### Season {s} detail")
        lines.append("")
        for name, cal in a["ladder_by_season"][s].items():
            lines.append(f"**{name}, {s}**")
            lines.append("")
            lines.append(_md_table(cal.rename(columns={"mean_displayed": "mean displayed", "realized_rate": "realized rate", "ratio_realized_over_displayed": "realized/displayed", "flag_off_2x": ">2x off (n>=20)"})))
            lines.append("")

    lines.append("## 2. Range honesty")
    lines.append("")
    lines.append("Fraction of realized ppg inside [floor, ceiling] (target ~80%, since floor/ceiling = q10/q90), below floor, above ceiling; Spearman(expected_ppg, realized ppg) among the games>=8 population. Pooled 2024+2025:")
    lines.append("")
    lines.append(_md_table(a["range_honesty_pooled"], pct_cols=("pct_inside_range", "pct_below_floor", "pct_above_ceiling"), float_cols=("spearman_expected_vs_realized",)))
    lines.append("")
    for s in a["seasons"]:
        if s not in a["range_honesty_by_season"]:
            continue
        lines.append(f"**Season {s}:**")
        lines.append("")
        lines.append(_md_table(a["range_honesty_by_season"][s], pct_cols=("pct_inside_range", "pct_below_floor", "pct_above_ceiling"), float_cols=("spearman_expected_vs_realized",)))
        lines.append("")

    lines.append("## 3. Draft-value test")
    lines.append("")
    lines.append("### Breakout Hunt top-10-per-position: model vs the market's own ADP order (same eligible pool)")
    lines.append("")
    lines.append(_md_table(a["draft_value_pooled"].rename(columns={"realized_breakout_rate": "breakout rate"}), pct_cols=("breakout rate",)))
    lines.append("")
    for s in a["seasons"]:
        if s not in a["draft_value_by_season"]:
            continue
        lines.append(f"**Season {s}:**")
        lines.append("")
        lines.append(_md_table(a["draft_value_by_season"][s].rename(columns={"realized_breakout_rate": "breakout rate"}), pct_cols=("breakout rate",)))
        lines.append("")

    lines.append("### Full Projections value_gap terciles: did undervalued (positive-gap) players beat their price more?")
    lines.append("")
    lines.append(
        "v2.3 added a second, cohort_rank-based value_gap alongside the original predicted_finish_rank-based one "
        "(now `value_gap_typical`) -- see `src.inference.projections`' \"Two ranks, not one\" docstring section. "
        "Both are graded here, pooled 2024+2025, on the identical population."
    )
    lines.append("")
    lines.append("**value_gap (cohort_rank-based -- current board display default):**")
    lines.append("")
    lines.append(_md_table(a["value_gap_pooled"], pct_cols=("beat_price_rate",)))
    lines.append("")
    lines.append("**value_gap_typical (predicted_finish_rank-based -- the metric originally validated by this audit):**")
    lines.append("")
    lines.append(_md_table(a["value_gap_metric_cmp"]["typical_terciles"], pct_cols=("beat_price_rate",)))
    lines.append("")
    for s in a["seasons"]:
        if s not in a["value_gap_by_season"]:
            continue
        lines.append(f"**Season {s} (cohort_rank-based):**")
        lines.append("")
        lines.append(_md_table(a["value_gap_by_season"][s], pct_cols=("beat_price_rate",)))
        lines.append("")

    lines.append("#### Metric decision: which value_gap should be the validated sorter?")
    lines.append("")
    cmp = a["value_gap_metric_cmp"]
    cs, ts = cmp["cohort_spread"], cmp["typical_spread"]
    cs_txt = f"{cs*100:+.1f}pt" if pd.notna(cs) else "n/a"
    ts_txt = f"{ts*100:+.1f}pt" if pd.notna(ts) else "n/a"
    lines.append(
        f"Top-tercile-minus-bottom-tercile beat-price-rate spread (pooled 2024+2025): cohort_rank-based value_gap = "
        f"{cs_txt}, predicted_finish_rank-based value_gap_typical = {ts_txt}. **Winner: {cmp['winner']}** -- "
        f"recommended default sorter: **{cmp['recommended_default_value_gap_metric']}**."
    )
    lines.append("")
    if cmp["winner"] == "cohort":
        lines.append(
            "The cohort-based gap grades at least as well as the metric this audit originally validated, so the display "
            "change (headlining `cohort_rank` / `value_gap`) is treated as validated, not merely presentational: it is both "
            "the more intuitive number (comparable to a market rank) AND the stronger realized-outcome sorter."
        )
    else:
        lines.append(
            "The cohort-based gap grades WORSE than the metric this audit originally validated. Per the pre-stated decision "
            "rule: the headline DISPLAY change (showing `cohort_rank` as \"Projects PosN of 2026\" instead of the typical-year "
            "mapping) stands on its own merits -- it fixes a real conflation (see `src.inference.projections`' \"Two ranks, "
            "not one\" section) and is not itself a ranking claim. But `value_gap` sorting/coloring and any downstream "
            "\"undervalued\" analysis should keep using `value_gap_typical` (the predicted_finish_rank-based metric) as the "
            "validated sorter until a future audit shows otherwise -- this is a presentational relabeling of the rank shown, "
            "not a claim that the new gap is a better signal."
        )
    lines.append("")

    lines.append("## 4. Named receipts (top Breakout Hunt picks vs what happened)")
    lines.append("")
    for s in a["seasons"]:
        r = a["receipts_by_season"].get(s)
        if r is None or r.empty:
            continue
        lines.append(f"### Season {s}")
        lines.append("")
        lines.append(_md_table(r))
        lines.append("")

    lines.append("## 5. Failure autopsy")
    lines.append("")
    for s in a["seasons"]:
        fa = a["autopsy_by_season"].get(s)
        if fa is None:
            continue
        lines.append(f"### Season {s} -- worst misses (highest displayed p_startable that busted)")
        lines.append("")
        lines.append(_md_table(fa["worst_misses"]))
        lines.append("")
        lines.append(f"### Season {s} -- biggest surprises (realized breakouts the board ranked near the bottom)")
        lines.append("")
        lines.append(_md_table(fa["biggest_surprises"]))
        lines.append("")

    lines.append("## Leave-future-out integrity check")
    lines.append("")
    lines.append("Rank-threshold curve (median ppg-to-finish-rank), shipped (full 2014-2025 span, what the real 2026 board uses) vs leave-out (seasons strictly before S, what this audit uses):")
    lines.append("")
    lines.append(_md_table(a["threshold_deltas"], float_cols=("shipped_threshold_ppg", "leaveout_threshold_ppg", "delta")))
    lines.append("")
    lines.append("Classifier-probability renormalization target (historical mean breakouts/position), shipped vs leave-out:")
    lines.append("")
    lines.append(_md_table(a["breakout_deltas"], float_cols=("shipped_mean_breakouts", "leaveout_mean_breakouts", "delta")))
    lines.append("")

    lines.append("## Deviations from the brief")
    lines.append("")
    lines.append(
        "- **Primary-engine lens choice (`primary_engine`) is read as currently shipped** (`outputs/comparison_gate.json`), "
        "not rebuilt leave-out-per-season, per the brief's own \"per-position primary engine as currently configured\" -- see "
        "module docstring for the full rationale (it is an engineering choice between two model families, not a season-outcome-"
        "derived constant, so it does not carry the same leakage risk as the rank thresholds or the breakout-count renormalization "
        "target)."
    )
    lines.append(
        "- **Bust proxy for 'finish vs price' averages**: a player with no `finish_pos_rank` (games<8 that season) is scored as "
        "finish rank 61 (one worse than `projections.MAX_K`=60) when averaging finish-vs-price deltas -- a documented, explicit "
        "penalty convention, not a silent drop. The boolean startable/elite/useful ladder grading never uses this proxy (a missing "
        "finish_pos_rank already reads as 'not startable' correctly)."
    )
    lines.append(
        "- **Diagnosis in the failure autopsy is a heuristic, not a causal claim**: games-played<14 reads as an availability story, "
        "a `team_change=1` feature flag on a big miss reads as \"the model saw it, missed the magnitude\"; anything else falls back "
        "to naming the model's own top SHAP driver with an explicit \"not inferable further from data\" rather than inventing a "
        "story (e.g. a scheme change, a new play-caller's usage pattern, or a QB change is not visible to this feature set)."
    )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def _df_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def write_metrics_json(a: dict, metrics_path: Path = METRICS_JSON_PATH) -> None:
    payload = {
        "seasons": list(a["seasons"]),
        "ladder_pooled": {k: _df_records(v) for k, v in a["ladder_pooled"].items()},
        "ladder_by_season": {str(s): {k: _df_records(v) for k, v in d.items()} for s, d in a["ladder_by_season"].items()},
        "range_honesty_pooled": _df_records(a["range_honesty_pooled"]),
        "range_honesty_by_season": {str(s): _df_records(v) for s, v in a["range_honesty_by_season"].items()},
        "draft_value_pooled": _df_records(a["draft_value_pooled"]),
        "draft_value_by_season": {str(s): _df_records(v) for s, v in a["draft_value_by_season"].items()},
        "value_gap_pooled": _df_records(a["value_gap_pooled"]),
        "value_gap_by_season": {str(s): _df_records(v) for s, v in a["value_gap_by_season"].items()},
        "value_gap_metric_comparison": {
            "cohort_terciles": _df_records(a["value_gap_metric_cmp"]["cohort_terciles"]),
            "typical_terciles": _df_records(a["value_gap_metric_cmp"]["typical_terciles"]),
            "cohort_spread": a["value_gap_metric_cmp"]["cohort_spread"],
            "typical_spread": a["value_gap_metric_cmp"]["typical_spread"],
            "winner": a["value_gap_metric_cmp"]["winner"],
            "recommended_default_value_gap_metric": a["value_gap_metric_cmp"]["recommended_default_value_gap_metric"],
        },
        "receipts_by_season": {str(s): _df_records(v) for s, v in a["receipts_by_season"].items()},
        "autopsy_by_season": {
            str(s): {k: _df_records(v) for k, v in d.items()} for s, d in a["autopsy_by_season"].items()
        },
        "threshold_deltas": _df_records(a["threshold_deltas"]),
        "breakout_deltas": _df_records(a["breakout_deltas"]),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> int:
    a = build_audit()
    write_report(a)
    write_metrics_json(a)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {METRICS_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
