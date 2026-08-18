"""BreakoutLab dashboard builder (v2 / plain-language multi-view upgrade).

``make dashboard`` (this module's ``main``) writes ``outputs/dashboard/index.html``:
one self-contained page, all data embedded as a single JSON payload inside a
``<script type="application/json">`` tag, read by vanilla client-side JS. No
external requests at view time except Google Fonts.

Every number in the payload traces back to a real artifact on disk, verified
against its actual schema before use (see each section below):

1. **Board rows** -- ``outputs/breakout_board_2026.csv`` (the checked-in
   deliverable), both sections (veteran + rookie), read as-is.
2. **Feature profiles** -- reuses ``src.inference.board_2026.build_veteran_feature_matrix``
   (imported, not reimplemented) to get each scored veteran's actual 2026
   feature row, and ``src.explain.shap_report.global_mean_abs_importance``
   (also reused, not reimplemented) to pick each position's top-N SHAP
   features on the pooled validation frame. Each player's within-position
   percentile on those features is computed here, freshly, from the 2026
   population.
3. **ECR market drift** -- ``data/raw/ff_ecr.parquet``'s 2026 preseason
   snapshots (page_type=redraft-overall, fp_page=ppr-cheatsheets,
   ecr_type=ro, scrape_date>=2026-06-01), reusing ``src.ingest.adp``'s own
   snapshot-window filter (``_load_ecr_preseason``) and
   ``src.ingest.id_map.match_to_gsis`` for name resolution -- the identical
   machinery ``src.ingest.adp`` itself uses. The per-date dedupe/positional-rank
   step is NOT reused as-is (see ``_rank_snapshot_deterministic``'s
   docstring): ``adp._rank_snapshot`` breaks same-ecr ties via polars'
   default ``maintain_order=False`` sort/unique, which is fine for adp's own
   single-snapshot use but not reproducible enough for this module's
   build-twice-byte-identical contract, so the same dedupe/rank *semantics*
   are re-applied here with an explicit tiebreak key instead.
4. **Model trust** -- ``outputs/model_{pos}_metrics.json`` (holdout/validation
   PR-AUC, baselines, blend weights) plus, since the named holdout top-10
   hit/miss lists are NOT present in that JSON (verified directly -- see
   ``main``'s printed report), a small markdown-table parse of
   ``outputs/model_{pos}_report.md``'s "Holdout top-10 named lists" section,
   which every position's report carries in an identical format.
5. **Meta** -- snapshot date (latest 2026 ECR scrape_date actually used to
   build the board), model-generation timestamp (the four bundle files'
   mtime, NOT ``datetime.now()`` -- see "Determinism" below), and counts.

Determinism
-----------
``tests/test_dashboard.py`` asserts building twice from unchanged inputs
produces byte-identical HTML. There is therefore no wall-clock timestamp
anywhere in the payload -- ``meta.model_generated_at`` is derived from
``os.path.getmtime`` on the model bundle files, which does not change
between two back-to-back builds. Every float is rounded to 3dp and every
date truncated to an ISO day before serialization, both to keep the payload
small and to avoid float/format noise between runs.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.dashboard.template import render_html
from src.explain import shap_report as shp
from src.inference import board_2026 as bd
from src.inference import projections
from src.ingest import adp
from src.ingest.id_map import build_id_map, match_to_gsis
from src.labels.build import load_labels_config
from src.models import quantile as qmod
from src.models import train

REPO_ROOT = train.REPO_ROOT
OUTPUTS_DIR = train.OUTPUTS_DIR
DASHBOARD_DIR = OUTPUTS_DIR / "dashboard"
DASHBOARD_HTML_PATH = DASHBOARD_DIR / "index.html"
LABELS_PARQUET_PATH = train.PROCESSED_DIR / "labels.parquet"

POSITIONS = train.POSITIONS  # ("wr", "rb", "te", "qb")
N_TOP_FEATURES = 15
ECR_WINDOW_START = dt.date(2026, 6, 1)
BASELINE_NAMES = ("adp_knows_best", "prior_season_ppg_rank", "age_adjusted_adp")
BASELINE_LABELS = {
    "adp_knows_best": "Market rank alone",
    "prior_season_ppg_rank": "Last year's points alone",
    "age_adjusted_adp": "Market rank + age",
}

# Plain-language feature names -- copied verbatim from the plain-language
# board template's FEAT map (see the artifact this dashboard extends), so
# the two surfaces speak about a feature identically.
FEAT = {
    "expectation_pos_rank": "Cheap draft price", "receptions_pg_n1": "Caught a lot of passes last yr",
    "targets_pg_n1": "Heavily targeted last yr", "rec_yards_pg_n1": "Receiving yardage",
    "ppr_ppg_n1": "Last year's scoring", "ppr_ppg_yoy_delta": "Scoring trending",
    "expected_ppr_ppg_n1": "Earned more than he scored", "efficiency_residual_pg_yoy_delta": "Efficiency trending",
    "target_share_n1": "Big slice of team's targets", "target_share_yoy_delta": "Target share trending",
    "air_yards_share_n1": "Deep-ball usage", "snap_share_n1": "Playing-time share", "team_change": "New team",
    "team_pass_rate_prior": "Pass-happy offense", "team_rush_att_pg_prior": "Run-heavy offense",
    "vacated_target_share": "Targets freed up by departures", "vacated_carry_share": "Carries freed up by departures",
    "competition_draft_capital": "Team drafted his competition", "draft_pick": "His own draft pedigree", "age": "Age",
    "rz_target_share_n1": "Red-zone usage", "ez_target_share_yoy_delta": "Goal-line targets trending",
    "goal_line_carry_share_yoy_delta": "Goal-line carries trending", "carry_share_yoy_delta": "Carry share trending",
    "yards_per_carry_n1": "Yards per carry", "rush_td_rate_n1": "Rushing TD rate", "rush_yard_share_n1": "Runs a lot for a QB",
    "rush_yard_share_yoy_delta": "QB rushing trending", "pass_attempts_pg_yoy_delta": "Pass volume trending",
    "sack_rate_yoy_delta": "Sack rate trending", "td_rate_n1": "TD rate", "weighted_opportunity_pg_n1": "Total touches",
    "catch_percentage_n1": "Sure hands", "catch_percentage_yoy_delta": "Catch rate trending",
}


def _fallback_label(code: str) -> str:
    """Plain-ish label for a SHAP top feature the template's FEAT map doesn't cover.

    Reuses ``src.inference.board_2026``'s own humanizer (built for the
    on-board rationale sentences) rather than inventing a second translation
    table.
    """
    label = bd._humanize_feature(code)
    return label[0].upper() + label[1:] if label else code


def feature_label(code: str) -> str:
    return FEAT.get(code) or _fallback_label(code)


# --------------------------------------------------------------------------
# Sanitizing for json.dumps(..., allow_nan=False)
# --------------------------------------------------------------------------


def _sanitize(obj):
    if obj is None:
        return None
    if isinstance(obj, bool) or isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 3)
    if isinstance(obj, (pd.Timestamp, dt.date, dt.datetime)):
        return str(obj)[:10]
    if isinstance(obj, (int, str)):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# --------------------------------------------------------------------------
# 1. Board rows
# --------------------------------------------------------------------------


def load_board() -> pd.DataFrame:
    df = pd.read_csv(bd.BOARD_CSV_PATH)
    df["is_rookie"] = df["section"] == "rookie (heuristic)"
    df["key"] = df["player_name"].astype(str) + "|" + df["pos"].astype(str)
    df["display_prob"] = np.where(df["is_rookie"], df["probability_heuristic"], df["probability"])
    # v2.0: hunt_score is whichever engine Deliverable 3's comparison gate picked as
    # primary for that position (classifier `probability` or quantile `p_startable`) --
    # computed once here so the client never has to re-derive the gate decision itself.
    if "primary_engine" in df.columns:
        df["hunt_score"] = np.where(df["primary_engine"] == "quantile", df.get("p_startable"), df["probability"])
    else:
        df["hunt_score"] = df["probability"]
    return df


def board_rows(board: pd.DataFrame, ecr_by_key: dict) -> list[dict]:
    rows = []
    for _, r in board.iterrows():
        drift = ecr_by_key.get(r["key"])
        rows.append(
            {
                "key": r["key"],
                "n": r["player_name"],
                "p": r["pos"],
                "a": r["age"] if pd.notna(r["age"]) else None,
                "t": r["team"] if pd.notna(r["team"]) else None,
                "rk": bool(r["is_rookie"]),
                "pr": r["display_prob"] if pd.notna(r["display_prob"]) else None,
                "rs": r["raw_score"] if pd.notna(r["raw_score"]) else None,
                "tier": r["tier"] if pd.notna(r.get("tier")) else None,
                "brm": r["base_rate_multiple"] if pd.notna(r.get("base_rate_multiple")) else None,
                "e": r["consensus_ecr_pos_rank"] if pd.notna(r["consensus_ecr_pos_rank"]) else None,
                "dr": r["draft_round"] if pd.notna(r["draft_round"]) else None,
                "d": r["expected_rank_delta"] if pd.notna(r["expected_rank_delta"]) else None,
                "r": r["rationale"] if pd.notna(r["rationale"]) else (r["note"] if pd.notna(r["note"]) else None),
                "s": r["shap_top3"] if pd.notna(r["shap_top3"]) else None,
                "adp": r["adp"] if pd.notna(r["adp"]) else None,
                "sadp": r["sleeper_adp_pos_rank"] if pd.notna(r["sleeper_adp_pos_rank"]) else None,
                "gap": r["adp_gap"] if pd.notna(r["adp_gap"]) else None,
                "vegas": r["implied_pts"] if pd.notna(r["implied_pts"]) else None,
                "edge": r["edge"] if pd.notna(r["edge"]) else None,
                "avail": r["availability"] if pd.notna(r["availability"]) else None,
                "sat": bool(r["probability_saturated"]) if pd.notna(r["probability_saturated"]) else None,
                "drift": drift,
                # v2.1 addendum: "broke out last season" transparency badge -- players who
                # already posted breakout==1 in 2025 read as confusing on THIS year's board
                # without an explanation (e.g. Daniel Jones). Presentation only.
                "bo": bool(r["broke_out_last_season"]) if pd.notna(r.get("broke_out_last_season")) else False,
                # v2.0 quantile projections (Deliverable 4) -- null for a position with
                # no quantile bundle yet, same "absent -> None" convention as every
                # column above.
                "hs": r["hunt_score"] if pd.notna(r.get("hunt_score")) else None,
                "elig": bool(r["breakout_eligible"]) if pd.notna(r.get("breakout_eligible")) else None,
                "eng": r["primary_engine"] if pd.notna(r.get("primary_engine")) else None,
                "fppg": r["floor_ppg"] if pd.notna(r.get("floor_ppg")) else None,
                "eppg": r["expected_ppg"] if pd.notna(r.get("expected_ppg")) else None,
                "cppg": r["ceiling_ppg"] if pd.notna(r.get("ceiling_ppg")) else None,
                "pfr": r["predicted_finish_rank"] if pd.notna(r.get("predicted_finish_rank")) else None,
                "pel": r["p_elite"] if pd.notna(r.get("p_elite")) else None,
                "pst": r["p_startable"] if pd.notna(r.get("p_startable")) else None,
                "pus": r["p_useful"] if pd.notna(r.get("p_useful")) else None,
                "seg": (
                    [r["seg_elite"], r["seg_starter"], r["seg_useful"], r["seg_bust"]]
                    if pd.notna(r.get("seg_elite"))
                    else None
                ),
                "vg": r["value_gap"] if pd.notna(r.get("value_gap")) else None,
                "aps": bool(r["already_priced_as_starter"]) if pd.notna(r.get("already_priced_as_starter")) else None,
                "psent": r["projection_sentence"] if pd.notna(r.get("projection_sentence")) else None,
            }
        )
    return rows


# --------------------------------------------------------------------------
# 2. Feature matrices + percentiles (reuses board_2026 + shap_report)
# --------------------------------------------------------------------------


def build_feature_matrices(raw: dict, bundles: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """{pos: pandas df} -- calls board_2026.build_veteran_feature_matrix per position

    (the exact same function that produced the checked-in board's underlying
    2026 population), never reimplementing its feature logic.
    """
    out = {}
    for pos, bundle in bundles.items():
        feats = bd.build_veteran_feature_matrix(pos, raw, bundle)
        out[pos] = feats.to_pandas()
    return out


def top_shap_features(bundles: dict[str, dict], top_n: int = N_TOP_FEATURES) -> dict[str, list[str]]:
    """{pos: [feature names]} -- the position's top-N SHAP drivers on the pooled

    validation frame (src.explain.shap_report.global_mean_abs_importance,
    reused as-is -- the identical global-importance computation the SHAP
    beeswarm report itself uses).
    """
    out = {}
    for pos, bundle in bundles.items():
        tree_cols = bundle["tree_feature_cols"]
        val_df = shp.pooled_validation_frame(pos, bundle)
        table = shp.global_mean_abs_importance(bundle, val_df[tree_cols], tree_cols, top_n=top_n)
        out[pos] = table["feature"].tolist()
    return out


def feature_profiles(feats: dict[str, pd.DataFrame], top_feats: dict[str, list[str]]) -> dict[str, list[dict]]:
    """{player key: [{feat, label, value, pctl}, ...]} -- within-position percentile

    (0-100, ``rank(pct=True)*100``) of each top-SHAP feature, computed fresh
    on the 2026 population each position's feature matrix carries.
    """
    profiles: dict[str, list[dict]] = {}
    for pos, df in feats.items():
        cols = [c for c in top_feats.get(pos, []) if c in df.columns]
        pct = df[cols].rank(pct=True, na_option="keep") * 100.0
        for i in range(len(df)):
            key = f"{df.iloc[i]['player_name']}|{pos.upper()}"
            entries = []
            for c in cols:
                val = df.iloc[i][c]
                p = pct.iloc[i][c]
                if pd.isna(val) or pd.isna(p):
                    continue
                entries.append({"feat": c, "label": feature_label(c), "value": float(val), "pctl": float(p)})
            profiles[key] = entries
    return profiles


# --------------------------------------------------------------------------
# 3. ECR market-drift trajectory (reuses src.ingest.adp's snapshot helpers)
# --------------------------------------------------------------------------


def name_pos_to_gsis(feats: dict[str, pd.DataFrame], rookie_feats: pl.DataFrame | None) -> dict[tuple[str, str], str]:
    """{(player_name, POS): gsis_id} built from the same feature frames the board's

    rows were scored from -- board_2026's board CSV carries no gsis_id column
    of its own, so this reconstructs the join key from the frames that do.
    """
    out: dict[tuple[str, str], str] = {}
    for pos, df in feats.items():
        for name, gid in zip(df["player_name"], df["gsis_id"]):
            out[(name, pos.upper())] = gid
    if rookie_feats is not None:
        rp = rookie_feats.to_pandas()
        for name, posn, gid in zip(rp["player_name"], rp["position"], rp["gsis_id"]):
            out.setdefault((name, posn), gid)
    return out


def load_rookie_feature_frame() -> pl.DataFrame | None:
    if bd.ROOKIE_FEATURES_PATH.exists():
        return pl.read_parquet(bd.ROOKIE_FEATURES_PATH)
    return None


def _rank_snapshot_deterministic(snap: pl.DataFrame) -> pl.DataFrame:
    """Same semantics as ``adp._rank_snapshot`` (skill-position filter, id-based

    dedupe keeping the best/lowest ecr, positional ordinal rank) but with
    every tie broken by an explicit, reproducible key. ``adp._rank_snapshot``
    itself relies on plain ``.sort(ecr_col)`` / ``.unique(...)`` calls, which
    polars defaults to ``maintain_order=False`` (i.e. NOT guaranteed stable,
    verified directly: two back-to-back calls on identical input produced
    different tie order for same-ecr players) -- fine for adp's own use (an
    arbitrary tie order among equal-ecr players is a don't-care there), but
    not for this dashboard's build-twice-byte-identical contract
    (tests/test_dashboard.py). Sorting on a fully-specifying key
    (id, falling back to player name when id is null) leaves no tie for the
    final ordinal rank to break arbitrarily, so the result is reproducible
    regardless of the underlying sort algorithm's stability.
    """
    snap = snap.filter(pl.col("pos").is_in(adp.SKILL_POSITIONS))
    snap = snap.with_columns(pl.coalesce([pl.col("id"), pl.col("player")]).alias("_tiebreak"))
    with_id = (
        snap.filter(pl.col("id").is_not_null())
        .sort(["ecr", "_tiebreak"])
        .unique(subset=["id"], keep="first", maintain_order=True)
    )
    without_id = snap.filter(pl.col("id").is_null())
    merged = pl.concat([with_id, without_id], how="vertical_relaxed")
    merged = merged.sort(["pos", "ecr", "_tiebreak"])
    merged = merged.with_columns(
        pl.col("ecr").rank(method="ordinal").over("pos").cast(pl.Int64).alias("pos_rank")
    )
    return merged.drop("_tiebreak")


def build_ecr_trajectory(crosswalk: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """All-snapshot ranked ECR rows (season 2026, window >= ECR_WINDOW_START), resolved

    to gsis_id, plus a small coverage-report dict for the final printed
    summary. Reuses ``src.ingest.adp``'s own snapshot filter
    (``_load_ecr_preseason``) and per-date dedupe/rank
    (``_rank_snapshot``) -- applied to every snapshot date in the window
    instead of only the latest one, which is all ``src.ingest.adp`` itself
    needs.
    """
    ecr = adp._load_ecr_preseason()
    window = ecr.filter(pl.col("scrape_date") >= ECR_WINDOW_START).filter(pl.col("pos").is_in(adp.SKILL_POSITIONS))
    dates = sorted(window.select("scrape_date").unique().get_column("scrape_date").to_list())

    parts = [_rank_snapshot_deterministic(window.filter(pl.col("scrape_date") == d)) for d in dates]
    all_ranked = pl.concat(parts, how="vertical_relaxed") if parts else window

    distinct = all_ranked.filter(pl.col("id").is_not_null()).unique(subset=["id"], keep="first").select(
        "id", "player", "pos"
    )
    matched = match_to_gsis(
        distinct.rename({"player": "player_name", "pos": "position"}),
        name_col="player_name",
        pos_col="position",
        fp_id_col="id",
        season=2026,
        crosswalk=crosswalk,
    )
    all_ranked = all_ranked.join(matched.select("id", "gsis_id"), on="id", how="left")

    report = {
        "snapshot_dates": [str(d) for d in dates],
        "n_snapshot_dates": len(dates),
        "n_distinct_players": distinct.height,
        "n_matched": int(matched.filter(pl.col("gsis_id").is_not_null()).height),
    }
    return all_ranked, report


def ecr_series_by_gsis(all_ranked: pl.DataFrame) -> dict[str, dict]:
    """{gsis_id: {dates, ranks, drift}} -- drift = earliest pos_rank - latest pos_rank,

    positive = market rising on him (rank number falling over time).
    """
    tagged = all_ranked.filter(pl.col("gsis_id").is_not_null()).sort(["gsis_id", "scrape_date"])
    grouped = tagged.group_by("gsis_id", maintain_order=True).agg(
        pl.col("scrape_date").alias("dates"), pl.col("pos_rank").alias("ranks")
    )
    out = {}
    for gid, dates, ranks in zip(
        grouped.get_column("gsis_id").to_list(),
        grouped.get_column("dates").to_list(),
        grouped.get_column("ranks").to_list(),
    ):
        out[gid] = {
            "dates": [str(d) for d in dates],
            "ranks": [int(r) for r in ranks],
            "drift": int(ranks[0] - ranks[-1]),
            "n": len(ranks),
        }
    return out


def ecr_by_board_key(board: pd.DataFrame, crosswalk_map: dict, ecr_by_gsis: dict) -> dict[str, dict]:
    out = {}
    for key, name, pos in zip(board["key"], board["player_name"], board["pos"]):
        gid = crosswalk_map.get((name, pos))
        series = ecr_by_gsis.get(gid) if gid else None
        if series is not None:
            out[key] = series
    return out


# --------------------------------------------------------------------------
# 4. Model trust: metrics JSON + named holdout top-10 lists (parsed from .md)
# --------------------------------------------------------------------------

_HOLDOUT_LIST_RE = re.compile(
    r"## Holdout top-10 named lists\n\n### 2024\n\n(?P<y2024>.*?)\n\n### 2025\n\n(?P<y2025>.*?)\n\n## ",
    re.S,
)


def _parse_md_table(block: str) -> list[dict]:
    lines = [ln for ln in block.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:  # skip header + separator
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def parse_named_holdout_lists(report_path: Path) -> dict:
    """{"2024": [...], "2025": [...]} of {player, probability, expectation_rank,

    actual_rank, hit} -- parsed from outputs/model_{pos}_report.md's "Holdout
    top-10 named lists" section, since (verified directly) the metrics JSON
    does NOT carry these named lists, only aggregate pr_auc/top10_precision.
    """
    if not report_path.exists():
        return {}
    text = report_path.read_text()
    m = _HOLDOUT_LIST_RE.search(text)
    if not m:
        return {}
    out = {}
    for year_key in ("y2024", "y2025"):
        year = year_key[1:]
        rows = _parse_md_table(m.group(year_key))
        parsed = []
        for row in rows:
            try:
                prob = float(row.get("Calibrated probability", ""))
            except ValueError:
                prob = None
            parsed.append(
                {
                    "player": row.get("Player"),
                    "prob": prob,
                    "expect_rank": row.get("Expectation rank"),
                    "actual_rank": row.get("Actual finish rank"),
                    "hit": (row.get("Breakout", "").strip().lower() == "yes"),
                }
            )
        out[year] = parsed
    return out


def _metric_row(rows: list[dict], model: str, season) -> dict | None:
    for r in rows:
        if r.get("model") == model and r.get("season") == season:
            return r
    return None


def build_quantile_trust() -> dict:
    """{pos: {holdout coverage/spearman/pinball_q50, gate decision}} -- v2.0's Deliverable-1

    holdout metrics (``src.models.quantile``) plus Deliverable-3's comparison-gate decision
    (``outputs/comparison_gate.json``, written by ``src.inference.projections.comparison_gate``),
    read fresh here (never re-run) so the dashboard build stays fast and side-effect-free.
    """
    out = {}
    gate_by_pos = {}
    gate_path = projections.COMPARISON_GATE_JSON_PATH
    if gate_path.exists():
        for row in json.loads(gate_path.read_text()):
            gate_by_pos[row["position"]] = row
    for pos in POSITIONS:
        qspec = qmod.quantile_spec(pos)
        entry = {"coverage_q10_q90": None, "spearman_q50_actual": None, "pinball_q50": None, "gate": None}
        if qspec.metrics_json_path.exists():
            m = json.loads(qspec.metrics_json_path.read_text())["holdout_metrics"]
            entry["coverage_q10_q90"] = m.get("coverage_q10_q90")
            entry["spearman_q50_actual"] = m.get("spearman_q50_actual")
            entry["pinball_q50"] = (m.get("pinball_by_alpha") or {}).get("0.50")
        gate_row = gate_by_pos.get(pos.upper())
        if gate_row:
            entry["gate"] = {
                "primary_engine": gate_row["primary_engine"],
                "quantile_top10_precision": gate_row["quantile_top10_precision"],
                "classifier_top10_precision": gate_row["classifier_top10_precision"],
            }
        out[pos] = entry
    return out


def build_trust_view() -> tuple[dict, dict]:
    """(trust_payload, report_dict) -- report_dict is for main()'s printed report,

    not embedded in the page.
    """
    trust = {"positions": {}, "named_lists": {}, "quantile": build_quantile_trust()}
    report = {}
    for pos in POSITIONS:
        spec = train.position_spec(pos)
        if not spec.metrics_json_path.exists():
            continue
        metrics = json.loads(spec.metrics_json_path.read_text())
        holdout = metrics.get("holdout_metrics", [])
        validation = metrics.get("validation_metrics", [])

        model_pooled = _metric_row(holdout, "model", "pooled") or {}
        best_baseline_name, best_baseline_pr_auc = None, None
        for bname in BASELINE_NAMES:
            row = _metric_row(holdout, bname, "pooled")
            if row and (best_baseline_pr_auc is None or row["pr_auc"] > best_baseline_pr_auc):
                best_baseline_name, best_baseline_pr_auc = bname, row["pr_auc"]

        trust["positions"][pos] = {
            "holdout_pr_auc": model_pooled.get("pr_auc"),
            "holdout_top10_precision": model_pooled.get("top10_precision"),
            "holdout_n": model_pooled.get("n"),
            "holdout_n_pos": model_pooled.get("n_pos"),
            "best_baseline_name": BASELINE_LABELS.get(best_baseline_name, best_baseline_name),
            "best_baseline_pr_auc": best_baseline_pr_auc,
            "validation_pooled_pr_auc": (_metric_row(validation, "model", "pooled") or {}).get("pr_auc"),
            "blend_weights": metrics.get("blend_weights"),
            "holdout_regression_rmse": metrics.get("holdout_regression_rmse"),
        }

        named = parse_named_holdout_lists(spec.report_path)
        if named:
            trust["named_lists"][pos] = named
            report[pos] = {
                "metrics_json_has_named_lists": False,
                "report_md_has_named_lists": True,
                "n_2024": len(named.get("2024", [])),
                "n_2025": len(named.get("2025", [])),
            }
        else:
            report[pos] = {"metrics_json_has_named_lists": False, "report_md_has_named_lists": False}
    return trust, report


# --------------------------------------------------------------------------
# Positions view: counts, historical base rate, probability distribution
# --------------------------------------------------------------------------


def historical_base_rates() -> dict[str, float]:
    """Per-position historical breakout rate (mean of ``breakout`` over the

    training pool, labels.parquet) -- the real number behind the "a random
    late pick has a 3-6% chance" claim, computed fresh rather than
    hardcoded.
    """
    if not LABELS_PARQUET_PATH.exists():
        return {}
    df = pl.read_parquet(LABELS_PARQUET_PATH).filter(pl.col("in_training_pool") == 1)
    rates = df.group_by("position").agg(pl.col("breakout").mean().alias("rate"))
    return {row["position"]: row["rate"] for row in rates.to_dicts()}


def build_positions_view(board: pd.DataFrame, base_rates: dict[str, float]) -> dict:
    out = {}
    vets = board[~board["is_rookie"]]
    for pos in POSITIONS:
        sub = vets[vets["pos"] == pos.upper()]
        if sub.empty:
            continue
        top5 = sub.sort_values("display_prob", ascending=False).head(5)
        bins = [0] * 10
        for p in sub["display_prob"].dropna():
            idx = min(9, int(p * 10))
            bins[idx] += 1
        out[pos] = {
            "count": int(len(sub)),
            "base_rate": base_rates.get(pos.upper()),
            "top5": [
                {"n": r["player_name"], "pr": r["display_prob"] if pd.notna(r["display_prob"]) else None}
                for _, r in top5.iterrows()
            ],
            "histogram": bins,
        }
    return out


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------


def build_meta(board: pd.DataFrame, bundles: dict[str, dict], ecr_report: dict) -> dict:
    mtimes = [os.path.getmtime(train.position_spec(pos).artifact_path) for pos in bundles]
    model_generated_at = (
        dt.datetime.fromtimestamp(max(mtimes)).date().isoformat() if mtimes else None
    )
    snapshot_date = max(ecr_report.get("snapshot_dates", ["2026-08-14"])) if ecr_report.get("snapshot_dates") else None
    vets = board[~board["is_rookie"]]
    rooks = board[board["is_rookie"]]
    return {
        "season": 2026,
        "snapshot_date": snapshot_date,
        "model_generated_at": model_generated_at,
        "counts": {
            "veterans": int(len(vets)),
            "rookies": int(len(rooks)),
            "by_position": {pos: int((vets["pos"] == pos.upper()).sum()) for pos in POSITIONS},
        },
        # per-position historical base rates live on payload.positions[pos].base_rate
        # (build_positions_view) -- not duplicated here.
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def assemble_payload() -> tuple[dict, dict]:
    """Returns (payload, report) -- report is printed by main(), never embedded."""
    board = load_board()

    raw = bd.load_raw_frames()
    bundles = {}
    for pos in POSITIONS:
        spec = train.position_spec(pos)
        if spec.artifact_path.exists():
            bundles[pos] = joblib.load(spec.artifact_path)

    feats = build_feature_matrices(raw, bundles)
    top_feats = top_shap_features(bundles)
    profiles = feature_profiles(feats, top_feats)

    rookie_feats = load_rookie_feature_frame()
    crosswalk_map = name_pos_to_gsis(feats, rookie_feats)

    crosswalk = build_id_map()
    all_ranked, ecr_report = build_ecr_trajectory(crosswalk)
    ecr_gsis = ecr_series_by_gsis(all_ranked)
    ecr_key = ecr_by_board_key(board, crosswalk_map, ecr_gsis)
    coverage_3plus = sum(1 for v in ecr_gsis.values() if v["n"] >= 3)

    trust, trust_report = build_trust_view()
    base_rates = historical_base_rates()
    positions_view = build_positions_view(board, base_rates)
    meta = build_meta(board, bundles, ecr_report)

    rows = board_rows(board, ecr_key)

    payload = {
        "meta": meta,
        "board": rows,
        "features": profiles,
        "ecr": ecr_key,
        "positions": positions_view,
        "trust": trust,
        "labels_config": _labels_config_summary(),
    }

    report = {
        "payload_players_features": len(profiles),
        "payload_players_ecr": len(ecr_key),
        "ecr_snapshot_report": ecr_report,
        "ecr_players_with_ge3_snapshots": coverage_3plus,
        "ecr_total_players_matched": len(ecr_gsis),
        "trust_report": trust_report,
        "positions_view_counts": {k: v["count"] for k, v in positions_view.items()},
        "top_features_by_position": top_feats,
    }
    return payload, report


def _labels_config_summary() -> dict:
    cfg = load_labels_config()
    return {
        "thresholds": cfg["thresholds"],
        "min_games": cfg["min_games"],
    }


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------


def build_dashboard() -> tuple[Path, dict]:
    payload, report = assemble_payload()
    clean = _sanitize(payload)
    # sort_keys=True: dict key order from polars group_by/pandas conversions is
    # not guaranteed stable across process runs even with unchanged inputs
    # (hash-based grouping) -- sorting keys makes the serialized bytes
    # deterministic regardless, without needing to chase every such source
    # upstream. List order (board rows, ECR date/rank series) is untouched by
    # this -- only dict *key* order is affected, and nothing downstream reads
    # dict key order as meaningful.
    payload_json = json.dumps(clean, allow_nan=False, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    html_text = render_html(payload_json, clean["meta"])

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_HTML_PATH.write_text(html_text, encoding="utf-8")
    report["html_bytes"] = len(html_text.encode("utf-8"))
    return DASHBOARD_HTML_PATH, report


def main() -> int:
    path, report = build_dashboard()
    print(f"wrote {path} ({report['html_bytes']:,} bytes)")
    print(f"board players with feature profiles: {report['payload_players_features']}")
    print(f"board players with ECR drift series: {report['payload_players_ecr']}")
    print(
        f"ECR trajectory: {report['ecr_snapshot_report']['n_snapshot_dates']} snapshot dates, "
        f"{report['ecr_total_players_matched']} players matched, "
        f"{report['ecr_players_with_ge3_snapshots']} with >=3 snapshots"
    )
    for pos, r in report["trust_report"].items():
        print(f"  {pos}: metrics.json named lists={r['metrics_json_has_named_lists']}, report.md named lists={r['report_md_has_named_lists']}")
    print(f"positions view counts: {report['positions_view_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
