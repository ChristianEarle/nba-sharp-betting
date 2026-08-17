# BreakoutLab

Preseason machine-learning pipeline that predicts NFL fantasy breakouts. For every
fantasy-relevant skill player (QB/RB/WR/TE) it is intended to output a calibrated
breakout probability, an expected finish-vs-ADP delta, and the SHAP drivers behind
each call.

> **Status: Phase 1a complete.** nflverse ingest is built and cached; nothing is
> modelled yet. See [Progress](#progress) for what is and is not done.

## Setup

```bash
uv sync --extra dev
uv run python -m src.ingest.nflverse   # pull + cache nflverse data (~40s cold)
uv run pytest -q
```

Python 3.11+. The raw cache lands in `data/` and is gitignored; re-running the
ingest is idempotent and reuses parquet unless `--force` is passed.

## Layout

```
src/ingest/      nflverse pulls, ADP acquisition, ID crosswalk
src/features/    per-position feature builders
src/labels/      breakout definitions + labeling
src/models/      training, CV, calibration, ensemble
src/explain/     SHAP, reports
src/inference/   current-season scoring + ADP/Vegas overlay
configs/         data, label, scoring and override config (YAML/CSV)
archive/         previous unrelated contents of this repo; not part of the build
```

## Data

Thirteen seasons (2013–2025) pulled via **`nflreadpy`** — not `nfl_data_py`, which
was deprecated and archived in September 2025. All loaders return polars.

| Dataset | Rows | Coverage |
|---------|------|----------|
| `player_stats` (weekly) | 234,738 | 2013–2025 |
| `snap_counts` | 324,611 | 2013–2025 |
| `depth_charts` | 993,055 | 2013–**2024** |
| `rosters` | 37,485 | 2013–2025 |
| `schedules` | 3,562 | 2013–2025 |
| `draft_picks` | 12,670 | full history |
| `combine` | 8,649 | full history |
| `players` | 25,033 | full history |
| `ff_playerids` | 12,472 | crosswalk |
| `ff_opportunity` | 74,731 | 2013–2025 |
| `ngs_passing / receiving / rushing` | 5,933 / 14,731 / 6,059 | 2016+ |
| `ftn_charting` | 185,215 | 2022+ |

Play-by-play is deliberately not pulled. Thirteen seasons is multiple GB, and the
pre-aggregated weekly/NGS/ff_opportunity tables cover Phase 3's needs. Only pull
pbp later, column-restricted, for red zone / end zone share features.

### Known data issues

- **`depth_charts` stops at 2024.** nflverse returns zero rows for 2025; the
  loader does not error. Depth-chart features need another source, or must be
  dropped for the most recent season. Affects Phase 6 inference most.
- **`ff_playerids` is fetched via a fallback URL.** nflreadpy hardcodes the
  `github.com/<org>/<repo>/raw/...` form, which this environment's egress proxy
  rejects with 403. `configs/data.yaml` carries the canonical
  `raw.githubusercontent.com` URL for the identical file.
- **Duplicate `gsis_id`s in the ID map.** The upstream dynastyprocess map assigns
  one `gsis_id` to two players in ~10 same-name cases. Nine sit outside QB/RB/WR/TE.
  The one that does not is corrected in `configs/id_overrides.csv`.
- **Weekly stats include postseason.** `player_stats` ships REG and POST rows
  together; Puka Nacua's 2023 reads 18 games unfiltered versus 17 REG. Finish
  ranks must filter `season_type == "REG"` or labels silently credit playoff
  production. Locked by a test.

## Ground rules

**Leakage rule (non-negotiable):** every feature predicting season N must be
knowable before Week 1 of season N — season N-1 and earlier stats, plus offseason
facts (draft, trades, coaching changes, depth charts). To be enforced by test in
Phase 3.

**Scoring is computed, not sourced.** Full-PPR scoring will be defined in
`configs/scoring.yaml` and fantasy points derived from weekly stats.

## Progress

- [x] **Phase 0** — environment, structure, dependency pins
- [x] **Phase 1a** — nflverse pulls cached, row counts verified per season
- [ ] **Phase 1d** — ID crosswalk (next, with 1b)
- [ ] **Phase 1b** — historical ADP acquisition — *gates labeling*
- [ ] **Phase 1c** — Vegas / The Odds API (paid; out of v1 training features)
- [ ] **Phase 2** — labels
- [ ] **Phase 3** — features (WR first)
- [ ] **Phase 4** — modelling and validation
- [ ] **Phase 5** — SHAP explainability
- [ ] **Phase 6** — 2026 inference and market overlay
- [ ] **Phase 7** — tests, guardrails, docs

## Limitations

Roughly 4–5k usable training rows are expected, so modest lift over ADP baselines
is the realistic target — top-K precision gains of 10–20% over market would be a
real edge. FTN features exist only for 2022+ and models must run with them null.
The 2020 season is distorted by COVID opt-outs.
