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

### Market expectation (ADP substitute)

This environment's egress proxy blocks FantasyPros, Sleeper, and every other
non-GitHub ADP source, so preseason market expectation comes from the
**dynastyprocess historical FantasyPros ECR archive** (the brief's priority-2
source): the final preseason PPR consensus snapshot per season, normalized to
positional rank. Coverage and quality:

- **2020–2025: real market data** (`adp_source=ecr`), snapshot dated 4–7 days
  before each season's Week 1. 2,842 player-seasons, 100% of each season's
  top-200 matched to `gsis_id`.
- **2014–2019: no reachable source.** Phase 2 will label these with the
  prior-season-finish proxy (`adp_source=proxy`), per the brief's fallback,
  and they are flagged for sensitivity analysis.
- **Manual escape hatch:** drop `data/external/adp/<season>.csv` (schema in
  `data/external/adp/README.md`) to supply real ADP for any season;
  it takes precedence as `adp_source=manual`.

ECR is consensus *rank*, not literal draft position, but the pipeline uses
positional rank everywhere precisely so this distinction stays immaterial.

The ID crosswalk resolves external names/ids to nflverse `gsis_id` via a
three-rung ladder (FantasyPros-id join → exact normalized name+position →
fuzzy ≥93 with margin). One subtlety worth knowing: `ff_playerids` stamps
retired players' position as `XX` (current status, not playing position), so
position is a match *constraint*, never an eligibility filter — filtering
would silently drop every since-retired player from history. Every fuzzy or
failed match is logged to `data/id_map_review.csv`, never silently accepted.

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
- [x] **Phase 1d** — ID crosswalk: 100% top-200 match rate, all seasons
- [x] **Phase 1b** — market expectation: FantasyPros ECR 2020–2025 (real),
      2014–2019 to be proxy-labeled in Phase 2 — see below
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
