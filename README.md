# BreakoutLab

Preseason machine-learning pipeline that predicts NFL fantasy breakouts. For every
fantasy-relevant skill player (QB/RB/WR/TE) it is intended to output a calibrated
breakout probability, an expected finish-vs-ADP delta, and the SHAP drivers behind
each call.

> **Status: v1 complete.** nflverse ingest, market expectation, breakout
> labels, all four positions' features/models (WR/RB/TE/QB), SHAP
> explainability, and the 2026 breakout board (veteran scoring + a
> separate rookie heuristic + market overlay) are built. See
> [Progress](#progress) for verdicts and [Known Limitations](#known-limitations)
> for what to take with a grain of salt.

## Setup

```bash
uv sync --extra dev
uv run python -m src.ingest.nflverse   # pull + cache nflverse data (~40s cold)
uv run python -m src.ingest.adp        # build market_expectation.parquet
uv run python -m src.labels.build      # build labels.parquet
uv run python -m src.features.wr       # + .rb / .te / .qb: build features_{pos}.parquet
uv run python -m src.models.train_wr   # + train_rb / train_te / train_qb: train + save bundles
uv run python -m src.models.rookie_heuristic  # train the 2026-rookie heuristic
uv run python -m src.explain.shap_report      # outputs/shap_{pos}.png + global top-5 tables
uv run python -m src.inference.board_2026     # outputs/breakout_board_2026.{csv,md}
uv run pytest -q
```

Python 3.11+. The raw cache lands in `data/` and is gitignored; re-running the
ingest is idempotent and reuses parquet unless `--force` is passed.

Everything above is also wired up as `make` targets — `make test` (pytest),
`make board` (score-only, no data pull or retraining), `make refresh`
(re-pull the 2026-relevant data + re-score the board, no retraining), and
`make retrain` (the full pipeline: refresh + all five training runs — the
four position models plus the rookie heuristic — + SHAP + board). See the
`Makefile`; each target echoes what it's doing as it runs.

Explain any one player's call: `uv run python -m src.explain.shap_report
--why "Player Name" [--season 2026]` prints their calibrated probability,
expected finish-rank delta, and top-5 SHAP drivers with direction and raw
value.

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

Fourteen seasons (2013–2026) pulled via **`nflreadpy`** — not `nfl_data_py`, which
was deprecated and archived in September 2025. All loaders return polars.
2026 is the current offseason: rosters/schedules/draft_picks/combine reflect
real preseason data (no games played yet), while anything games-derived
(weekly stats, snaps, NGS, FTN, ff_opportunity) has nothing for 2026 until
Week 1 — see `configs/data.yaml`'s per-dataset `last_available` cap and
"Known data issues" below.

| Dataset | Rows | Coverage |
|---------|------|----------|
| `player_stats` (weekly) | 234,738 | 2013–2025 |
| `snap_counts` | 324,611 | 2013–2025 |
| `depth_charts` | 993,055 | 2013–**2024** |
| `rosters` | 40,415 | 2013–2026 |
| `schedules` | 3,834 | 2013–2026 |
| `draft_picks` | 12,927 | full history (incl. 257 2026 picks) |
| `combine` | 8,968 | full history (incl. 2026 combine) |
| `players` | 25,033 | full history |
| `ff_playerids` | 12,472 | crosswalk |
| `ff_opportunity` | 74,731 | 2013–2025 |
| `ngs_passing / receiving / rushing` | 5,933 / 14,731 / 6,059 | 2016+ |
| `ftn_charting` | 185,215 | 2022+ |

Play-by-play is deliberately not pulled. Thirteen seasons is multiple GB, and the
pre-aggregated weekly/NGS/ff_opportunity tables cover Phase 3's needs. Only pull
pbp later, column-restricted, for red zone / end zone share features.

### Known data issues

- **`depth_charts` stops at 2024.** nflverse returns zero rows for 2025, and a
  *different, schema-incompatible* payload for 2026 (12 columns, no `season`
  column at all — verified directly, not assumed) rather than erroring or
  returning zero rows the way 2025 does. `configs/data.yaml` caps this
  dataset's `last_available` at 2024 rather than requesting a season that
  silently changes shape. Depth-chart features need another source, or must
  be dropped for the most recent seasons.
- **`player_stats` / `snap_counts` / `rosters_weekly` / `ff_opportunity` /
  `ngs_*` / `ftn_charting` all reject or have nothing for season 2026.**
  `player_stats(seasons=[2026])` 404s on a release file that doesn't exist
  yet; `snap_counts`/`rosters_weekly` raise `ValueError` for any season past
  2025 (nflreadpy hard-validates the upper bound); the rest simply have no
  rows for a season with no games played. Each carries a `last_available: 2025`
  cap in `configs/data.yaml` so a fresh pull never crashes on this — delete
  the cap for a given dataset once its 2026 data starts flowing.
- **`rosters.parquet`'s 2026 snapshot spells Arizona "AZ".** Every other
  season, and every other 2026 source (`schedules`, `draft_picks`,
  `player_stats`), uses "ARI". Added to `src.features.shared.TEAM_ALIASES`.
- **2026 `draft_picks` ships each rookie's own *temporary*, non-nflverse
  `gsis_id`** (0/257 rows start with the standard "00-" prefix, vs. 100% for
  every prior draft class) — upstream hasn't back-filled the real crosswalk
  yet. `src.models.rookie_heuristic.resolve_2026_rookie_ids` recovers the
  real roster `gsis_id` via a two-rung pfr_id / normalized-name match
  (100% coverage for the 2026 skill-position class as of ingest time).
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

- **2020–2026: real market data** (`adp_source=ecr`), snapshot dated 4–7 days
  before each season's Week 1 (2026's is 2026-08-14, the archive's latest
  as of this build — pre-kickoff, since 2026's Week 1 hasn't happened).
  3,282 player-seasons total, 100% of each season's top-200 matched to
  `gsis_id`.
- **2014–2019: no reachable source.** These are labeled with the
  prior-season-finish proxy (`adp_source=proxy`), per the brief's fallback,
  and they are flagged for sensitivity analysis.
- **Manual escape hatch:** drop `data/external/adp/<season>.csv` (schema in
  `data/external/adp/README.md`) to supply real ADP for any season;
  it takes precedence as `adp_source=manual`.

ECR is consensus *rank*, not literal draft position, but the pipeline uses
positional rank everywhere precisely so this distinction stays immaterial.

### Breakout labels (Phase 2)

`data/processed/labels.parquet` (`src/labels/build.py`) is one row per
(season, gsis_id), 2014–2025, for every QB/RB/WR/TE with a REG-season
appearance. Fantasy points are **computed** from weekly stats via
`configs/scoring.yaml`'s `standard_ppr` profile — never the shipped
`fantasy_points_ppr` column, which additionally credits punt/kickoff-return
TDs that standard redraft PPR scoring doesn't count (mean |diff| ~0.012
pts/player-week; see the module docstring and `__main__`'s cross-check for
detail).

A player's *finish* is their within-position rank by PPR points-per-game
(min 8 REG games to qualify). Their *expectation* is the preseason market's
positional rank: `market_expectation.parquet` directly for 2020–2025
(`adp_source=ecr`), the player's own prior-season finish rank for
2014–2019 (`adp_source=proxy`, since no real market data reaches that far
back), or — for anyone either source has no rank for — one slot behind the
deepest rank that era's signal reaches for that season+position
(`adp_source=capped`). Capped players are never dropped: an unranked player
who breaks out is exactly the case this model exists to catch.
`breakout = 1` iff the finish clears the position's `finish_top` threshold
*and* the expectation was `adp_worse_than` threshold or deeper
(`configs/labels.yaml`). `in_training_pool` additionally drops rookie
seasons and thin/low-signal player-seasons (low prior-season ppg + deep
preseason rank) from the *modeling* pool without removing them from the
table.

The ID crosswalk resolves external names/ids to nflverse `gsis_id` via a
three-rung ladder (FantasyPros-id join → exact normalized name+position →
fuzzy ≥93 with margin). One subtlety worth knowing: `ff_playerids` stamps
retired players' position as `XX` (current status, not playing position), so
position is a match *constraint*, never an eligibility filter — filtering
would silently drop every since-retired player from history. Every fuzzy or
failed match is logged to `data/id_map_review.csv`, never silently accepted.

### Explainability (Phase 5)

`src/explain/shap_report.py` explains the four position bundles with
`shap.TreeExplainer`, restricted to whichever tree model(s) — LGBM and/or
XGB — carry a nonzero weight in that position's frozen blend (every
position's logistic head carries weight 0 as of this build, so it's
skipped; see the module docstring for the LinearExplainer branch that
would be needed if a future retrain ever changes that). Two outputs:

- **Global**: `outputs/shap_{pos}.png`, one beeswarm panel per
  nonzero-weight tree model, computed on the pooled validation rows
  (2020–2023, the exact universe Phase 4 tunes/calibrates against).
- **Per-player**: `uv run python -m src.explain.shap_report --why "Player
  Name" [--season 2026]` — blend-weighted top-5 SHAP drivers (feature,
  signed value, raw value), the calibrated probability, and the expected
  finish-rank delta for that one row.

SHAP values here are in each tree model's own **margin space**, not the
calibrated probability the board reports — see the module docstring's
"Margin space, not calibrated-probability space" note for why (the
isotonic/Platt calibrator on top is a monotone but nonlinear transform
SHAP doesn't pass through). Read every driver as a *direction and
relative-magnitude ranking*, not an exact probability contribution.

### 2026 breakout board (Phase 6)

`src/inference/board_2026.py` builds `outputs/breakout_board_2026.{csv,md}`
from three independently-computed pieces:

1. **Veteran scoring** — the *same* Phase-3 feature builders
   (`src.features.{wr,rb,te,qb}`), pointed at a synthetic 2026 population
   (2026 roster, `years_exp >= 1`) instead of `labels.parquet`, scored by
   the saved holdout-retrained bundles (classifier trio blended +
   calibrated; `expected_rank_delta` is the unweighted mean of the two
   regression models — no blend-weight search ever ran for that head, see
   the module docstring). 688 veterans scored across the four positions on
   this build.
2. **Rookie heuristic** (`src.models.rookie_heuristic`) — a separate,
   deliberately simple shallow-logistic model for the 2026 draft class
   (draft capital, combine athleticism where joinable, landing-team
   vacated shares, positional draft competition), trained on 2014–2025
   historical rookies and time-split-checked on 2024–2025. 80
   skill-position rookies scored on this build, in their own board
   section, **never on the same probability scale as the veteran model.**
3. **Overlay** (presentation only, never fed back into either model):
   consensus ADP is the 2026 ECR positional rank; `data/external/sleeper_adp_2026.csv`
   and `data/external/vegas_implied_2026.csv` are optional manual CSVs
   (schema in `data/external/README.md`) joined in when present, null
   otherwise; `configs/availability_2026.csv` is a manual, user-maintained
   injury/availability screen (ships as a template only — see Known
   Limitations) that *flags*, never drops, a player.

Run the whole thing with `make board` (score-only) or `make refresh`
(re-pull the 2026-relevant data first). See `outputs/breakout_board_2026.md`
for the actual current board.

## Ground rules

**Leakage rule (non-negotiable):** every feature predicting season N must be
knowable before Week 1 of season N — season N-1 and earlier stats, plus offseason
facts (draft, trades, coaching changes, depth charts). To be enforced by test in
Phase 3.

**Scoring is computed, not sourced.** Full-PPR scoring is defined in
`configs/scoring.yaml`; fantasy points are derived from weekly stats, never
read from the shipped `fantasy_points_ppr` column. See
[Breakout labels](#breakout-labels-phase-2).

## Progress

- [x] **Phase 0** — environment, structure, dependency pins
- [x] **Phase 1a** — nflverse pulls cached, row counts verified per season
- [x] **Phase 1d** — ID crosswalk: 100% top-200 match rate, all seasons
- [x] **Phase 1b** — market expectation: FantasyPros ECR 2020–2026 (real),
      2014–2019 proxy-labeled in Phase 2 — see above
- [ ] **Phase 1c** — Vegas / The Odds API (paid; out of v1 training features
      — Vegas is overlay-only, see Known Limitations)
- [x] **Phase 2** — labels: `configs/scoring.yaml`, `configs/labels.yaml`,
      `data/processed/labels.parquet` (2014–2025, QB/RB/WR/TE, 6,978 rows)
- [x] **Phase 3/4 (all four positions)** — feature builders (leakage-tested
      two ways per position: season-N stats stripped ⇒ identical output;
      team context keyed to Week-1 rosters, not final-team rosters) +
      LGBM/XGB/L2-logistic blend, isotonic-calibrated, expanding-window CV
      (val 2020–2023), one frozen holdout eval (2024–2025) each. Honest
      verdicts (holdout, pooled 2024+2025, model vs. baseline 2
      prior-season-ppg-rank — copied verbatim from each position's
      `outputs/model_{pos}_report.md`; regenerate with `make retrain`):
      - **WR** — DOES NOT beat baseline on top-10 precision (0.000 vs
        0.000) but BEATS it on PR-AUC (0.101 vs 0.038, +0.063).
      - **RB** — DOES NOT beat baseline on top-10 precision (0.000 vs
        0.000) but BEATS it on PR-AUC (0.112 vs 0.028, +0.083).
      - **TE** — BEATS baseline on both top-10 precision (0.100 vs 0.000,
        +0.100) and PR-AUC (0.107 vs 0.069, +0.039).
      - **QB** — BEATS baseline on both top-10 precision (0.200 vs 0.000,
        +0.200) and PR-AUC (0.167 vs 0.054, +0.113).

      Read these with the brief's own honesty rule in mind: holdout
      positives are single digits per position (RB/TE/QB especially — see
      `src/models/train.py`'s "Small-sample caveat" docstring), so a
      handful of hits/misses moves these numbers a lot. Modest, real,
      uneven lift — not proof of a large edge, and WR/RB's top-10 metric
      in particular is thin evidence either way at this sample size.
      **Caveat on these exact numbers:** the `outputs/model_{pos}_report.md`
      files these are copied from predate this session's changes and were
      not regenerated here (a full real-Optuna retrain across four
      positions is expensive — this build's `data/models/*_model_bundle.joblib`
      were instead used as-is, with only the missing regression-head
      objects backfilled at their already-frozen hyperparameters, never
      re-tuned). The directional read (two positions beat the baseline
      cleanly, two don't on top-10 precision) is almost certainly stable;
      the exact decimals may drift a little after `make retrain`.
- [x] **Phase 5** — SHAP explainability: `outputs/shap_{pos}.png` (global
      beeswarm, nonzero-weight tree models) + `--why "Player Name"`
      per-player cards (`src/explain/shap_report.py`)
- [x] **Phase 6** — 2026 inference: veteran scoring (688 players across
      four positions) + rookie heuristic (80 skill-position rookies) +
      market/Sleeper/Vegas/availability overlay, written to
      `outputs/breakout_board_2026.{csv,md}` (`src/inference/board_2026.py`,
      `src/models/rookie_heuristic.py`)
- [x] **Phase 7** — guardrails + docs: `Makefile` (`refresh` / `retrain` /
      `test` / `board`), `tests/test_inference.py` (golden-player
      regression, board schema, 2026 feature-matrix sanity), this README

## Known Limitations

- **Roughly 4–5k usable training rows** are expected across the whole
  pipeline, so modest lift over ADP baselines is the realistic target —
  top-K precision gains of 10–20% over market would be a real edge. See
  the per-position verdicts above; two of four positions don't clear that
  bar on top-10 precision at this sample size.
- **ADP before 2020 is a proxy, not real market data.** 2014–2019 seasons
  use the player's own prior-season finish rank as `expectation_pos_rank`
  (`adp_source=proxy`) because no reachable ADP archive covers that far
  back in this environment — see "Market expectation" above.
- **No Vegas data anywhere in training.** The brief scopes Vegas/odds data
  to the overlay only (`data/external/vegas_implied_2026.csv`, optional,
  manual, presentation-layer); Phase 1c (a live Vegas/Odds-API feed) was
  never built — it's a paid API this environment can't reach.
- **The rookie model is a heuristic, not a Phase-4-grade model.** A single
  shallow L2 logistic on draft capital + combine + landing-team context,
  time-split sanity-checked (not tuned) on ~450–550 historical rookie
  rows — see `src/models/rookie_heuristic.py`'s module docstring. Its
  probabilities are never on the same scale as the veteran model's and are
  kept in a visibly separate board section for exactly that reason.
- **FTN charting (`ftn_charting`) only exists for 2022+** and no feature
  builder currently reads it at all (it's cached, not yet wired into any
  position's feature set) — every model must and does run fine without it.
- **The 2020 season is distorted by COVID opt-outs** (16-game season,
  empty stadiums). `label_season_2020` flags both 2020 itself and 2021
  (whose N-1 features are built entirely from 2020's disrupted data) but
  does not otherwise correct for it.
- **`configs/coaching_changes.csv` (`new_oc`) is unverified and has no
  2026 rows.** Every row's `verified` column is 0 — seeded from model
  knowledge (cutoff January 2026) because this environment's proxy blocks
  the usual coaching-change trackers; expect a handful of misses in
  2019–2025. 2026 is entirely absent (not guessed at), so `new_oc` reads
  null for every 2026 row this build scores — verifying the file and
  seeding 2026 is a manual, ungated follow-up task, not automated here.
- **`configs/availability_2026.csv` is a manual, user-maintained file,**
  shipped as a template only (header + commented examples, no real rows —
  see the file's own header for why). Fill in real injury/availability
  rows yourself before trusting the board's `availability` column; a
  player absent from the file reads as "not flagged," never as "confirmed
  healthy." A ~5 minute task against any current injury report.
- **Sleeper ADP and Vegas implied points are manual-CSV hooks, not live
  feeds.** This environment's egress proxy blocks Sleeper's API directly
  (same restriction as the historical ADP source); `data/external/sleeper_adp_2026.csv`
  and `data/external/vegas_implied_2026.csv` (schema in
  `data/external/README.md`) are the intended escape hatch and are empty
  by default — the board's `sleeper_adp_pos_rank`, `adp_gap`, and
  `implied_pts` columns are null until one is supplied.
- **Isotonic calibration saturates at the extremes.** At this pool size, a
  position's highest/lowest raw-score OOF bucket can land entirely on one
  class, which isotonic maps to an exact 0.0 or 1.0 — verified on this
  build: five 2026 QBs and three 2026 TEs hit an exact 1.000. That's a
  small-bucket artifact, not model certainty, so `outputs/breakout_board_2026.csv`'s
  `probability` column is clamped to `[0.01, 0.95]` for display
  (`probability_saturated` flags which rows hit the boundary pre-clamp;
  `raw_score`, the pre-calibration blend score, breaks the resulting ties
  — see `src/inference/board_2026.py`'s module docstring). A principled
  fix (e.g. Laplace-smoothed isotonic buckets, or a confidence interval
  instead of a point estimate) needs persisted per-fold OOF predictions
  and is noted here as a v1.5 follow-up, not built in this pass.
