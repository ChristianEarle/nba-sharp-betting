# BreakoutLab

Preseason machine-learning pipeline that predicts NFL fantasy breakouts. For every
fantasy-relevant skill player (QB/RB/WR/TE) it is intended to output a calibrated
breakout probability, an expected finish-vs-ADP delta, and the SHAP drivers behind
each call.

> **Status: v2.0 complete.** v1 (nflverse ingest, market expectation,
> breakout labels, all four positions' features/models, SHAP
> explainability, the 2026 breakout board) plus v1.5 (Odds API ingest,
> Laplace-smoothed calibration, pbp-derived red-zone/goal-line share
> features, the Vegas team/prop promotion gate) plus v1.7 (a per-position
> proxy-era training-pool sensitivity check, a 3-seed Optuna ensemble,
> Platt calibration, per-position base-rate renormalization, two more
> "fair" baselines) plus **v2.0: a second model — quantile regression over
> each player's full per-game scoring distribution
> (`src/models/quantile.py`) — predicting floor/expected/ceiling points,
> predicted finish rank, and an honest outcome ladder (elite / startable /
> useful / bust odds) for every scored veteran, eligible or not. The v1.7
> classifier is not replaced: a pre-stated, binding comparison gate
> (Deliverable 3) decides per position which model runs the board's
> "Breakout Hunt" lens (TE moved to quantile; WR/RB/QB kept the
> classifier), and a new "Full Projections" lens ranks every veteran by
> expected points regardless of breakout eligibility.** See
> [Progress](#progress) for v1.7 verdicts, "v2.0: structural redesign"
> below for the full quantile/gate/lens writeup, and
> [Known Limitations](#known-limitations) for what to take with a grain of
> salt. (v1.6 was an internal-only checkpoint folded into v1.7 — no
> separate v1.6 section exists.)

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
| `pbp` (v1.5, 9 columns only) | 628,163 | 2013–2025 |

v1 deliberately did not pull play-by-play at all: thirteen seasons is multiple
GB unrestricted, and the pre-aggregated weekly/NGS/ff_opportunity tables
covered Phase 3's needs. v1.5 pulls it column-restricted for the red-zone/
end-zone/goal-line share features (`configs/data.yaml`'s `pbp` entry, 9
columns: season, week, season_type, posteam, yardline_100, pass_attempt,
rush_attempt, receiver_player_id, rusher_player_id, complete_pass,
touchdown) — `nflreadpy.load_pbp` has no column-selection kwarg of its own
(checked directly), so `src.ingest.nflverse.pull_dataset` selects
immediately after the loader call, before anything touches disk; only the
slim frame is ever cached. Result on this build: **2.7MB** on disk (well
under any reasonable budget), 628,163 rows across 2013–2025 (0 for 2026 —
no games played yet, same `last_available` convention as every other
games-derived dataset).

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

## v1.5: Laplace-smoothed calibration

Fixes v1's isotonic terminal-bucket saturation (see the "Isotonic
calibration" note in Known Limitations for what that was). `src.models.
train.SmoothedIsotonic` replaces each isotonic PAV block's raw empirical
rate (pos/n — which can be an exact 0.0 or 1.0 for a small terminal
bucket) with its Laplace/rule-of-succession-smoothed value ((pos+1)/(n+2)),
then re-monotonizes the block sequence with a second PAVA pass (weighted
by block size) so ordering is never broken by the smoothing step — see the
class's docstring in `src/models/train.py` for the full two-pass
construction and why it can never reintroduce an exact 0 or 1.

Every position's bundle now carries the pooled OOF validation predictions
(`pooled_oof_val`) plus the smoothed calibrator
(`smoothed_calibration_method`/`smoothed_calibrator`) **alongside**, not
instead of, the original `calibration_method`/`calibrator` — nothing was
deleted. `src/inference/board_2026.py` scores off the smoothed calibrator;
the `[0.01, 0.95]` display clamp and `probability_saturated` flag from v1
stay on as a no-op safety net. Verified on this build: **478 → 0** veteran
rows with an exact pre-clamp 0.0/1.0 calibrated probability, board-wide
across all four positions (`tests/test_inference.py::test_board_has_zero_saturated_rows_v1_5`;
per-position strict-(0,1)-and-monotonic checks in
`tests/test_models_positions.py`).

## v1.5: pbp red-zone features + retrain

v1 skipped play-by-play entirely (see the Data section's `pbp` note).
v1.5 pulls it column-restricted (9 columns, 2.7MB on disk — see Data
above) and adds four nullable, N-1-shifted features
(`src.features.shared.redzone_share_table`):

- **WR/TE**: `rz_target_share` (share of the player's team's own targets
  inside yardline_100<=20) and `ez_target_share` (the same, <=10,
  "end-zone-adjacent") — TE inherits these unchanged from WR's
  `BASE_METRICS`/`build_raw_stat_table` (same receiving family it already
  shared everything else with).
- **RB**: `rz_carry_share` (<=20) and `goal_line_carry_share` (<=5), same
  "share of the team's own usage inside this threshold" definition.
- **QB**: none, per the brief.

Leakage discipline is identical to every other feature: N-1-shifted
(season-N rows never read season-N pbp), and structurally tested the same
way as `player_stats` — `tests/test_features_pbp.py` strips season-2023
rows out of `pbp` and asserts season-2023 feature rows are byte-identical
to the full build. Nullable by design: `pbp=None` (never pulled) yields
null columns, never an error — exercised directly in the same test file,
and by `src/inference/board_2026.py`'s 2026 population (populated from
real season-2025 pbp when `pbp.parquet` exists, same N-1-shift convention
every other feature uses; null only if pbp were absent entirely).

All four positions were retrained from scratch (real Optuna config: 60
classifier trials + 30 regression trials per model type, same as v1 —
this is a new model generation, not a re-tune; the v1 holdout verdict
stands untouched in git history). Holdout (2024+2025 pooled) PR-AUC and
top-10 precision, v1 vs v1.5, reported plainly — improvements and
regressions alike, no iteration after seeing these numbers:

| Position | PR-AUC (v1) | PR-AUC (v1.5) | Δ | Top-10 precision (v1) | Top-10 precision (v1.5) | Δ |
|---|---|---|---|---|---|---|
| WR | 0.101 | 0.104 | +0.003 | 0.000 | 0.000 | +0.000 |
| RB | 0.112 | 0.119 | +0.007 | 0.000 | 0.200 | **+0.200** |
| TE | 0.107 | 0.093 | **−0.014** | 0.100 | 0.200 | +0.100 |
| QB | 0.167 | 0.167 | +0.000 | 0.200 | 0.200 | +0.000 (no pbp features for QB) |

RB is the clear win (rz/goal-line carry share moved it from "doesn't beat
baseline on top-10" to beating it on both metrics). TE is a genuine mixed
result — better top-10 precision, *worse* PR-AUC — reported as-is rather
than cherry-picked; at TE's holdout sample size (see the small-sample
caveat below) a couple of ranking flips easily move both metrics this
much. WR moved only marginally. QB is unchanged (no new features), which
also serves as an implicit determinism/no-regression check on everything
*except* the new features.

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
- [x] **Phase 1c (v1.5)** — The Odds API ingest is code-complete
      (`src/ingest/odds_api.py`, `configs/odds_api.yaml`, see "v1.5 Odds
      API (local run required)" above) but never executed against the
      network from this environment (the egress proxy blocks
      `api.the-odds-api.com`) — still overlay-only, not a training
      feature, until a local run produces real `data/external/odds_api/`
      data, see Known Limitations
- [x] **Phase 2** — labels: `configs/scoring.yaml`, `configs/labels.yaml`,
      `data/processed/labels.parquet` (2014–2025, QB/RB/WR/TE, 6,978 rows)
- [x] **Phase 3/4 (all four positions)** — feature builders (leakage-tested
      two ways per position: season-N stats stripped ⇒ identical output;
      team context keyed to Week-1 rosters, not final-team rosters) +
      LGBM/XGB/L2-logistic blend, Platt-calibrated (v1.7 — was isotonic in
      v1, SmoothedIsotonic in v1.5; see "v1.7: fixes" above), expanding-window
      CV, one frozen holdout eval (2024–2025) each. **v1.7 retrained all
      four from scratch** under the new 3-seed Optuna ensemble (RB also
      moved to a real-market-only 2020+ training pool — see "v1.7: fixes"
      → "Step 1"/"Step 2" above for the full comparison tables). Current
      (v1.7) verdicts, copied verbatim from each position's
      `outputs/model_{pos}_report.md` (regenerate with `make retrain`):
      - **WR** — BEATS baseline 2 (prior-season ppg rank) on both top-10
        precision (0.200 vs 0.000, +0.200) and PR-AUC (0.132 vs 0.038,
        +0.095) — the top-10 flip from v1.5 (was 0.000 vs 0.000) is this
        generation's single biggest qualitative change.
      - **RB** — BEATS baseline on both top-10 precision (0.200 vs 0.000,
        +0.200) and PR-AUC (0.177 vs 0.028, +0.149).
      - **TE** — BEATS baseline on both top-10 precision (0.200 vs 0.000,
        +0.200) and PR-AUC (0.383 vs 0.069, +0.315).
      - **QB** — BEATS baseline on both top-10 precision (0.300 vs 0.000,
        +0.300) and PR-AUC (0.395 vs 0.054, +0.341).

      All four also beat or tie both v1.7 "fair" baselines
      (`eligible_prior_ppg`, `napkin_logistic`) on holdout top-10 precision
      — the one PR-AUC exception (RB's `napkin_logistic` edges the model,
      0.202 vs 0.177) is reported plainly in "v1.7: fixes" → "Step 4"
      above, not hidden. Read all of this with the brief's own honesty
      rule in mind: holdout positives are single digits per position
      (RB/TE/QB especially — see `src/models/train.py`'s "Small-sample
      caveat" docstring), so a handful of hits/misses moves these numbers
      a lot — TE/QB's large PR-AUC jumps this generation are plausible at
      this sample size, not proof of a proportionally large capability
      jump. Real, broad-based lift across every position on this holdout
      pass — still not proof of a large edge at this sample size.
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
- [x] **v1.5 A — Odds API ingest**: code-complete
      (`src/ingest/odds_api.py`, `configs/odds_api.yaml`), zero-network
      tested (`tests/test_odds_api.py`); never executed against the
      network from this environment (proxy blocks the API) — see "v1.5
      Odds API (local run required)" below
- [x] **v1.5 B — Laplace-smoothed calibration**: `src.models.train.SmoothedIsotonic`,
      backfilled onto the (pre-retrain) v1 bundles then carried forward
      natively by the v1.5 retrain; board saturated-row count 478 → 0 —
      see "v1.5: Laplace-smoothed calibration" below
- [x] **v1.5 C — pbp red-zone features + retrain**: `configs/data.yaml`'s
      `pbp` entry (9-column slim pull), `src.features.shared.redzone_share_table`,
      full from-scratch Optuna retrain of all four positions — see "v1.5:
      pbp red-zone features + retrain" below
- [x] **v1.5 D (Phase C) — Vegas team/prop features + promotion gate**:
      `src.ingest.vegas` (real local pull, 2020–2025), `implied_ppg`/
      `implied_win_prob`/`has_vegas` wired into every position's feature
      matrix (nullable, `src.features.shared.attach_vegas_team`), an
      overlay backtest (`src.models.overlay_backtest`), and the binding
      promotion experiment (`src.models.vegas_experiment`) — **all four
      positions rejected**, so the Vegas columns stay in the matrix but
      excluded from every position's trees — see "v1.5 Phase C: Vegas
      team/prop features" below
- [x] **v1.7 Step 1 — proxy-era training-pool sensitivity**:
      `src/models/proxy_sensitivity.py` (`outputs/proxy_sensitivity.md`),
      binding decision rule applied mechanically per position — **RB
      moved to `train_season_start: 2020`**, WR/TE/QB keep the full
      2014-2025 pool — see "v1.7: fixes" → "Step 1" above
- [x] **v1.7 Step 2 — seed-ensemble retrain**: `configs/model_{pos}.yaml`'s
      `optuna.seeds: [42, 1337, 2024]`, `src.models.train`'s ensemble mode
      (per-seed LGBM+XGB tune/OOF, one shared logistic head, raw score =
      mean of per-seed blended scores), full retrain of all four positions
      — every position improved or held on holdout PR-AUC and top-10
      precision vs. the prior generation, no regressions — see "v1.7:
      fixes" → "Step 2" above
- [x] **v1.7 Step 3 — honest probabilities**: Platt replaces
      SmoothedIsotonic as the ACTIVE calibrator (strictly monotone —
      displayed ranking == raw-score ranking by construction) plus a new
      per-position base-rate renormalization layer at board time
      (`board_2026.attach_renormalized_probability`) so each position's
      displayed probabilities sum to its historical mean per-season
      breakout count; tiers are now base-rate multiples, not fixed
      cutoffs — see "v1.7: fixes" → "Step 3" above
- [x] **v1.7 Step 4 — fair baselines**: `eligible_prior_ppg` and
      `napkin_logistic` (`src/models/baselines.py`), reported alongside
      the original three in every report/metrics JSON, no promotion logic
      — all four positions beat or tie both on holdout top-10 precision;
      the one PR-AUC exception (RB vs. `napkin_logistic`) is reported
      plainly — see "v1.7: fixes" → "Step 4" above
- [x] **v1.7 Step 5 — regenerated every consumer**: board, dashboard, SHAP
      pngs, every model report/metrics JSON, this README — see "v1.7:
      fixes" → "Step 5" above

## v1.5 Odds API (local run required)

`src/ingest/odds_api.py` + `configs/odds_api.yaml` is Phase 1c's revisit:
a code-complete ingest for [the-odds-api.com](https://the-odds-api.com)'s
historical NFL odds, covering preseason team-level lines (h2h/spreads/
totals, 2020–2025) and a probe for season-long player-futures markets
(2023+). **This environment's egress proxy blocks
`api.the-odds-api.com` outright** — verified directly, every request to
that host fails at the proxy layer before reaching the API — so nothing
in this module has ever been executed against the network from here.
`tests/test_odds_api.py` covers it completely (planner math, snapshot-date
derivation, manifest schema, `--normalize` against a checked-in fixture)
with zero network calls. **Run the network subcommands locally**, outside
this proxy, where `api.the-odds-api.com` is reachable.

### Command sequence

```bash
export ODDS_API_KEY=<your key>                     # required for every subcommand below except --normalize

uv run python -m src.ingest.odds_api --dry-run       # zero network calls: prints every planned request,
                                                       # key REDACTED, cost per request, projected total

uv run python -m src.ingest.odds_api --verify-futures # ONE historical call, earliest props-era (2023)
                                                       # snapshot, probes configs/odds_api.yaml's
                                                       # candidate_futures_markets; ~10 credits

uv run python -m src.ingest.odds_api --pull-team       # featured markets (h2h/spreads/totals), all six
                                                       # preseason snapshots (2020-2025); ~180 credits

uv run python -m src.ingest.odds_api --pull-props      # props-era snapshots (2023-2025): cheap (one call
                                                       # per season) if --verify-futures found a season-long
                                                       # futures key; otherwise refuses unless --allow-expensive
                                                       # is also passed (falls back to Week-1 event-level
                                                       # player-prop odds, an order-of-magnitude ~1,920 credits)
uv run python -m src.ingest.odds_api --pull-props --allow-expensive  # only needed for the expensive fallback

uv run python -m src.ingest.odds_api --normalize      # pure local, no network, no key: raw/*.json ->
                                                       # data/external/odds_api/team_lines.parquet
```

Expected costs (against `configs/odds_api.yaml`'s `credit_budget: 3000`):
**`--verify-futures` ≈10 credits, `--pull-team` ≈180 credits, `--pull-props`
≈270 credits if a season-long futures key exists (3 props-era seasons x
the verified market(s)) up to an estimated ~1,920 credits for the
expensive Week-1 event-props fallback** — every network subcommand reads
`x-requests-remaining`/`x-requests-used` off every response, prints
running totals, and hard-aborts *before* any call that would push the
projected total past `credit_budget` or past the account's own remaining
quota. Every raw response is written verbatim to
`data/external/odds_api/raw/<season>_<market>_<snapshot>.json` plus a
`manifest.json` (redacted URL, timestamp, cost, headers) —
`data/external/` is git-tracked by an existing `.gitignore` negation, and
`data/external/odds_api/**` inherits it unchanged (verified with `git
check-ignore`, locked by `tests/test_odds_api.py::test_raw_json_output_is_git_tracked`),
so the raw JSON is a real cross-machine contract: pull it locally once,
commit it, and this environment (or any other) can `--normalize` and use
it without ever touching the network itself.

### Local run results (actual)

The local pull has since happened; `data/external/odds_api/` now carries
real 2020–2025 data, git-tracked per the negation above. Verified
directly against `data/external/odds_api/manifest.json` (the module's own
authoritative cost ledger) rather than estimated: **2,514 credits total**
— 180 for `--pull-team` (6 snapshots), 4 for 4 events-lookups, and 2,330
for 46 Week-1 event-props calls (`--pull-props --allow-expensive`, the
expensive per-event fallback — the manifest carries no
`--verify-futures`-labeled entries at all, so whether a season-long
futures market was probed and rejected, or the run skipped straight to
`--pull-props --allow-expensive`, isn't determinable from the artifacts
on disk: `execute_request`'s tolerated-422 path returns before writing
either a raw file or a manifest entry, so an all-422 futures probe would
leave the identical zero-trace footprint as never running one at all —
worth tightening if that distinction ever matters). See "v1.5 Phase C:
Vegas team/prop features" below for what was built from this data.

## v1.5 Phase C: Vegas team/prop features

Two derived tables (`src.ingest.vegas`, `python -m src.ingest.vegas`),
built from the local pull above:

- **`data/processed/vegas_team.parquet`** — per (season, team):
  market-implied points-per-game (from `total`/`home_spread`) and a
  de-vigged moneyline win probability, averaged across every event
  `team_lines.parquet` priced for that team that season. Coverage (every
  team_lines-listed team counts, `has_vegas` is always 1 in this table —
  see its docstring for why):

  | Season | Teams priced | Avg. priced events / team |
  |---|---|---|
  | 2020 | 32 | 2.0 |
  | 2021 | 32 | 2.3 |
  | 2022 | 32 | 2.7 |
  | 2023 | 32 | 1.7 |
  | 2024 | 32 | 17.0 |
  | 2025 | 32 | 16.4 |

  All 32 teams are listed every season 2020–2025 (books priced at least
  one game per team even 4+ days out); 2020–2023 only got a handful of
  early-marquee games this far ahead of kickoff (books "didn't hang
  full-season lines 4 days out" those years), while 2024–2025 are close
  to the full 17-game slate.
- **`data/processed/vegas_props.parquet`** — per (season, gsis_id): a
  Week-1-only PPR-points rate proxy from the props raw JSON (median
  betting line per market, scored through `configs/scoring.yaml`'s
  `standard_ppr` weights), matched to `gsis_id` via
  `src.ingest.id_map.match_to_gsis` off a market-composition position
  guess (the props JSON carries no position column). Match rates:

  | Season | Players | Matched | Rate |
  |---|---|---|---|
  | 2023 | 15 | 13 | 86.7% |
  | 2024 | 131 | 130 | 99.2% |
  | 2025 | 173 | 170 | 98.3% |

  (2023 has only 1 of 15 Week-1 events with any bookmaker data at all —
  the Thursday opener — matching the local pull's own coverage note.)

**Overlay backtest** (`src.models.overlay_backtest`,
`outputs/overlay_backtest.md`): does `probability * log1p(expectation_pos_rank)`
(the board's own overlay formula) or a Vegas-weighted variant
(`* clip(1 + z(implied_ppg), 0.5, 1.5)`) beat the model alone at ranking
2023–2025 breakouts? **No consistent winner** — model-only wins outright
at WR and TE, the Vegas variant wins at RB, all three tie at QB on
top-10 precision — and every position's pooled holdout-era positive
count (5–10) is well under what's needed to call any of these deltas a
real edge rather than noise; see that report for the full per-season +
pooled tables and the explicit small-sample caveat on every verdict line.

**Promotion gate** (`src.models.vegas_experiment`,
`outputs/vegas_experiment.md`): per position, reuse every v1.5-frozen
hyperparameter/blend-weight (no new Optuna anywhere), add
`implied_ppg`/`implied_win_prob`/`has_vegas` to the tree heads only
(the logistic head's curated subset stays low-null by design — see that
module's docstring), and compare pooled 2024+2025 holdout top-10
precision against the v1.5-only baseline. **All four positions
rejected** (WR/RB/TE/QB: tie or regression, never an improvement) — per
the binding rule, none were promoted. Mechanically, rejection means
`configs/model_{pos}.yaml`'s `excluded_features: [implied_ppg,
implied_win_prob, has_vegas]` stays populated for every position, which
`src.models.train.tree_feature_columns` reads to keep those three
columns out of that position's LGBM/XGB feature set — the columns
remain in every `features_{pos}.parquet` as nullable, just unused by any
currently-deployed model. Had any position promoted instead,
`src.models.vegas_experiment.promote_position` regenerates that
position's bundle (frozen hyperparameters, real OOF-recomputed
calibration, one holdout re-eval — no new Optuna) and clears that
position's `excluded_features`; the 2026 board would then need
`vegas_team` wired into `board_2026.load_raw_frames`, which is a
mechanical follow-up (see `outputs/vegas_experiment.md`'s summary) never
exercised here since nothing promoted.

## v1.7: fixes (seed ensemble, honest probabilities, fair baselines, proxy-era pool)

A five-part pass over the modeling pipeline itself (not new features): all
four positions retrained from scratch under this generation.

### Step 1 — proxy-era training-pool sensitivity

`src/models/proxy_sensitivity.py` (`outputs/proxy_sensitivity.md`) asks,
per position, holding every hyperparameter and blend weight FROZEN at the
then-shipped bundle's values (no Optuna retuning): does dropping the
pre-2020 `adp_source=proxy` rows (prior-season-finish label, not real
market data — see "Market expectation" above) and training on the
real-market-only 2020+ pool do at least as well on holdout? Both arms
retrain on train≤2023 and are scored ONCE on the identical 2024+2025
holdout rows; Arm B's fold structure is 2 folds (val 2022, val 2023) via
`cv.validation_folds`'s `MIN_FOLD_TRAIN_YEARS` rule, not 4. Decision rule
(binding, applied mechanically): choose Arm B iff its holdout top-10
precision ≥ Arm A's AND its holdout PR-AUC ≥ Arm A's − 0.02.

| Position | Decision | Arm A top-10 / PR-AUC | Arm B top-10 / PR-AUC | 2026 board Spearman (A vs B) |
|---|---|---|---|---|
| WR | **keep Arm A** (full 2014-2025) | 0.200 / 0.133 | 0.000 / 0.083 | 0.626 |
| RB | **adopt Arm B** (2020+ only) | 0.100 / 0.142 | 0.200 / 0.188 | 0.472 |
| TE | **keep Arm A** (full 2014-2025) | 0.200 / 0.426 | 0.000 / 0.049 | 0.053 |
| QB | **keep Arm A** (full 2014-2025) | 0.300 / 0.388 | 0.100 / 0.151 | 0.794 |

RB is the one position where the real-market-only pool won outright on
both metrics — implemented as `configs/model_rb.yaml`'s `train_season_start:
2020`, consumed by `train.load_modeling_frame` (pool filter) and
`train.run_full_pipeline` (fold generation), never hardcoded per position.
WR/TE/QB keep the full pool. See `outputs/proxy_sensitivity.md` for the
per-season tables and each position's fold structure.

### Step 2 — seed-ensemble retrain

Single-seed Optuna tuning was replaced with a 3-seed ensemble
(`configs/model_{pos}.yaml`'s `optuna.seeds: [42, 1337, 2024]`): LGBM+XGB
are tuned and OOF'd independently per seed (own frozen hyperparameters,
own blend weights against the ONE shared logistic head, tuned once), and
the model's raw score is the mean of every seed's blended score — same
mechanism at holdout time (`src.models.train.holdout_predictions`) and at
2026 board-scoring time (`src.inference.board_2026.score_veterans_batch`).
Determinism (`tests/test_models_positions.py::test_determinism_seed_ensemble_two_runs_identical`,
`test_seed_ensemble_raw_score_is_mean_of_per_seed_blends`): a fixed seed
list reproduces byte-identical holdout predictions and per-seed structure
across independent runs. All four positions were retrained through this
new ensemble path; holdout (2024+2025 pooled), v1.6-era single-seed
generation vs this v1.7 seed-ensemble generation, reported plainly:

| Position | PR-AUC (prev.) | PR-AUC (v1.7) | Δ | Top-10 (prev.) | Top-10 (v1.7) | Δ |
|---|---|---|---|---|---|---|
| WR | 0.104 | 0.132 | +0.028 | 0.000 | 0.200 | **+0.200** |
| RB | 0.119 | 0.177 | +0.058 | 0.200 | 0.200 | +0.000 |
| TE | 0.093 | 0.383 | **+0.290** | 0.200 | 0.200 | +0.000 |
| QB | 0.167 | 0.395 | **+0.228** | 0.200 | 0.300 | +0.100 |

Every position improved or held on both metrics — no regressions.
("prev." = the last recorded verdict from this repo's prior generation,
before this pass's retrain overwrote the gitignored `outputs/model_*`
artifacts those numbers came from — WR's is read directly off the JSON
this session captured before retraining; RB/TE/QB's are the exact
figures this README's own Progress section carried, git-tracked, from
that generation. See the honesty note in Known Limitations about this.)
TE/QB's large PR-AUC jumps are consistent with — not necessarily entirely
caused by ensembling alone; the pool didn't change, so this is one seed's
worth of retrain variance plus ensembling — this pipeline's own documented
small-sample caveat (4-5 holdout positives): a single ranking flip moves
PR-AUC a lot at this sample size. Read the fold-level spread in each
`outputs/model_{pos}_report.md`, not just the pooled mean.

### Step 3 — honest probabilities

**Calibration:** the ACTIVE calibration method is now Platt scaling
(`src.models.train.fit_platt`, a sigmoid on the pooled OOF ensemble
score) instead of v1.5's SmoothedIsotonic. Platt is strictly monotone, so
the calibrated ranking is identical to the raw-score ranking by
construction (`tests/test_inference.py::test_board_displayed_ranking_matches_raw_score_ranking`)
— isotonic-family calibrators are only *non-decreasing*, and their flat
blocks could shift ranking (see the "Calibrated vs. raw-blend metrics"
limitation below, which this doesn't retire, since raw-vs-calibrated
metric equivalence still isn't guaranteed for every calibrator this repo
keeps around). The old isotonic/Platt-fallback choice
(`legacy_calibration_method` in every `outputs/model_{pos}_metrics.json`)
and v1.5's SmoothedIsotonic are both still fit and stored on every bundle,
never deleted — just no longer what the board scores off of.

**Renormalization (board/inference time, presentation-layer only,
`src/inference/board_2026.py`):** a calibrated probability is only ever
calibrated against its own OOF validation pool — nothing forces a
position's SUM of 2026 calibrated probabilities to equal how many
breakouts that position actually produces in a season, and pre-v1.7 it
didn't (verified: several positions summed to well over their historical
per-season breakout count). `attach_renormalized_probability` fixes this
at board time: `scale = historical_mean_breakouts / sum(calibrated p over
that position's 2026 veterans)` (computed from `labels.parquet`, 2014-2025,
never hardcoded), `displayed_probability = min(0.97, p * scale)`. Because
the scale is one positive constant per position, it cannot change
within-position ranking. The board CSV now carries `probability`
(displayed, renormalized), `probability_calibrated_raw` (pre-rescale
Platt output), and `raw_score` (pre-calibration) side by side. Verified on
this build:

| Position | Renorm scale | Sum(displayed probability) | Historical mean breakouts/season | Base rate (this year's pool) |
|---|---|---|---|---|
| WR | 1.065 | 5.08 | ~5.08 | 0.0183 (n=278 scored) |
| RB | 0.714 | 5.25 | ~5.25 | 0.0318 (n=165 scored) |
| TE | 0.601 | 2.67 | ~2.67 | 0.0180 (n=148 scored) |
| QB | 1.101 | 4.42 | ~4.42 | 0.0455 (n=97 scored) |

(Sum(displayed) matches the historical mean by construction post-rescale,
modulo the 0.97 cap on any individual player, which didn't bind hard
enough on this build to move any position's sum outside a tight band —
locked by `tests/test_inference.py::test_board_displayed_probability_sum_near_historical_mean_breakouts`,
±20% tolerance.)

**Tiers are now base-rate multiples**, not fixed absolute cutoffs: a
position's `base_rate` = historical mean breakouts ÷ players scored at
that position this year (table above), and a player's tier is their
displayed probability's multiple of it — **Elite target** (8x+), **Strong
swing** (4-8x), **Worth a flier** (2-4x), **Long shot** (under 2x) —
computed server-side (`board_2026.attach_renormalized_probability`) and
carried on every board row (`tier`, `base_rate_multiple`), not
recomputed against a fixed threshold client-side the way v1's dashboard
did. The dashboard's tier legend and "what the percentage means" banner
were updated to match (`src/dashboard/template.py`).

### Step 4 — fair baselines

Two more baselines (`src/models/baselines.py`), evaluated on validation
folds and the one holdout pass, reported alongside the original three in
every `outputs/model_{pos}_report.md` / `metrics.json`:

- **`eligible_prior_ppg`** — ranks only label-ELIGIBLE players
  (`expectation_pos_rank >= adp_worse_than`, the identical gate
  `src.labels.build` uses to define a breakout at all) by `ppr_ppg_n1`
  descending; every ineligible player scores below every eligible one.
- **`napkin_logistic`** — an L2 logistic on 3 features
  (`expectation_pos_rank`, `ppr_ppg_n1`, `age`), median-imputed, fit per
  fold for validation and once on train≤2023 for holdout — a genuinely
  competitive "napkin math" baseline, not a straw man.

No promotion logic — these never feed back into feature selection or
hyperparameters, purely a harder honesty check. Holdout (2024+2025
pooled), model vs. the better of these two on top-10 precision (the
brief's binding metric for this check):

| Position | Model top-10 | eligible_prior_ppg top-10 | napkin_logistic top-10 | Model PR-AUC | eligible PR-AUC | napkin PR-AUC |
|---|---|---|---|---|---|---|
| WR | 0.200 | 0.100 | 0.000 | 0.132 | 0.106 | 0.048 |
| RB | 0.200 | 0.000 | 0.100 | 0.177 | 0.061 | **0.202** |
| TE | 0.200 | 0.200 (tie) | 0.000 | 0.383 | 0.161 | 0.045 |
| QB | 0.300 | 0.100 | 0.000 | 0.395 | 0.113 | 0.050 |

Said plainly: the model beats or ties both fair baselines on top-10
precision at every position (TE ties `eligible_prior_ppg` rather than
beating it outright). On PR-AUC, RB is the one exception —
`napkin_logistic` (0.202) edges the model (0.177) — reported here rather
than hidden; RB's holdout pool is thin (4 positives) and this position
also just moved to the smaller real-market-only training pool (Step 1),
both of which push PR-AUC toward noisier at this sample size.

### Step 5 — regenerated consumers

`make board` (renormalized 2026 board), `make dashboard-build` (tier
legend + percentage-meaning banner updated), `outputs/shap_{pos}.png` +
every `outputs/model_{pos}_{report.md,metrics.json}` all regenerated
through the normal pipeline from this generation's bundles — nothing
hand-edited. Golden-player regression tests
(`tests/test_inference.py::GOLDEN_PLAYERS`) were left as originally
encoded (they check "scores above the position+season median" via each
bundle's legacy single-base-seed path, which this pass keeps unchanged
alongside the new ensemble path specifically so modules out of this
pass's scope — `src.explain.shap_report`, `src.models.vegas_experiment`,
`src.models.overlay_backtest` — and this regression check keep working
off a normal single-model bundle shape); board-schema tests were updated
for the new/renamed columns (`probability_calibrated_raw`, `tier`,
`base_rate_multiple`) and the new probability range ((0, 0.97] instead of
the old fixed [0.01, 0.95] clamp) — see `tests/test_inference.py`'s diff
for the exact assertions changed and why.

## v2.0: structural redesign — quantile projections, two lenses, comparison gate

The binary breakout classifier above predicts a rare joint event (top-K
finish AND cheap preseason ADP). At RB/TE's sample sizes that starves the
model of signal: calibrated probabilities collapse toward the position's
flat base rate (every TE reading ~1.8% regardless of who they are), and a
player who is *definitionally* ineligible (already priced as a starter)
still carries a meaningless "breakout odds" number. v2.0 adds a second
model — `src/models/quantile.py` — that predicts each player's full PPR
points-per-game *distribution* instead of one binary label, and derives
every displayed quantity (finish rank, startable odds, floor/ceiling) from
that distribution. The v1.7 classifier is **not replaced**: it stays the
comparator, and Deliverable 3's pre-stated comparison gate decides, per
position, which model actually runs the board's "Breakout Hunt" lens.

### Deliverable 1 — quantile regression head

LightGBM `objective="quantile"`, one model per position per alpha in
`[0.10, 0.25, 0.50, 0.75, 0.90]`, trained on `in_training_pool==1 AND
games>=8` rows of `features_{pos}.parquet` (target: `ppr_ppg`) — the exact
per-game rate `finish_pos_rank` itself is computed from, and the same
feature matrices/pools/`excluded_features` the classifier uses (including
RB's 2020+-only pool from Step 1). Hyperparameters are tuned ONCE at
alpha=0.50 (Optuna, 40 trials, single seed 42 — the v1.7 seed-ensemble fix
targeted small-positive-count PR-AUC noise specific to the rare-event
classifier; this head trains on every player-season with a continuous
target and no class imbalance, so it isn't exposed to that instability —
see `configs/model_{pos}.yaml`'s `quantile` block) and reused unmodified
for all 5 alphas (only `n_estimators` is refit per alpha, via that alpha's
own early stopping). Non-crossing (q10≤q25≤...≤q90) is enforced by sorting
the 5 predictions per row. Validation is the same four expanding-window
folds (`src.models.cv`); holdout is the identical 2024+2025 split, one
evaluation. Results (pinball loss per alpha not reproduced here — see
`outputs/model_{pos}_quantile_report.md` — pooled 80%-interval coverage
and Spearman(q50, actual) are the headline numbers):

| Position | Validation coverage (q10-q90, target 0.80) | Validation Spearman(q50, actual) | Holdout coverage | Holdout Spearman |
|---|---|---|---|---|
| WR | 0.760 | 0.783 | 0.657 | 0.736 |
| RB | 0.731 | 0.725 | 0.705 | 0.738 |
| TE | 0.779 | 0.656 | 0.685 | 0.743 |
| QB | 0.712 | 0.631 | 0.785 | 0.590 |

Holdout coverage runs a bit below the 0.80 target at this sample size
(2024+2025 pooled is ~120-250 rows per position) — read as "the ranges are
a little too tight, not badly miscalibrated," consistent with the general
small-sample noise this whole pipeline documents elsewhere. Spearman in the
0.7+ range says the median projection tracks actual per-game finish well.

### Deliverable 2 — derived quantities (`src/inference/projections.py`)

**Rank-to-ppg thresholds:** for every position and finish rank K, the ppg
needed to finish rank K is computed per season directly off
`labels.parquet` (the K-th ranked qualifying player's own `ppr_ppg`), then
the MEDIAN across 2014-2025 is taken and isotonic-smoothed (monotone
decreasing) into a full K=1..60 curve — `data/processed/rank_thresholds.parquet`.

**Per-player probabilities:** a player's 5 non-crossing quantiles are
treated as 5 points on his personal ppg CDF; `P(finish<=K)` for any K is
`P(ppg >= T_pos(K))`, read off that CDF by linear interpolation (linearly
extrapolated beyond q10/q90, clamped to [0.02, 0.98]).
`predicted_finish_rank` inverts the same curve at the player's own q50.

**Outcome ladder (mid-build addition):** four cumulative probabilities off
the identical CDF, no new training — `p_elite` (finish<=5), `p_startable`
(finish<= position's startable K: QB12/RB15/WR18/TE8), `p_useful`
(finish<= 2x that K — flex/bench value). Each is independently
honesty-renormalized per position (same v1.7 mechanism: a per-position
linear rescale, capped at 0.97) so the sum over ALL scored veterans at a
position matches its target exactly — 5 / K / 2K respectively — then
`p_elite<=p_startable<=p_useful` is enforced per player by clipping upward
cumulatively (independent scaling doesn't guarantee that ordering survives
for every individual player even though each curve's own aggregate sum is
exactly right). A mutually-exclusive 4-way decomposition (elite /
starter-not-elite / useful-not-startable / bust) is derived for display —
a stacked outcome bar — and sums to exactly 1 per player by construction.
The clip-up pass can push a position's realized sum slightly past its
exact target: on the live 2026 board, `p_elite`/`p_startable` land within
0.0 of target for every position, `p_useful` lands within 0.75% for WR and
1.4% for RB (QB/TE exact) — small, expected, and documented rather than
hidden (`tests/test_quantile.py` checks this on synthetic data with a ±1%
tolerance; the two positions above sit just outside that band on the real
board, a known, understood side effect of the monotonic clip, not a bug).

**Eligibility & value gap:** `breakout_eligible` reuses the exact label gate
(`expectation_pos_rank >= adp_worse_than`, `configs/labels.yaml`) — a
capped player's inflated rank makes him eligible automatically, no special
case. `value_gap = expectation_pos_rank - predicted_finish_rank` (positive
= undervalued: the market has him going later than the model expects him
to finish).

**Games-played caveat:** every quantity here is a PER-GAME projection.
Injury/availability risk is not modeled anywhere in this pipeline — see
Known Limitations.

### Deliverable 3 — the comparison gate (pre-stated, binding)

On the SAME holdout rows (`in_training_pool==1`, season in {2024,2025})
the v1.7 classifier was judged on: rank ELIGIBLE players by quantile-derived
`p_startable` (raw, pre-renormalization — a per-position positive rescale
cannot change within-position ranking, so raw and renormalized give an
identical top-10) and compute top-10 precision against `breakout==1`,
pooled across both holdout years — exactly how the classifier's own
`"model"/"pooled"/"top10_precision"` is computed. Decision: the quantile
head becomes a position's primary board engine iff its precision is >= the
classifier's own; ties favor quantile. Full table:
`outputs/comparison_gate.md` / `.json`.

| Position | Quantile top-10 precision | Classifier top-10 precision | n eligible (holdout) | Primary engine |
|---|---|---|---|---|
| WR | 0.100 | 0.200 | 193 | **classifier** |
| RB | 0.100 | 0.200 | 107 | **classifier** |
| TE | 0.200 | 0.200 | 104 | **quantile** (tie -> quantile) |
| QB | 0.200 | 0.300 | 99 | **classifier** |

TE is the one position where the quantile head ties the classifier's own
holdout top-10 precision — per the pre-stated, binding rule, ties favor the
richer quantile output, so TE's Breakout Hunt lens is ranked by
`p_startable` going forward; WR/RB/QB keep the classifier as primary (the
quantile head still supplies every Full Projections column for all four
positions regardless of this decision). All four positions' `n eligible`
counts are small enough (~100-200 holdout rows) that a single ranking flip
moves top-10 precision by a full 0.1 — read this table with the same
small-sample caution the rest of this README applies to top-10 precision
throughout.

### Deliverable 4 — board + dashboard restructure

`outputs/breakout_board_2026.csv` gains `floor_ppg`/`expected_ppg`/`ceiling_ppg`
(q10/q50/q90), `predicted_finish_rank`, `p_startable`/`p_elite`/`p_useful`,
`value_gap`, `breakout_eligible`, `primary_engine`, `projection_sentence`
alongside every existing v1.7 column. The board `.md` and the dashboard both
get two lenses: **Breakout Hunt** (eligible players only, ranked by whichever
engine Deliverable 3 picked as primary for that position — ineligible
players excluded entirely) and **Full Projections** (every scored veteran,
eligible or not, ranked by `expected_ppg`, with a floor-expected-ceiling
range bar, predicted-finish-vs-market `value_gap`, and the outcome ladder;
an ineligible player gets the badge "already priced as a starter" instead
of breakout framing). Player detail carries the projection sentence
("Projects 11.4 pts/gm (range 7.8-15.1) -> ~WR24 finish; priced WR39 -> +15
spots of value; 34% to be a weekly starter, 8% top-5"). The dashboard's
Trust view reports each position's holdout 80%-interval coverage plainly
("its 80% ranges contained the real result X% of the time in test years")
and the comparison-gate table above; Method explains the two-model,
two-lens structure in plain language.

### Deliverable 5 — tests

`tests/test_quantile.py`: non-crossing quantiles, threshold-curve
monotonicity, `p_startable` bounded in (0, 0.97], per-position ladder sums
matching 5/K/2K within ±1%, ladder monotonicity per player, segments
summing to 1, value-gap sign convention, eligible-lens exclusion of
ineligible players, tiny-config determinism (`output_root` passed
throughout, same guard as every other pipeline test in this repo), and a
comparison-gate schema check. Full suite: 307 pre-existing tests plus the
new quantile suite, all green — see the final report / CI for the exact count.

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
- **Vegas features are acquired and measured — twice, independently — and
  off by default: they don't help.** (Retires v1.5's "No Vegas data
  anywhere in training" entry.) The Odds API acquisition ran locally
  2026-08-17: preseason team lines 2020–2025 plus Week-1 player props are
  checked in under `data/external/odds_api/` (2,514 credits by the
  manifest's ledger — the data commit's 1,704 figure undercounted). Two
  separate evaluations then reached the same verdict:
  `src.models.vegas_experiment` (frozen v1.5 hyperparameters, one holdout
  eval per position under the brief's binding promotion rule: all four
  positions rejected — tie or regression on holdout top-10 precision,
  never an improvement) and `src/models/ablation_vegas.py` (independent
  fixed-hyperparameter LGBM A/B authored in a separate local session:
  no reliable lift anywhere, QB consistently mildly negative, holdout
  PR-AUC −0.09/−0.13). Shared interpretation: `expectation_pos_rank`
  (ADP/ECR is itself a market) already prices the Vegas signal, and
  2020+ coverage leaves early folds learning from nulls. The columns
  remain in every `features_{pos}.parquet` (nullable;
  `src.ingest.vegas` + `src/features/vegas.py`), excluded from every
  deployed model via `configs/model_{pos}.yaml`'s `excluded_features`
  and absent from the default `make retrain`. Week-1 props (2024–2025
  plus one 2023 game, measured) sit entirely inside the holdout years —
  zero trainable rows; raw JSON retained for when future seasons make
  them trainable. The 2026 board's manual
  `data/external/vegas_implied_2026.csv` overlay hook (presentation
  only) is unaffected.
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
- **(Fixed in v1.5, superseded in v1.7) Isotonic calibration saturated at
  the extremes in v1; v1.7 replaced the active calibrator entirely and
  added per-position renormalization.** History: at this pool size, a
  position's highest/lowest raw-score OOF bucket could land entirely on
  one class, which plain isotonic maps to an exact 0.0 or 1.0 — verified
  on the v1 build: hundreds of veteran rows across all four positions hit
  an exact 0.0 or 1.0 (mostly the low end). v1.5's
  `src.models.train.SmoothedIsotonic` fixed this at the source (Laplace/
  rule-of-succession smoothing — never exactly 0/1). v1.7 went further:
  the ACTIVE calibrator is now Platt (strictly monotone, so displayed
  ranking == raw-score ranking by construction — a stronger guarantee than
  "never exactly 0/1"), and a separate honesty gap SmoothedIsotonic never
  addressed — nothing forces a position's SUMMED calibrated probabilities
  to match how many breakouts that position actually produces in a season
  — is closed by a per-position base-rate renormalization at board time.
  See "v1.7: fixes" → "Step 3 — honest probabilities" above for the full
  writeup, the renormalization-factor table, and why tiers are now
  base-rate multiples instead of fixed cutoffs. The `[0.01, 0.95]` clamp
  and `probability_saturated` flag still exist on the pre-rescale
  `probability_calibrated_raw` column as a no-op safety net (0 rows hit it
  on this build); the displayed `probability` column's range is now
  `(0, 0.97]`, capped post-rescale.
- **The new pbp red-zone/goal-line share features are nullable, not a
  requirement.** `configs/data.yaml`'s `pbp` entry is `required: false`
  and genuinely has 0 rows for the current season (2026, no games played)
  — every model and the board must and does run fine with them null (see
  "v1.5: pbp red-zone features + retrain" above); a null rate of
  ~18-26% on the 2026 population (players with no qualifying 2025
  red-zone usage, or no 2025 games at all) is expected, not a bug.
- **Calibrated vs. raw-blend metrics are not interchangeable — compare
  like with like.** `outputs/model_{pos}_report.md` scores "the model"
  off the frozen isotonic/Platt `pred_calibrated` column;
  `outputs/overlay_backtest.md` and `outputs/vegas_experiment.md` score
  off the raw pre-calibration blend (`pred_blend`/`pred_blend_raw`) —
  deliberately, for `vegas_experiment.py`'s apples-to-apples comparison
  between two variants that never differ in calibration. These are NOT
  generally the same number even for the identical fitted models:
  isotonic regression is only *non-decreasing*, not *strictly*
  increasing, and its flat (tied) blocks can shift both PR-AUC and
  top-10 precision away from the raw score's value. Verified directly on
  this build's WR bundle — byte-identical underlying predictions
  (`pred_blend_raw` vs. `vegas_experiment.score_baseline`'s output,
  `max abs diff == 0.0`), yet PR-AUC 0.133 (raw) vs. 0.104 (calibrated)
  and top-10 precision 0.200 (raw) vs. 0.000 (calibrated). Treat a
  cross-file number mismatch as this — not as evidence of a different
  model generation — unless you've confirmed both sides used the same
  score column.
- **(Mitigated in v1.7 by seed ensembling — root cause still not fixed)
  Cross-process-invocation instability in Optuna's hyperparameter search.**
  History (v1.5/v1.6 review): re-running `run_full_pipeline` for WR three
  times in immediate succession, same seed/config/data, produced three
  different Optuna-selected hyperparameter sets (blend weights `{lgbm:
  0.1, xgb: 0.9}` vs. `{lgbm: 0.2, xgb: 0.8}`, `n_estimators` 246 vs. 488)
  — most likely BLAS/OpenMP thread-count-sensitive floating point
  somewhere in the tuning loop, since `n_jobs=1`/`deterministic=True` only
  pin the GBM libraries' own parallelism, not every linear-algebra call
  upstream of them; never root-caused. v1.7 does not fix this at the
  source (still flagged, not resolved — pinning
  `OMP_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1` and re-verifying remains
  the concrete next step). What v1.7 changes: `configs/model_{pos}.yaml`'s
  `optuna.seeds: [42, 1337, 2024]` means the SHIPPED score was never
  resting on one seed's search outcome to begin with — it's already the
  mean of three independent Optuna seeds' blended scores
  (`src.models.train`'s ensemble mode, "v1.7: fixes" → "Step 2" above),
  which directly blunts the practical impact of any one seed's search
  landing somewhere unlucky, even without root-causing why. Determinism
  itself (same seed list, same everything else ⇒ byte-identical reruns,
  including the full per-seed structure) IS locked by test within a
  single process/session
  (`tests/test_models_positions.py::test_determinism_seed_ensemble_two_runs_identical`)
  — what was never re-verified in this pass is whether the underlying
  per-seed instability across SEPARATE process invocations (the original
  finding) still reproduces; that would need the same three-separate-`python
  -m src.models.train_wr`-invocations experiment repeated against the v1.7
  ensemble path, which this pass didn't have separate scope to run. Same
  practical guidance as before still applies: regenerate
  `data/models/*.joblib`, every `outputs/model_*_{report.md,metrics.json}`,
  `outputs/overlay_backtest.md`, `outputs/vegas_experiment.md`, and
  `outputs/breakout_board_2026.*` together, from the same retrain pass,
  never independently.
