# Manual overlay files

This directory holds hand-maintained CSVs the pipeline reads *if present*
and otherwise skips cleanly (every join against them is a left join —
absence means null columns downstream, never an error). See
`data/external/adp/README.md` for the historical per-season ADP override
mechanism (`src.ingest.adp`); this file covers the two 2026 board overlay
files `src.inference.board_2026` reads (Phase 6.4). Neither ever feeds a
model — both are presentation-layer only, joined onto the board after
scoring.

Both files are resolved to `gsis_id` through the same ladder every other
manual name source in this repo uses
(`src.ingest.id_map.match_to_gsis` — fantasypros_id when present, then
normalized-name + position, then fuzzy ≥93 with margin). Neither file
currently exists in this repo; the board runs fine without them (the
corresponding board columns are simply null).

## `sleeper_adp_2026.csv`

Optional. Three columns, header row required:

| column   | type | meaning                                                    |
|----------|------|-------------------------------------------------------------|
| player   | str  | player display name                                        |
| position | str  | QB, RB, WR, or TE (case-insensitive)                        |
| adp      | float| Sleeper average draft position (overall, not positional)    |

`src.inference.board_2026` re-ranks `adp` within `position` to get
`sleeper_adp_pos_rank`, then computes `adp_gap = sleeper_adp_pos_rank -
consensus_pos_rank` (the board's ECR-based consensus positional rank) —
positive means Sleeper's market has the player going later than the ECR
consensus does, i.e. a possible market inefficiency worth a second look.

This environment's egress proxy blocks Sleeper's API directly (see the
main README's "Market expectation" section for the identical restriction
on live ADP sources) — this file is the intended manual escape hatch:
export/paste current Sleeper ADP into this schema yourself.

## `vegas_implied_2026.csv`

Optional. Three columns, header row required:

| column       | type  | meaning                                                  |
|--------------|-------|------------------------------------------------------------|
| player       | str   | player display name                                        |
| position     | str   | QB, RB, WR, or TE (case-insensitive)                        |
| implied_pts  | float | a season-long Vegas-implied fantasy-points estimate (however the filler derives it — win-total/team-total-implied, a specific book's player props, etc.; this pipeline does not compute or validate the derivation) |

Passed straight through to the board's `implied_pts` column — no
re-ranking, no formula applied to it. The brief scopes Vegas data out of
*training* entirely (see the README's Known Limitations); this file is
strictly a manual, presentation-layer cross-check a reader can eyeball
next to the model's own probability.

`odds_api/` (v1.5) is a separate, code-generated subtree — see the main
README's "v1.5 Odds API (local run required)" section and
`src/ingest/odds_api.py`'s module docstring. It is not this directory's
manual-CSV mechanism: `odds_api/raw/*.json` and `odds_api/manifest.json`
are produced by `src.ingest.odds_api`'s network subcommands (local run
only, this environment's proxy blocks the API), and
`odds_api/team_lines.parquet` by its `--normalize`. Nothing currently
reads `team_lines.parquet` into `vegas_implied_2026.csv`'s schema
automatically -- a real team-level moneyline/spread/total isn't the same
quantity as `implied_pts` (a season-long fantasy-points estimate), so
that would need its own explicit conversion, not a blind copy; left as a
manual follow-up.
