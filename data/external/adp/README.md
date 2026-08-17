# Manual ADP / market-expectation overrides

`src/ingest/adp.py` builds `data/processed/market_expectation.parquet` from
the FantasyPros ECR archive (`data/raw/ff_ecr.parquet`) by default. If a file
named `<season>.csv` exists in this directory (e.g. `2024.csv`), it
**replaces** the ECR snapshot for that season entirely — used when the ECR
archive is missing a season, or its preseason snapshot is known to be bad.

## Schema

Three columns, header row required:

| column   | type | meaning                                              |
|----------|------|-------------------------------------------------------|
| player   | str  | player display name, as it would appear in nflverse   |
| position | str  | QB, RB, WR, or TE (case-insensitive — "wr" and "WR" both work; anything else is rejected with the offending values named) |
| rank     | int  | overall market rank for that preseason, 1 = consensus #1 draft priority |

Rows are resolved to `gsis_id` through the same name-matching ladder as the
ECR path (`src.ingest.id_map.match_to_gsis`). `pos_rank` is derived by
re-ranking `rank` within `position`. The resulting rows carry
`adp_source='manual'`.

## Coverage and the schedule-date asymmetry

Built coverage today is **2020-2025**, all `adp_source='ecr'`. **2026 is
pending**: ECR preseason snapshots for 2026 already exist in the archive,
but `configs/data.yaml`'s `seasons.end` is 2025, so `data/raw/schedules.parquet`
has no 2026 kickoff date to bound the snapshot window against — the ECR path
needs that date to pick the right weekly scrape out of a whole preseason of
them. It'll pick up automatically once `seasons.end` is extended and
`python -m src.ingest.nflverse` is re-run.

A manual CSV does **not** have this dependency. `build_season` checks for
`<season>.csv` before it ever looks at the schedule, so a manual file for a
season without cached schedule data (2026 included) is still ingested. This
asymmetry is intentional: a manually curated file is trusted as the final
answer for that season, while the ECR path is picking one snapshot out of
many and needs the kickoff date to know which one is "final".

No files currently exist in this directory.
