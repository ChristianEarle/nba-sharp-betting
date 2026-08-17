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
| position | str  | one of QB, RB, WR, TE                                  |
| rank     | int  | overall market rank for that preseason, 1 = consensus #1 draft priority |

Rows are resolved to `gsis_id` through the same name-matching ladder as the
ECR path (`src.ingest.id_map.match_to_gsis`). `pos_rank` is derived by
re-ranking `rank` within `position`. The resulting rows carry
`adp_source='manual'`.

No files currently exist in this directory — every in-range season
(2020-2026) is sourced from ECR.
