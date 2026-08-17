.PHONY: refresh retrain test board

# All targets assume `uv sync --extra dev` has already been run once; see README Setup.

# Re-pull only the seasonal datasets that actually carry 2026 rows (rosters,
# schedules; draft_picks/combine/players/ff_playerids are non-seasonal and
# just get re-clipped to the current seasons.end), rebuild the historical
# labels/features tables for reproducibility, then re-score the 2026 board
# with whatever model bundles already exist on disk. Does NOT retrain --
# safe to run often (e.g. after nflverse republishes a roster move) without
# burning an Optuna budget.
refresh:
	@echo ">> refresh: pulling 2026-relevant nflverse data (rosters, schedules, draft_picks, combine, players, ff_playerids)"
	uv run python -m src.ingest.nflverse --force --only rosters schedules draft_picks combine players ff_playerids
	@echo ">> refresh: rebuilding market_expectation.parquet (ADP/ECR, now including any new 2026 snapshot)"
	uv run python -m src.ingest.adp
	@echo ">> refresh: rebuilding labels.parquet (historical breakout labels, 2014-2025)"
	uv run python -m src.labels.build
	@echo ">> refresh: rebuilding features_{wr,rb,te,qb}.parquet (historical training features)"
	uv run python -m src.features.wr
	uv run python -m src.features.rb
	uv run python -m src.features.te
	uv run python -m src.features.qb
	@echo ">> refresh: re-scoring outputs/breakout_board_2026.{csv,md} with existing model bundles (NO retraining)"
	uv run python -m src.inference.board_2026

# Full pipeline: everything refresh does, plus the five training runs
# (WR/RB/TE/QB's real Optuna-tuned classifier+regressor pipeline, and the
# rookie heuristic's shallow logistic -- five, not the brief's four,
# because the rookie heuristic is also a training step even though it's
# not one of the four Phase-4 position models; see README) and SHAP
# report regeneration. Expensive (real Optuna trial counts, not the tests'
# tiny config) -- run this when the underlying data has meaningfully
# changed, not on every board refresh.
retrain: refresh
	@echo ">> retrain: WR classifier+regressor (full Optuna config)"
	uv run python -m src.models.train_wr
	@echo ">> retrain: RB classifier+regressor (full Optuna config)"
	uv run python -m src.models.train_rb
	@echo ">> retrain: TE classifier+regressor (full Optuna config)"
	uv run python -m src.models.train_te
	@echo ">> retrain: QB classifier+regressor (full Optuna config)"
	uv run python -m src.models.train_qb
	@echo ">> retrain: rookie heuristic (shallow logistic, historical rookies)"
	uv run python -m src.models.rookie_heuristic
	@echo ">> retrain: SHAP global beeswarm plots (outputs/shap_{pos}.png)"
	uv run python -m src.explain.shap_report
	@echo ">> retrain: re-scoring outputs/breakout_board_2026.{csv,md} with freshly retrained bundles"
	uv run python -m src.inference.board_2026

# Full pytest suite.
test:
	@echo ">> test: uv run pytest -q"
	uv run pytest -q

# Score-only: rebuild the 2026 board from whatever data + bundles already
# exist on disk. No data pull, no retraining -- the fast path for "I just
# want the board."
board:
	@echo ">> board: scoring outputs/breakout_board_2026.{csv,md} from existing data + bundles"
	uv run python -m src.inference.board_2026
