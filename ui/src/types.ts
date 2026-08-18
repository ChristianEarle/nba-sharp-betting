/**
 * TypeScript types for outputs/board_payload.json, produced by
 * src/dashboard/export_json.py (which reuses src/dashboard/build.py's
 * assemble_payload() read-only). Field names/shapes here must track that
 * payload's schema exactly -- see build.py's board_rows() for the
 * short-key -> meaning mapping (e.g. "n" = player name, "p" = position,
 * "pr" = display probability).
 */

export type Position = "QB" | "RB" | "WR" | "TE";

export interface DriftSeries {
  dates: string[];
  ranks: number[];
  /** earliest pos_rank - latest pos_rank; positive = market rising on him (rank number falling). */
  drift: number;
  n: number;
}

/** Four mutually-exclusive shares [elite, starter, useful, bust] that always sum to 1. */
export type OutcomeSegments = [number, number, number, number];

export interface BoardRow {
  key: string;
  n: string;
  p: Position;
  a: number | null;
  t: string | null;
  rk: boolean;
  pr: number | null;
  rs: number | null;
  tier: string | null;
  brm: number | null;
  e: number | null;
  dr: number | null;
  d: number | null;
  r: string | null;
  s: string | null;
  adp: number | null;
  sadp: number | null;
  gap: number | null;
  vegas: number | null;
  edge: number | null;
  avail: string | null;
  sat: boolean | null;
  drift: DriftSeries | null;
  bo: boolean;
  hs: number | null;
  elig: boolean | null;
  eng: string | null;
  fppg: number | null;
  eppg: number | null;
  cppg: number | null;
  cr: number | null;
  pfr: number | null;
  pel: number | null;
  pst: number | null;
  pus: number | null;
  seg: OutcomeSegments | null;
  vg: number | null;
  vgc: number | null;
  aps: boolean | null;
  psent: string | null;
  /** v2.5 roster-situation lines: arrivals, departures w/ vacated usage+TDs, competition, pecking order. */
  sit?: string[] | null;
}

export interface FeatureEntry {
  feat: string;
  label: string;
  value: number;
  pctl: number;
}

export interface PositionTop5Entry {
  n: string;
  pr: number | null;
}

export interface PositionView {
  count: number;
  base_rate: number | null;
  top5: PositionTop5Entry[];
  histogram: number[];
}

export interface TrustPositionEntry {
  holdout_pr_auc: number | null;
  holdout_top10_precision: number | null;
  holdout_n: number | null;
  holdout_n_pos: number | null;
  best_baseline_name: string | null;
  best_baseline_pr_auc: number | null;
  validation_pooled_pr_auc: number | null;
  blend_weights: Record<string, number> | null;
  holdout_regression_rmse: Record<string, number> | null;
}

export interface QuantileGate {
  primary_engine: string;
  quantile_top10_precision: number | null;
  classifier_top10_precision: number | null;
}

export interface QuantileTrustEntry {
  coverage_q10_q90: number | null;
  spearman_q50_actual: number | null;
  pinball_q50: number | null;
  gate: QuantileGate | null;
}

export interface NamedListEntry {
  player: string | null;
  prob: number | null;
  expect_rank: string | null;
  actual_rank: string | null;
  hit: boolean;
}

export interface BacktestModelPick {
  player: string;
  prob: number;
  exp_rank: number | null;
  finish_rank: number | null;
  breakout: boolean;
}

export interface BacktestActualFinisher {
  player: string;
  finish_rank: number;
  exp_rank: number | null;
  /** Where the holdout-time model ranked him (1 = its favorite); null = never scored (rookie / outside pool). */
  model_rank: number | null;
  prob: number | null;
  rookie: boolean;
  breakout: boolean;
  /** false = priced better than the breakout gate preseason; never in the model's scope. */
  eligible: boolean;
}

export interface BacktestSeason {
  model: BacktestModelPick[];
  actual: BacktestActualFinisher[];
  n_scored: number;
}

export interface Trust {
  positions: Record<string, TrustPositionEntry>;
  named_lists: Record<string, Record<string, NamedListEntry[]>>;
  quantile: Record<string, QuantileTrustEntry>;
  backtest_top10?: Record<string, Record<string, BacktestSeason>>;
}

export interface Meta {
  season: number;
  snapshot_date: string | null;
  model_generated_at: string | null;
  counts: {
    veterans: number;
    rookies: number;
    by_position: Record<string, number>;
  };
}

export interface LabelsConfig {
  thresholds: Record<string, { finish_top: number; adp_worse_than: number }>;
  min_games: number;
}

export interface Payload {
  meta: Meta;
  board: BoardRow[];
  features: Record<string, FeatureEntry[]>;
  ecr: Record<string, DriftSeries>;
  positions: Record<string, PositionView>;
  trust: Trust;
  labels_config: LabelsConfig;
}
