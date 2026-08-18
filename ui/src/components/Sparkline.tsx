import type { DriftSeries } from "../types";
import { driftSentence } from "../lib/format";

interface Props {
  series: DriftSeries | null | undefined;
  pos: string;
}

/**
 * ECR market-drift sparkline: 2px line, endpoint dot, ported from
 * template.py's sparklineHTML(). Rank axis is inverted (lower/better rank
 * draws higher) so "line trending up" always reads as "market warming up on
 * him," matching the outcome-bar and drift-cell up/down semantics.
 */
export function Sparkline({ series, pos }: Props) {
  if (!series || series.n < 2) return null;
  const w = 320;
  const h = 64;
  const pad = 6;
  const ranks = series.ranks;
  const lo = Math.min(...ranks);
  const hi = Math.max(...ranks);
  const span = Math.max(1, hi - lo);
  const x = (i: number) => pad + (i / (ranks.length - 1)) * (w - 2 * pad);
  const y = (r: number) => pad + ((r - lo) / span) * (h - 2 * pad);
  const pts = ranks.map((r, i) => `${x(i)},${y(r)}`).join(" ");
  const lastX = x(ranks.length - 1);
  const lastY = y(ranks[ranks.length - 1]);
  const title = series.dates.map((d, i) => `${d}: rank ${ranks[i]}`).join(" | ");
  const sentence = driftSentence(series, pos);

  return (
    <div className="spark">
      <span className="lbl">Market drift — positional rank over the summer (up = rising)</span>
      <div className="spark-wrap" title={title}>
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label="Market rank trend">
          <polyline className="spark-line" points={pts} />
          <circle className="spark-dot" cx={lastX} cy={lastY} r={4.5} />
        </svg>
      </div>
      {sentence && <div className="spark-note">{sentence}</div>}
    </div>
  );
}
