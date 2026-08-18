import type { Payload } from "../types";
import { POS_NAME, POS_ORDER, pct } from "../lib/format";

interface Props {
  payload: Payload;
}

/** Ported from template.py's renderPositions(). */
export function PositionsView({ payload }: Props) {
  const posView = payload.positions;
  return (
    <div className="view">
      <h1 className="vtitle">Positions</h1>
      <p className="vsub">
        Most players at every position are long shots — the model just tells you which few are worth the bench slot.
        The tail on the right is where you shop.
      </p>
      {POS_ORDER.map((pos) => {
        const pv = posView[pos.toLowerCase()];
        if (!pv) return null;
        const base = pv.base_rate != null ? (pv.base_rate * 100).toFixed(1) + "%" : "—";
        const maxBin = Math.max(1, ...pv.histogram);
        return (
          <div className="poscard" key={pos}>
            <div className="hd">
              <span className="pname">{POS_NAME[pos]}</span>
              <span className="stat">{pv.count} players scored</span>
              <span className="stat">
                historical breakout base rate <b>{base}</b>
              </span>
            </div>
            <div className="top5">
              <span className="lbl top5-lbl">Top 5 right now</span>
              {pv.top5.map((t) => (
                <span className="p5item" key={t.n}>
                  <span className="name">{t.n}</span>
                  <span className="pv">{pct(t.pr)}</span>
                </span>
              ))}
            </div>
            <span className="lbl">How conviction is distributed (share of players by chance-of-breakout bucket)</span>
            <div className="hist">
              {pv.histogram.map((n, i) => (
                <div
                  key={i}
                  className="bar"
                  data-n={n}
                  style={{ height: `${Math.max(2, Math.round((100 * n) / maxBin))}%` }}
                  title={`${i * 10}–${i * 10 + 10}%: ${n} player${n === 1 ? "" : "s"}`}
                />
              ))}
            </div>
            <div className="hist-labels">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
