import type { BoardRow } from "../types";

interface Props {
  r: Pick<BoardRow, "fppg" | "eppg" | "cppg">;
  lo: number;
  hi: number;
}

/** Floor-expected-ceiling range bar, ported from template.py's rangeBarHTML(). */
export function RangeBar({ r, lo, hi }: Props) {
  if (r.eppg == null || r.fppg == null || r.cppg == null) {
    return <span className="rangebar-val">—</span>;
  }
  const span = Math.max(1, hi - lo);
  const left = Math.max(0, Math.min(100, (100 * (r.fppg - lo)) / span));
  const right = Math.max(0, Math.min(100, (100 * (r.cppg - lo)) / span));
  const dot = Math.max(0, Math.min(100, (100 * (r.eppg - lo)) / span));
  return (
    <>
      <span className="rangebar" title={`floor ${r.fppg.toFixed(1)} – ceiling ${r.cppg.toFixed(1)} pts/gm`}>
        <span className="rb-range" style={{ left: `${left}%`, width: `${Math.max(1, right - left)}%` }} />
        <span className="rb-dot" style={{ left: `${dot}%` }} />
      </span>
      <span className="rangebar-val">{r.eppg.toFixed(1)}</span>
    </>
  );
}

export function gapCellText(vg: number | null | undefined): { text: string; cls: string } {
  if (vg == null) return { text: "—", cls: "" };
  const sign = vg > 0 ? "+" : "";
  return { text: `${sign}${Math.round(vg)}`, cls: vg > 0 ? "pos" : vg < 0 ? "neg" : "" };
}
