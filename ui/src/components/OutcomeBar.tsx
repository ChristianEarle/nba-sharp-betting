import type { BoardRow } from "../types";
import { pct } from "../lib/format";

interface Props {
  r: Pick<BoardRow, "seg" | "pel" | "pst" | "pus">;
}

/**
 * Four-segment outcome-ladder bar -- elite / starter-not-elite /
 * useful-not-startable / bust, mutually exclusive, always summing to 100%.
 * Ported from template.py's outcomeBarHTML()/ladderWordsHTML().
 */
export function OutcomeBar({ r }: Props) {
  if (!r.seg) return null;
  const [elite, starter, useful, bust] = r.seg;
  const title = `Top-5: ${pct(r.pel)} · Weekly starter: ${pct(r.pst)} · Flex/bench value: ${pct(r.pus)} · Bust: ${pct(
    1 - (r.pus ?? 0),
  )}`;
  return (
    <div className="outcomebar" title={title} role="img" aria-label={title}>
      <span className="seg-elite" style={{ width: `${Math.max(0, elite * 100)}%` }} />
      <span className="seg-starter" style={{ width: `${Math.max(0, starter * 100)}%` }} />
      <span className="seg-useful" style={{ width: `${Math.max(0, useful * 100)}%` }} />
      <span className="seg-bust" style={{ width: `${Math.max(0, bust * 100)}%` }} />
    </div>
  );
}

export function LadderWords({ r }: Props) {
  if (r.pel == null) return null;
  return (
    <div className="ladder-words">
      <b>Outcome ladder</b> (cumulative): top-5 season <b>{pct(r.pel)}</b> &middot; weekly starter <b>{pct(r.pst)}</b>{" "}
      &middot; flex/bench value <b>{pct(r.pus)}</b> &middot; bust <b>{pct(1 - (r.pus ?? 0))}</b>
    </div>
  );
}
