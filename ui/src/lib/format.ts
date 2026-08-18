import type { DriftSeries, BoardRow, Position } from "../types";

/** Matches template.py's `pct` helper: null -> em-dash, else rounded percent. */
export function pct(v: number | null | undefined): string {
  return v == null ? "—" : Math.round(v * 100) + "%";
}

export interface CostWords {
  main: string;
  sub: string;
}

/** Draft-cost-in-words cell, ported from template.py's costCell(). */
export function costWords(r: Pick<BoardRow, "rk" | "dr" | "e" | "p">): CostWords {
  if (r.rk) {
    return { main: r.dr ? `Round ${r.dr} pick` : "—", sub: "NFL draft" };
  }
  if (r.e == null) {
    return { main: "Undrafted", sub: "free in most leagues" };
  }
  const note = r.e <= 12 ? "goes early" : r.e <= 30 ? "mid-round price" : r.e <= 55 ? "late-round price" : "basically free";
  return { main: `~${r.p}${r.e}`, sub: note };
}

export interface DriftLabel {
  text: string;
  dir: "up" | "dn" | "flat";
}

/** Market-drift arrow cell, ported from template.py's driftCell(). */
export function driftLabel(drift: DriftSeries | null | undefined): DriftLabel {
  if (!drift || drift.n < 2) return { text: "—", dir: "flat" };
  const delta = drift.drift;
  if (Math.abs(delta) < 2) return { text: "flat", dir: "flat" };
  const up = delta > 0;
  return { text: `${Math.abs(delta)} spot${Math.abs(delta) === 1 ? "" : "s"}`, dir: up ? "up" : "dn" };
}

/** Plain-language market-drift sentence, ported from template.py's driftSentence(). */
export function driftSentence(drift: DriftSeries | null | undefined, pos: string): string | null {
  if (!drift || drift.n < 2) return null;
  const delta = drift.drift;
  if (Math.abs(delta) < 2) return "The market hasn't moved much on him since early June.";
  const dir = delta > 0 ? "risen" : "fallen";
  const tail = delta > 0 ? " — other drafters are catching on." : " — other drafters are cooling on him too.";
  return `The market's own consensus rank has ${dir} ~${Math.abs(delta)} spot${Math.abs(delta) === 1 ? "" : "s"} at ${pos} since early June (${drift.n} weekly snapshots)${tail}`;
}

export interface DriverChip {
  label: string;
  dir: "up" | "dn" | null;
}

/**
 * Splits r.s (shap_top3, "<honest state label>:<+/->,...") into chip data.
 * Renders the server-resolved label verbatim -- never re-derives a label
 * from a feature code client-side (see build.py's driverChips() docstring
 * in template.py for why: a value-blind re-derivation can render a
 * misleading chip, e.g. "New team +" for a player whose actual boost came
 * from team_change=0).
 */
export function parseDriverChips(s: string | null | undefined, n: number): DriverChip[] {
  if (!s) return [];
  return s
    .split(", ")
    .filter(Boolean)
    .slice(0, n)
    .map((tok) => {
      const i = tok.lastIndexOf(":");
      const label = i > 0 ? tok.slice(0, i).trim() : tok.trim();
      const dirStr = i > 0 ? tok.slice(i + 1).trim() : "";
      const dir: "up" | "dn" | null = dirStr ? (dirStr.startsWith("+") ? "up" : "dn") : null;
      return { label, dir };
    });
}

/** Feature-value formatting, ported from template.py's featureProfileHTML(). */
export function fmtFeatureValue(v: number): string {
  return Math.abs(v) >= 100 ? String(Math.round(v)) : String(Math.round(v * 100) / 100);
}

export const POS_ORDER: Position[] = ["QB", "RB", "WR", "TE"];
export const POS_NAME: Record<Position, string> = {
  QB: "Quarterback",
  RB: "Running back",
  WR: "Wide receiver",
  TE: "Tight end",
};

export const TIER_CLASS: Record<string, string> = {
  "Elite target": "t4",
  "Strong swing": "t3",
  "Worth a flier": "t2",
  "Long shot": "t1",
};

export function tierInfo(r: Pick<BoardRow, "tier">): { cls: string; label: string } {
  const label = r.tier || "Long shot";
  return { cls: TIER_CLASS[label] || "t1", label };
}

/** Whichever engine the comparison gate picked as primary for this row's position. */
export function huntScore(r: Pick<BoardRow, "hs" | "pr">): number | null {
  return r.hs != null ? r.hs : r.pr;
}
