import { useEffect, useRef } from "react";
import type { BoardRow, FeatureEntry } from "../types";
import { DriverChips } from "./DriverChips";
import { OutcomeBar, LadderWords } from "./OutcomeBar";
import { FeatureProfile } from "./FeatureProfile";
import { Sparkline } from "./Sparkline";

interface Props {
  row: BoardRow | null;
  features: FeatureEntry[] | undefined;
  onClose: () => void;
}

/**
 * Side drawer opened by clicking a board row: projection sentence, full
 * drivers, feature-percentile bars, ECR drift sparkline, second-opinion /
 * outcome-ladder detail. Keyboard accessible (Escape closes, focus moves
 * into the drawer on open and returns to the triggering row on close --
 * handled by BoardView, which remembers the trigger button).
 */
export function PlayerDrawer({ row, features, onClose }: Props) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!row) return;
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [row, onClose]);

  if (!row) return null;

  const sub = [row.t, row.a ? `age ${Math.round(row.a)}` : null].filter(Boolean).join(" · ");

  return (
    <div className="drawer-root">
      <div className="drawer-backdrop" onClick={onClose} />
      <div
        className="drawer-panel"
        role="dialog"
        aria-modal="true"
        aria-label={`${row.n} detail`}
        tabIndex={-1}
        ref={panelRef}
      >
        <div className="drawer-head">
          <div>
            <div className="drawer-name">
              {row.n}
              {row.bo && (
                <span
                  className="badge-broke"
                  title="Posted a breakout season in 2025 -- this year's pick is not a first-time call."
                >
                  broke out last yr
                </span>
              )}
            </div>
            <div className="drawer-sub">{sub}</div>
          </div>
          <button className="drawer-close" onClick={onClose} aria-label="Close player detail">
            ×
          </button>
        </div>

        <div className="detail">
          {row.psent ? (
            <div className="say">{row.psent}</div>
          ) : row.aps ? (
            <div className="say">
              <span className="badge-priced">Already priced as a starter</span> — the market already has him going
              inside his own position's starter range, so "breakout odds" isn't a meaningful frame for him.
            </div>
          ) : null}
          {row.r ? (
            <div className="say">{row.r}</div>
          ) : !row.psent && !row.aps ? (
            <div className="say">{row.n} — no notes recorded.</div>
          ) : null}

          {row.seg && <OutcomeBar r={row} />}
          {row.seg && <LadderWords r={row} />}

          {row.sit && row.sit.length > 0 && (
            <div className="why">
              <span className="lbl">The situation</span>
              <ul className="sit-list">
                {row.sit.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          )}

          {row.s && (
            <div className="why">
              <span className="lbl">What's driving the call</span>
              <DriverChips s={row.s} n={9} />
            </div>
          )}

          {!row.rk && row.d != null && (
            <div className="also">
              <span className="lbl">Second opinion</span>
              A separate model predicts his finish vs his price:{" "}
              {row.d > 2
                ? `beats his price by ~${Math.round(row.d)} spots — the two models agree.`
                : row.d < -2
                  ? `finishes ~${Math.abs(Math.round(row.d))} spots below the hype — the two models disagree, so treat this pick with extra caution.`
                  : "roughly matches his price."}
            </div>
          )}

          {(features?.length || (row.drift && row.drift.n >= 2)) && (
            <div className="detail-grid">
              <div>
                <FeatureProfile features={features} />
              </div>
              <div>
                <Sparkline series={row.drift} pos={row.p} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
