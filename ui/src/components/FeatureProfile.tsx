import type { FeatureEntry } from "../types";
import { fmtFeatureValue } from "../lib/format";

interface Props {
  features: FeatureEntry[] | undefined;
}

/** Per-feature percentile bars, ported from template.py's featureProfileHTML(). */
export function FeatureProfile({ features }: Props) {
  if (!features || features.length === 0) return null;
  return (
    <div className="profile">
      <span className="lbl">His feature profile (percentile vs. every player at his position this year)</span>
      {features.map((f) => {
        const width = Math.max(2, Math.round(f.pctl));
        return (
          <div className="profile-row" key={f.feat}>
            <span className="flbl">{f.label}</span>
            <span className="ftrack">
              <span className="ffill" style={{ width: `${width}%` }} />
            </span>
            <span className="fval">{fmtFeatureValue(f.value)}</span>
          </div>
        );
      })}
    </div>
  );
}
