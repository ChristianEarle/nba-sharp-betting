import type { Meta } from "../types";

interface Props {
  meta: Meta;
}

export function Masthead({ meta }: Props) {
  return (
    <header className="masthead">
      <div className="wordmark">
        BREAKOUT<span className="lab">LAB</span>
      </div>
      <div className="season-chip">{meta.season} DASHBOARD</div>
      <div className="meta">
        Prices as of {meta.snapshot_date || "—"}
        <br />
        Models last trained {meta.model_generated_at || "—"}
      </div>
    </header>
  );
}
