import { useState } from "react";
import { usePayload } from "./lib/usePayload";
import { Masthead } from "./components/Masthead";
import { ViewNav, type ViewId } from "./components/ViewNav";
import { BoardView } from "./views/BoardView";
import { PositionsView } from "./views/PositionsView";
import { TrustView } from "./views/TrustView";
import { MethodView } from "./views/MethodView";

export default function App() {
  const state = usePayload();
  const [view, setView] = useState<ViewId>("board");

  if (state.status === "loading") {
    return (
      <div className="wrap">
        <div className="loading-state">Loading BreakoutLab…</div>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="wrap">
        <div className="error-state">
          <p>
            <b>Couldn't load the board data.</b>
          </p>
          <p>{state.message}</p>
          <p className="fine">
            Run <code>make ui-data</code> (or <code>uv run python -m src.dashboard.export_json</code>) to build
            outputs/board_payload.json first.
          </p>
        </div>
      </div>
    );
  }

  const payload = state.data;

  return (
    <div className="wrap">
      <Masthead meta={payload.meta} />
      <ViewNav current={view} onChange={setView} />
      {view === "board" && <BoardView payload={payload} />}
      {view === "positions" && <PositionsView payload={payload} />}
      {view === "trust" && <TrustView payload={payload} />}
      {view === "method" && <MethodView payload={payload} />}
    </div>
  );
}
