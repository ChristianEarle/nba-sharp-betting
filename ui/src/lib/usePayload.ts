import { useEffect, useState } from "react";
import type { Payload } from "../types";

export type PayloadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: Payload };

/**
 * Fetches /data/board_payload.json (dev: served by vite.config.ts's
 * board-payload plugin middleware from outputs/board_payload.json; build:
 * copied into dist/data/ by that same plugin's closeBundle hook). Uses
 * import.meta.env.BASE_URL so the relative fetch resolves correctly both at
 * dev-server root and under the GitHub Pages /nba-sharp-betting/ subpath
 * (vite.config.ts sets base: './').
 */
export function usePayload(): PayloadState {
  const [state, setState] = useState<PayloadState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    const url = `${import.meta.env.BASE_URL}data/board_payload.json`;
    fetch(url)
      .then((res) => {
        if (!res.ok) {
          throw new Error(
            `${res.status} ${res.statusText} fetching ${url}` +
              (res.status === 404 ? " -- run `make ui-data` to build outputs/board_payload.json first" : ""),
          );
        }
        return res.json() as Promise<Payload>;
      })
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data });
      })
      .catch((err: unknown) => {
        if (!cancelled) setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
