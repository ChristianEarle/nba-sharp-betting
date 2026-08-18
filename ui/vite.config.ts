import { copyFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// ui/ lives one directory below the repo root; outputs/board_payload.json
// (written by `uv run python -m src.dashboard.export_json`, see Makefile's
// ui-data target) lives at REPO_ROOT/outputs/board_payload.json.
const REPO_ROOT = resolve(import.meta.dirname, "..");
const PAYLOAD_PATH = resolve(REPO_ROOT, "outputs", "board_payload.json");
const PAYLOAD_URL_PATH = "/data/board_payload.json";

/**
 * Serves outputs/board_payload.json at /data/board_payload.json:
 *  - dev: a middleware reads the file fresh on every request (so `make
 *    ui-data` + a browser refresh picks up new data without restarting the
 *    dev server).
 *  - build: closeBundle copies it into dist/data/board_payload.json so the
 *    built site is self-contained (no separate data-publishing step).
 * App code fetches `${import.meta.env.BASE_URL}data/board_payload.json`
 * (src/lib/usePayload.ts), which the browser resolves to this absolute path
 * regardless of vite's `base` setting.
 */
function boardPayloadPlugin(): Plugin {
  return {
    name: "board-payload",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url !== PAYLOAD_URL_PATH) {
          next();
          return;
        }
        if (!existsSync(PAYLOAD_PATH)) {
          res.statusCode = 404;
          res.setHeader("Content-Type", "application/json");
          res.end(
            JSON.stringify({
              error: "outputs/board_payload.json not found -- run `make ui-data` first",
            }),
          );
          return;
        }
        res.setHeader("Content-Type", "application/json");
        res.end(readFileSync(PAYLOAD_PATH));
      });
    },
    closeBundle() {
      if (!existsSync(PAYLOAD_PATH)) return;
      const outDir = resolve(import.meta.dirname, "dist", "data");
      mkdirSync(outDir, { recursive: true });
      copyFileSync(PAYLOAD_PATH, resolve(outDir, "board_payload.json"));
    },
  };
}

export default defineConfig({
  // Relative base so the build works unmodified under the GitHub Pages
  // /nba-sharp-betting/ subpath (see Makefile's ui-publish target, which
  // copies ui/dist/* flat into docs/).
  base: "./",
  plugins: [react(), boardPayloadPlugin()],
  server: {
    port: 5173,
    fs: {
      allow: [REPO_ROOT],
    },
  },
});
