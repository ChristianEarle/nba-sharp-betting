# BreakoutLab UI

A hand-rolled Vite + React + TypeScript rebuild of the generated
[`src/dashboard/build.py`](../src/dashboard/build.py) single-file dashboard,
as a proper local dev app: React state instead of vanilla-JS DOM diffing, a
typed data contract, and componentized views. It reads the identical
underlying data -- same board rows, feature profiles, market-drift series,
trust metrics, and method copy -- through a standalone JSON export instead of
an HTML-embedded `<script type="application/json">` blob.

No UI framework, no chart library: every visual (meters, range bars, outcome
ladders, the ECR sparkline) is hand-built CSS/SVG against the same design
tokens the generated dashboard uses, so the two surfaces are visually
interchangeable.

## Data contract

`src/dashboard/export_json.py` reuses `src/dashboard/build.py`'s
`assemble_payload()` (read-only import, no reimplementation) and writes the
sanitized payload to `outputs/board_payload.json` -- compact
(`pretty=false`), `sort_keys=True`, `allow_nan=False`, byte-identical across
back-to-back runs from unchanged inputs (same determinism contract as the
HTML dashboard build; see that module's docstring).

```
uv run python -m src.dashboard.export_json [--out PATH]
```

`ui/src/types.ts` mirrors that payload's schema field-for-field (including
`build.py`'s short keys, e.g. `n` = player name, `p` = position, `pr` =
display probability -- see `board_rows()`'s comments for the full mapping).

## Running it

From the repo root, via `make` (see the Makefile's `ui-*` targets):

```
make ui-data    # writes outputs/board_payload.json
make ui         # ui-data, npm install if needed, then `vite dev` on :5173
make ui-build   # ui-data, then `tsc -b && vite build` -> ui/dist/
make ui-publish # ui-build, then flat-copies ui/dist/ into docs/ (GitHub Pages)
```

Or directly, once `outputs/board_payload.json` exists:

```
cd ui
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + production build -> ui/dist/
```

`npm run build` alone (without going through `make ui-build`) also produces
a self-contained `ui/dist/` -- `vite.config.ts`'s `board-payload` plugin
copies `outputs/board_payload.json` into `dist/data/` as part of the build,
the same way its dev-server middleware serves that file live at
`/data/board_payload.json` in `npm run dev`.

## How the data gets to the browser

The app fetches `` `${import.meta.env.BASE_URL}data/board_payload.json` ``
(`src/lib/usePayload.ts`). `vite.config.ts` sets `base: './'` (a relative
base) so the same build works unmodified whether served from the repo root
(local `dist/`) or under GitHub Pages' `/nba-sharp-betting/` subpath
(`docs/`, via `make ui-publish`) -- the browser resolves the relative fetch
against wherever `index.html` actually loaded from.

## Structure

```
src/
  types.ts                 payload schema (mirrors board_payload.json)
  lib/
    usePayload.ts           fetch + loading/error state
    format.ts                display helpers ported from template.py's JS
  components/
    Masthead.tsx, ViewNav.tsx
    DriverChips.tsx, OutcomeBar.tsx, RangeBar.tsx
    FeatureProfile.tsx, Sparkline.tsx    (hand-rolled SVG, no chart lib)
    PlayerDrawer.tsx          side drawer opened by clicking a board row
  views/
    BoardView.tsx             position tabs, lens toggle, search/sort, rows
    PositionsView.tsx, TrustView.tsx, MethodView.tsx
  App.tsx, main.tsx, index.css (design tokens + all component styles)
```

`PlayerDrawer` is this app's one deliberate UX deviation from the generated
dashboard: the template expands a row's detail inline; this app opens a real
side drawer (Escape or the backdrop closes it, focus moves in on open) since
that's a more natural fit for a stateful React app than an inline
expand-in-place list item.

## Design system

Tokens, fonts (Barlow Condensed / Barlow / IBM Plex Mono via Google Fonts),
and every component class name are ported directly from
`src/dashboard/template.py`'s `<style>` block, including the three-state
theme contract: bare `:root` light tokens, a
`prefers-color-scheme: dark`-guarded `:root:not([data-theme="light"])`
block, and an explicit `:root[data-theme="dark"]` override block for a
manual toggle to win in both directions.
