import { useMemo, useRef, useState } from "react";
import type { BoardRow, Payload, Position } from "../types";
import { costWords, driftLabel, huntScore, tierInfo } from "../lib/format";
import { RangeBar, gapCellText } from "../components/RangeBar";
import { PlayerDrawer } from "../components/PlayerDrawer";

type TabId = Position | "RK";
type Lens = "hunt" | "full";
type SortKey = "pr" | "e" | "a" | "drift";

const TABS: { id: TabId; label: string }[] = [
  { id: "QB", label: "QB" },
  { id: "RB", label: "RB" },
  { id: "WR", label: "WR" },
  { id: "TE", label: "TE" },
  { id: "RK", label: "ROOKIES" },
];

interface Props {
  payload: Payload;
}

export function BoardView({ payload }: Props) {
  const { board, features, meta } = payload;
  const [cur, setCur] = useState<TabId>("WR");
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("pr");
  const [lens, setLens] = useState<Lens>("hunt");
  const [openKey, setOpenKey] = useState<string | null>(null);

  const vets = useMemo(() => board.filter((r) => !r.rk), [board]);
  const rooks = useMemo(() => board.filter((r) => r.rk), [board]);
  const tabRows = useMemo<Record<TabId, BoardRow[]>>(
    () => ({
      QB: vets.filter((r) => r.p === "QB"),
      RB: vets.filter((r) => r.p === "RB"),
      WR: vets.filter((r) => r.p === "WR"),
      TE: vets.filter((r) => r.p === "TE"),
      RK: rooks,
    }),
    [vets, rooks],
  );

  const useFull = lens === "full" && cur !== "RK";
  const q = query.trim().toLowerCase();

  const rows = useMemo(() => {
    let list = tabRows[cur].filter((r) => !q || r.n.toLowerCase().includes(q) || (r.t || "").toLowerCase().includes(q));
    if (cur !== "RK") {
      if (useFull) {
        list = list.filter((r) => r.eppg != null);
      } else {
        list = list.filter((r) => r.elig !== false);
      }
    }
    const dir = sortKey === "e" || sortKey === "a" ? 1 : -1;
    if (useFull) {
      list = [...list].sort((x, y) => (y.eppg ?? -Infinity) - (x.eppg ?? -Infinity));
    } else {
      list = [...list].sort((x, y) => {
        if (sortKey === "drift") {
          const a = x.drift ? x.drift.drift : null;
          const b = y.drift ? y.drift.drift : null;
          if (a == null && b == null) return (huntScore(y) ?? 0) - (huntScore(x) ?? 0);
          if (a == null) return 1;
          if (b == null) return -1;
          return b - a;
        }
        const xk = sortKey === "pr" ? huntScore(x) : x[sortKey];
        const yk = sortKey === "pr" ? huntScore(y) : y[sortKey];
        if (xk == null && yk == null) return (huntScore(y) ?? 0) - (huntScore(x) ?? 0);
        if (xk == null) return 1;
        if (yk == null) return -1;
        return dir * ((xk as number) - (yk as number)) || (y.rs ?? 0) - (x.rs ?? 0);
      });
    }
    return list;
  }, [tabRows, cur, q, useFull, sortKey]);

  const rangeLo = useFull ? Math.min(...rows.map((r) => r.fppg).filter((v): v is number => v != null), 0) : 0;
  const rangeHi = useFull ? Math.max(...rows.map((r) => r.cppg).filter((v): v is number => v != null), 1) : 1;
  const maxP = cur === "RK" ? 1 : 0.7;

  const openRow = openKey != null ? board.find((r) => r.key === openKey) || null : null;
  const lastTriggerRef = useRef<HTMLButtonElement | null>(null);

  function selectTab(id: TabId) {
    setCur(id);
    setOpenKey(null);
  }

  function closeDrawer() {
    setOpenKey(null);
    // Return focus to the row that opened the drawer -- keyboard users
    // shouldn't lose their place in the list when the drawer closes.
    lastTriggerRef.current?.focus();
  }

  return (
    <div className="view" id="viewBoard">
      <div className="howto">
        <div className="step">
          <span className="k">1 &middot; Draft normally early</span>
          <p>
            Rounds 1–6, take your studs and <b>ignore this page</b>. Everyone here is, by definition, available
            later.
          </p>
        </div>
        <div className="step">
          <span className="k">2 &middot; From round 7, open this</span>
          <p>
            When your pick comes and the options all look the same, <b>take the highest percentage</b> on this
            board. That's the whole system.
          </p>
        </div>
        <div className="step">
          <span className="k">3 &middot; Check the news first</span>
          <p>
            This board <b>doesn't know about injuries</b>, suspensions, or holdouts. Thirty seconds of
            news-checking beats the math.
          </p>
        </div>
      </div>

      <p className="caveat">
        <b>What the percentage means:</b> the chance this cheap player finishes the season as a true weekly starter
        anyway (top-12 QB, top-15 RB, top-18 WR, top-8 TE). Each position's percentages are normalized so the total
        across everyone scored at that position matches how many breakouts actually happen there in a typical season
        — so a 25% WR and a 25% QB reflect the same kind of real base-rate comparison, not two calibration curves
        that happen to output the same number. The tier label below is a multiple of a random late pick's odds at
        that spot (see Verdict in "How to read this board") — even an Elite target won't hit every time.{" "}
        <b>Every row also has a projected per-game scoring range now</b> (tap to open) — switch to Full Projections
        below to browse by that instead of by breakout odds.
      </p>

      {cur !== "RK" && (
        <div className="lenstoggle" role="tablist" aria-label="Board lens">
          <button className="lenstab" role="tab" aria-selected={lens === "hunt"} onClick={() => { setLens("hunt"); setOpenKey(null); }}>
            Breakout Hunt
          </button>
          <button className="lenstab" role="tab" aria-selected={lens === "full"} onClick={() => { setLens("full"); setOpenKey(null); }}>
            Full Projections
          </button>
        </div>
      )}

      <nav className="controls" aria-label="Board controls">
        <div className="tabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              className="tab"
              role="tab"
              aria-selected={t.id === cur}
              onClick={() => selectTab(t.id)}
            >
              {t.label}
              <span className="ct">{tabRows[t.id].length}</span>
            </button>
          ))}
        </div>
        <input
          type="search"
          placeholder="Search player or team…"
          aria-label="Search player or team"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select aria-label="Sort by" value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}>
          <option value="pr">Best bets first</option>
          <option value="e">Cheapest first</option>
          <option value="a">Youngest first</option>
          <option value="drift">Rising fastest</option>
        </select>
      </nav>

      {cur === "RK" && (
        <p className="rookie-note" style={{ display: "block" }}>
          <b>Rookie section — bigger guesses.</b> Rookies have no NFL history, so a simpler model guesses from draft
          position and landing spot. Use these only to break ties on your last couple of picks.
        </p>
      )}

      <div className="colhead">
        {useFull ? (
          <>
            <div className="num">#</div>
            <div>Player</div>
            <div>Value gap</div>
            <div>Floor &ndash; expected &ndash; ceiling</div>
            <div>Market</div>
            <div>Note</div>
          </>
        ) : (
          <>
            <div className="num">#</div>
            <div>Player</div>
            <div>Draft cost</div>
            <div>Chance he breaks out</div>
            <div>Market</div>
            <div>Verdict</div>
          </>
        )}
      </div>

      <ol className="board">
        {rows.map((r, i) => (
          <BoardRowItem
            key={r.key}
            r={r}
            index={i}
            useFull={useFull}
            maxP={maxP}
            rangeLo={rangeLo}
            rangeHi={rangeHi}
            open={openKey === r.key}
            onOpen={(btn) => {
              lastTriggerRef.current = btn;
              setOpenKey(r.key);
            }}
          />
        ))}
      </ol>
      {rows.length === 0 && <div className="empty">No players match.</div>}

      <PlayerDrawer row={openRow} features={openRow ? features[openRow.key] : undefined} onClose={closeDrawer} />

      <section className="reading">
        <h2>How to read this board</h2>
        <div className="glossary">
          <div>
            <b>Draft cost — "goes ~WR39"</b>
            Where drafters are currently taking him: the 39th receiver picked, i.e. a bench-round price. Lower
            number = more expensive.
          </div>
          <div>
            <b>The percentage</b>
            The model's calibrated odds he finishes as a weekly starter despite that cheap price, rescaled so each
            position's percentages add up to how many breakouts that position actually produces in a season. It's
            the only number you need on draft night.
          </div>
          <div>
            <b>Verdict</b>
            That percentage as a multiple of a normal late pick's odds at his position: <b>Elite target</b> (8x or
            more), <b>Strong swing</b> (4–8x), <b>Worth a flier</b> (2–4x), <b>Long shot</b> (under 2x).
          </div>
          <div>
            <b>Market ▲ / ▼</b>
            How his preseason ranking has moved across the summer's weekly consensus snapshots. ▲ = the market is
            warming up on him since June; ▼ = it's cooling.
          </div>
          <div>
            <b>Breakout Hunt vs Full Projections</b>
            Breakout Hunt only shows players who are still cheap enough to qualify as a breakout candidate at all,
            ranked by breakout odds. Full Projections shows every scored veteran, ranked by his projected per-game
            scoring, with a floor–expected–ceiling range bar and an "already priced as a starter" badge on anyone too
            expensive to be a breakout by definition.
          </div>
          <div>
            <b>Two ranks, not one</b>
            Every projection carries two different ranks: <b>where he stacks up against this year's players</b> at
            his position (the "Projects PosN of 2026" line — the same field the market's own consensus ranking
            covers), and <b>what a typical season with his median projection would have finished historically</b>{" "}
            (the "typical finish" line). A median projection can't reach an all-time RB1's extreme season, so even
            this year's #1 projected back can read as a "typical RB10" — that's not a contradiction, just two
            different questions about the same number.
          </div>
          <div>
            <b>Value gap</b>
            Consensus market rank minus the model's typical-finish rank — positive means the market has him going
            later than a season like his median projection has historically finished. (The tested-and-validated
            version of this gap; see Method.)
          </div>
          <div>
            <b>Outcome bar (tap a row to see the words)</b>
            Four honest, non-overlapping shares of a player's season: top-5 finish, weekly-starter finish, still-useful
            bench/flex finish, and bust — they always add up to 100%.
          </div>
          <div>
            <b>▲ and ▼ inside a row (tap to open)</b>
            Why the model thinks so. ▲ = a fact working in his favor (lots of catches last year, new team, cheap
            price). ▼ = a fact working against him.
          </div>
          <div>
            <b>Numbers panel (inside a row)</b>
            Tap a row to open its full detail for the raw feature values and percentiles behind the call, plus the
            full market-drift chart — the detail underneath the plain-language verdict.
          </div>
        </div>
      </section>

      <p className="fine">
        Prices as of {meta.snapshot_date || "—"} · Models last trained {meta.model_generated_at || "—"}
      </p>
    </div>
  );
}

interface RowProps {
  r: BoardRow;
  index: number;
  useFull: boolean;
  maxP: number;
  rangeLo: number;
  rangeHi: number;
  open: boolean;
  onOpen: (trigger: HTMLButtonElement) => void;
}

function BoardRowItem({ r, index, useFull, maxP, rangeLo, rangeHi, open, onOpen }: RowProps) {
  const hv = huntScore(r);
  const p = hv == null ? 0 : Math.min(100, Math.round((100 * hv) / maxP));
  const shown = hv == null ? "—" : Math.round(hv * 100) + "%";
  const tr = tierInfo(r);
  const sub = [r.t, r.a ? `age ${Math.round(r.a)}` : null].filter(Boolean).join(" · ");
  const cost = costWords(r);
  const d = driftLabel(r.drift);
  const gap = gapCellText(r.vg);

  return (
    <li className={open ? "open" : undefined}>
      <button
        className="row"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={(e) => onOpen(e.currentTarget)}
      >
        <span className="rank">{index + 1}</span>
        <span className="who">
          <span className="nm">
            {r.n}
            {r.bo && (
              <span
                className="badge-broke"
                title="Posted a breakout season in 2025 -- this year's pick is not a first-time call."
              >
                broke out last yr
              </span>
            )}
          </span>
          <span className="sub">{sub}</span>
        </span>
        {useFull ? (
          <span className={`gap-cell ${gap.cls}`}>{gap.text}</span>
        ) : (
          <span className="cost">
            <span className="r">{cost.main}</span>
            <span className="rr">{cost.sub}</span>
          </span>
        )}
        {useFull ? (
          <span className="meter">
            <RangeBar r={r} lo={rangeLo} hi={rangeHi} />
          </span>
        ) : (
          <span className="meter">
            <span className="track">
              <span className="fill" style={{ width: `${p}%` }} />
            </span>
            <span className="val">{shown}</span>
          </span>
        )}
        <span className={`drift-cell ${d.dir}`}>
          {d.dir !== "flat" && <span className="arrow">{d.dir === "up" ? "▲" : "▼"}</span>}
          {d.text}
        </span>
        {useFull ? (
          r.aps ? (
            <span className="badge-priced">Priced starter</span>
          ) : (
            <span className={`tier ${tr.cls}`}>{tr.label}</span>
          )
        ) : (
          <span className={`tier ${tr.cls}`}>{tr.label}</span>
        )}
      </button>
    </li>
  );
}

