"""HTML shell for the BreakoutLab dashboard -- see src/dashboard/build.py for

how the embedded payload is assembled. Kept in its own module because it is
mostly a large formatted string, not data logic.
"""

from __future__ import annotations

PAGE_TEMPLATE = r"""<meta charset="utf-8">
<title>BreakoutLab Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {
    --bg: #F7F5EE; --surface: #EFECE1; --raised: #FFFFFF;
    --ink: #1D2721; --muted: #5D6B62; --faint: #8A968C;
    --line: #DDD8C9; --line-strong: #C8C2B0;
    --accent: #B47E13; --accent-ink: #8F6410;
    --meter-track: #E6E2D3;
    --pos: #3B6EA8; --neg: #B02E4A;
    --chip-bg: #ECE8DA; --focus: #8F6410;
    --banner-bg: #F0E9D6; --banner-line: #D9CBA4;
    --tier-elite-bg: #B47E13; --tier-elite-ink: #FFFFFF;
    --baseline-bar: #A9A296;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #131A16; --surface: #1A231E; --raised: #202B25;
      --ink: #EAE7DC; --muted: #93A398; --faint: #6C7A70;
      --line: #2B362F; --line-strong: #3A473F;
      --accent: #BE8A1E; --accent-ink: #E0A63C;
      --meter-track: #26312A;
      --pos: #4E7FC4; --neg: #C7405E;
      --chip-bg: #232E27; --focus: #E0A63C;
      --banner-bg: #221F14; --banner-line: #4A3E1D;
      --tier-elite-bg: #BE8A1E; --tier-elite-ink: #131A16;
      --baseline-bar: #4B564D;
    }
  }
  :root[data-theme="dark"] {
    --bg: #131A16; --surface: #1A231E; --raised: #202B25;
    --ink: #EAE7DC; --muted: #93A398; --faint: #6C7A70;
    --line: #2B362F; --line-strong: #3A473F;
    --accent: #BE8A1E; --accent-ink: #E0A63C;
    --meter-track: #26312A;
    --pos: #4E7FC4; --neg: #C7405E;
    --chip-bg: #232E27; --focus: #E0A63C;
    --banner-bg: #221F14; --banner-line: #4A3E1D;
    --tier-elite-bg: #BE8A1E; --tier-elite-ink: #131A16;
    --baseline-bar: #4B564D;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: 'Barlow', 'Helvetica Neue', Arial, sans-serif; font-size: 15px; line-height: 1.5; }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 0 20px 64px; }
  a { color: var(--accent-ink); }

  header.masthead { padding: 26px 0 14px; display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; border-bottom: 2px solid var(--ink); }
  .wordmark { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700; font-size: 36px; letter-spacing: .02em; line-height: 1; }
  .wordmark .lab { color: var(--accent-ink); }
  .season-chip { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 18px; border: 1.5px solid var(--line-strong); padding: 1px 10px 2px; border-radius: 3px; color: var(--muted); }
  .meta { margin-left: auto; font-size: 12.5px; color: var(--muted); text-align: right; line-height: 1.5; }

  nav.viewnav { display: flex; gap: 4px; margin: 14px 0 0; flex-wrap: wrap; }
  .vtab { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 15px; letter-spacing: .07em; text-transform: uppercase;
    padding: 7px 16px 8px; border: 1.5px solid var(--line-strong); border-radius: 4px; background: var(--raised); color: var(--muted); cursor: pointer; }
  .vtab:hover { color: var(--ink); }
  .vtab[aria-selected="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink); }
  .vtab:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }

  .howto { margin: 18px 0 0; display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .step { background: var(--surface); border: 1px solid var(--line); border-radius: 5px; padding: 12px 14px; }
  .step .k { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 15px; letter-spacing: .08em; text-transform: uppercase; color: var(--accent-ink); }
  .step p { margin: 5px 0 0; font-size: 13.5px; color: var(--muted); }
  .step p b { color: var(--ink); }
  @media (max-width: 700px) { .howto { grid-template-columns: 1fr; } }

  .caveat { margin: 12px 0 0; padding: 10px 14px; background: var(--banner-bg); border: 1px solid var(--banner-line); border-radius: 4px; font-size: 13px; color: var(--muted); }
  .caveat b { color: var(--ink); }

  nav.controls { position: sticky; top: 0; z-index: 5; background: var(--bg); padding: 14px 0 10px; border-bottom: 1px solid var(--line); display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .tabs { display: flex; gap: 4px; }
  .tab { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 18px; letter-spacing: .04em; padding: 5px 14px 6px; border: 1.5px solid transparent; border-radius: 3px; background: none; color: var(--muted); cursor: pointer; }
  .tab:hover { color: var(--ink); }
  .tab[aria-selected="true"] { background: var(--ink); color: var(--bg); }
  .tab .ct { font-size: 13px; opacity: .65; margin-left: 5px; font-family: 'IBM Plex Mono', monospace; }
  .controls input[type="search"] { flex: 1 1 150px; min-width: 130px; padding: 7px 11px; border: 1.5px solid var(--line-strong); border-radius: 3px; background: var(--raised); color: var(--ink); font: inherit; font-size: 14px; }
  .controls select { padding: 7px 8px; border: 1.5px solid var(--line-strong); border-radius: 3px; background: var(--raised); color: var(--ink); font: inherit; font-size: 14px; }
  .controls input:focus-visible, .controls select:focus-visible, .tab:focus-visible, .row:focus-visible, button:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }

  .rookie-note { display: none; margin: 12px 0 0; padding: 9px 14px; border-left: 3px solid var(--accent); background: var(--surface); font-size: 13px; color: var(--muted); }
  .rookie-note b { color: var(--ink); }

  .colhead { display: grid; grid-template-columns: 30px minmax(150px,1.2fr) 96px 1.1fr 84px 118px; gap: 12px; align-items: end;
    padding: 16px 10px 6px; font-size: 11px; letter-spacing: .09em; text-transform: uppercase; color: var(--faint); font-weight: 600; }
  .colhead .num { text-align: right; }

  ol.board { list-style: none; margin: 0; padding: 0; }
  .row { display: grid; grid-template-columns: 30px minmax(150px,1.2fr) 96px 1.1fr 84px 118px; gap: 12px; align-items: center;
    width: 100%; text-align: left; padding: 10px; border: 0; border-bottom: 1px solid var(--line); background: none; color: inherit; font: inherit; cursor: pointer; }
  .row:hover { background: var(--surface); }
  li.open .row { background: var(--surface); border-bottom-color: transparent; }
  .rank { font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--faint); text-align: right; }
  .who .nm { font-weight: 600; font-size: 15.5px; }
  .who .sub { display: block; font-size: 12px; color: var(--muted); margin-top: 1px; }
  .cost .r { font-weight: 600; font-size: 13.5px; }
  .cost .rr { display: block; font-size: 11.5px; color: var(--faint); }
  .meter { display: flex; align-items: center; gap: 9px; }
  .meter .track { flex: 1; height: 9px; background: var(--meter-track); border-radius: 4px; overflow: hidden; }
  .meter .fill { display: block; height: 100%; background: var(--accent); border-radius: 0 4px 4px 0; }
  .meter .val { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; font-size: 14.5px; font-weight: 600; width: 42px; text-align: right; }
  .drift-cell { font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; text-align: right; white-space: nowrap; }
  .drift-cell .arrow { font-weight: 700; margin-right: 2px; }
  .drift-cell.up { color: var(--pos); } .drift-cell.dn { color: var(--neg); } .drift-cell.flat { color: var(--faint); }
  .tier { justify-self: start; font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 14px; letter-spacing: .05em; text-transform: uppercase; padding: 2px 9px 3px; border-radius: 3px; white-space: nowrap; }
  .tier.t4 { background: var(--tier-elite-bg); color: var(--tier-elite-ink); }
  .tier.t3 { border: 1.5px solid var(--accent); color: var(--accent-ink); }
  .tier.t2 { border: 1.5px solid var(--line-strong); color: var(--muted); }
  .tier.t1 { color: var(--faint); }

  .detail { display: none; padding: 12px 10px 22px 52px; background: var(--surface); border-bottom: 1px solid var(--line); font-size: 13.5px; color: var(--muted); }
  li.open .detail { display: block; }
  .detail .say { max-width: 68ch; color: var(--ink); font-size: 14px; }
  .detail .why { margin-top: 10px; max-width: 68ch; }
  .detail .why .lbl, .detail .also .lbl, .detail .profile .lbl, .detail .spark .lbl { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--faint); font-weight: 600; display: block; margin-bottom: 6px; }
  .drv { display: inline-block; font-size: 12px; padding: 2px 9px; border-radius: 10px; background: var(--chip-bg); color: var(--muted); margin: 0 4px 4px 0; }
  .drv .dir { font-weight: 700; margin-right: 3px; }
  .drv .dir.up { color: var(--pos); } .drv .dir.dn { color: var(--neg); }
  .detail .also { margin-top: 10px; max-width: 68ch; font-size: 13px; }
  .detail-grid { display: grid; grid-template-columns: 1.15fr .85fr; gap: 22px; margin-top: 16px; }
  @media (max-width: 780px) { .detail-grid { grid-template-columns: 1fr; } }

  .profile-row { display: grid; grid-template-columns: 148px 1fr 52px; align-items: center; gap: 9px; margin-bottom: 6px; }
  .profile-row .flbl { font-size: 12.5px; color: var(--ink); }
  .profile-row .ftrack { height: 8px; background: var(--meter-track); border-radius: 4px; position: relative; overflow: hidden; }
  .profile-row .ffill { display: block; height: 100%; background: var(--accent); border-radius: 4px; }
  .profile-row .fval { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: var(--faint); text-align: right; }

  .spark-wrap svg { display: block; width: 100%; height: 64px; }
  .spark-line { fill: none; stroke: var(--accent); stroke-width: 2; }
  .spark-dot { fill: var(--accent); }
  .spark-note { font-size: 12px; color: var(--muted); margin-top: 4px; }

  .empty { padding: 40px 10px; text-align: center; color: var(--faint); }

  section.reading { margin-top: 44px; border-top: 2px solid var(--ink); padding-top: 18px; }
  section.reading h2 { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 24px; letter-spacing: .02em; margin: 0 0 10px; text-wrap: balance; }
  .glossary { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px 24px; margin: 0 0 22px; }
  .glossary div { font-size: 13.5px; color: var(--muted); max-width: 42ch; }
  .glossary b { color: var(--ink); display: block; font-size: 14px; }
  .vtable { overflow-x: auto; }
  .vtable table { border-collapse: collapse; font-size: 13.5px; min-width: 460px; width: 100%; }
  .vtable th { text-align: left; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--faint); font-weight: 600; padding: 5px 18px 5px 0; border-bottom: 1px solid var(--line-strong); }
  .vtable td { padding: 6px 18px 6px 0; border-bottom: 1px solid var(--line); }
  .vtable td.mono { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; font-size: 13px; }
  .fine { margin-top: 14px; font-size: 12.5px; color: var(--muted); max-width: 68ch; line-height: 1.6; }

  /* ---- Views (positions / trust / method) ---- */
  .view { margin-top: 20px; }
  .view h1.vtitle { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 26px; margin: 4px 0 2px; }
  .view p.vsub { font-size: 13.5px; color: var(--muted); margin: 0 0 20px; max-width: 68ch; }

  .poscard { background: var(--raised); border: 1px solid var(--line); border-radius: 6px; padding: 16px 18px; margin-bottom: 16px; }
  .poscard .hd { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
  .poscard .hd .pname { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 21px; }
  .poscard .hd .stat { font-size: 12.5px; color: var(--muted); }
  .poscard .hd .stat b { color: var(--ink); font-family: 'IBM Plex Mono', monospace; }
  .poscard .top5 { display: flex; flex-wrap: wrap; gap: 6px 10px; margin: 10px 0 14px; }
  .poscard .top5 span.name { font-size: 13px; color: var(--ink); }
  .poscard .top5 span.pv { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--accent-ink); font-weight: 600; margin-left: 3px; }
  .poscard .top5 .p5item { background: var(--chip-bg); border-radius: 10px; padding: 3px 10px; }
  .hist { display: flex; align-items: flex-end; gap: 2px; height: 64px; margin-top: 4px; }
  .hist .bar { flex: 1; background: var(--accent); border-radius: 1px 1px 0 0; min-height: 1px; position: relative; }
  .hist .bar[data-n="0"] { background: var(--meter-track); }
  .hist-labels { display: flex; justify-content: space-between; font-size: 10.5px; color: var(--faint); margin-top: 3px; font-family: 'IBM Plex Mono', monospace; }

  .barpair-chart { margin: 6px 0 18px; }
  .barpair-row { display: grid; grid-template-columns: 44px 1fr 56px; align-items: center; gap: 8px; margin-bottom: 5px; }
  .barpair-row .lbl { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 13px; color: var(--muted); }
  .barpair-row .track { height: 12px; background: var(--meter-track); border-radius: 2px; position: relative; }
  .barpair-row .fill { display: block; height: 100%; border-radius: 2px; }
  .barpair-row .fill.model { background: var(--accent); }
  .barpair-row .fill.baseline { background: var(--baseline-bar); }
  .barpair-row .val { font-family: 'IBM Plex Mono', monospace; font-size: 12px; text-align: right; }
  .legend { display: flex; gap: 16px; font-size: 12px; color: var(--muted); margin: 4px 0 14px; }
  .legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
  .legend .sw.model { background: var(--accent); } .legend .sw.baseline { background: var(--baseline-bar); }

  .toggle-btn { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 12.5px; letter-spacing: .05em; text-transform: uppercase;
    background: var(--raised); border: 1.5px solid var(--line-strong); border-radius: 3px; padding: 4px 10px; color: var(--muted); cursor: pointer; margin-bottom: 10px; }
  .toggle-btn:hover { color: var(--ink); }

  .namedlist { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 10px 0 20px; }
  @media (max-width: 780px) { .namedlist { grid-template-columns: 1fr; } }
  .namedlist .yr h3 { font-family: 'Barlow Condensed', sans-serif; font-size: 16px; margin: 0 0 6px; }
  .hitrow { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; padding: 3px 0; border-bottom: 1px solid var(--line); }
  .hitrow .hm { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--faint); }
  .hitrow.hit { color: var(--pos); font-weight: 600; }
  .hitrow.hit .chk { margin-right: 5px; }

  .method h2 { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 19px; margin: 26px 0 6px; }
  .method h2:first-child { margin-top: 0; }
  .method p { font-size: 14px; color: var(--muted); max-width: 70ch; line-height: 1.65; margin: 0 0 10px; }
  .method p b { color: var(--ink); }
  .method ul { margin: 0 0 10px; padding-left: 20px; color: var(--muted); font-size: 14px; max-width: 70ch; line-height: 1.6; }

  @media (max-width: 700px) {
    .colhead { display: none; }
    .row { grid-template-columns: 1fr auto; grid-template-areas: "who tier" "meter meter"; row-gap: 7px; }
    .rank, .cost, .drift-cell { display: none; }
    .who { grid-area: who; } .meter { grid-area: meter; } .tier { grid-area: tier; align-self: start; }
    .detail { padding-left: 10px; }
  }
  @media (prefers-reduced-motion: no-preference) { .row { transition: background .12s ease; } }
</style>

<div class="wrap">
  <header class="masthead">
    <div class="wordmark">BREAKOUT<span class="lab">LAB</span></div>
    <div class="season-chip">2026 DASHBOARD</div>
    <div class="meta" id="mastMeta"></div>
  </header>
  <nav class="viewnav" id="viewnav" role="tablist" aria-label="Dashboard views"></nav>

  <div id="viewBoard" class="view"></div>
  <div id="viewPositions" class="view" hidden></div>
  <div id="viewTrust" class="view" hidden></div>
  <div id="viewMethod" class="view" hidden></div>
</div>

<script type="application/json" id="data">__PAYLOAD_JSON__</script>
<script>
(function () {
  "use strict";
  const FEAT_FALLBACK = f => f.replace(/_/g, " ");
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const BOARD = DATA.board, FEATURES = DATA.features, ECR = DATA.ecr, POS_VIEW = DATA.positions, TRUST = DATA.trust, META = DATA.meta, LCFG = DATA.labels_config;

  const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pct = v => v == null ? "—" : Math.round(v * 100) + "%";
  const POS_ORDER = ["QB", "RB", "WR", "TE"];
  const POS_NAME = {QB:"Quarterback", RB:"Running back", WR:"Wide receiver", TE:"Tight end"};

  document.getElementById("mastMeta").innerHTML =
    "Prices as of " + esc(META.snapshot_date || "—") + "<br>Models last trained " + esc(META.model_generated_at || "—");

  // ---------------------------------------------------------------- nav
  const VIEWS = [
    {id: "board", label: "Board"},
    {id: "positions", label: "Positions"},
    {id: "trust", label: "Trust"},
    {id: "method", label: "Method"},
  ];
  let currentView = "board";
  const navEl = document.getElementById("viewnav");
  VIEWS.forEach(v => {
    const b = document.createElement("button");
    b.className = "vtab"; b.setAttribute("role", "tab"); b.id = "vtab-" + v.id;
    b.textContent = v.label;
    b.addEventListener("click", () => switchView(v.id));
    navEl.appendChild(b);
  });
  function switchView(id) {
    currentView = id;
    VIEWS.forEach(v => {
      document.getElementById("vtab-" + v.id).setAttribute("aria-selected", String(v.id === id));
      document.getElementById("view" + v.id[0].toUpperCase() + v.id.slice(1)).hidden = v.id !== id;
    });
    if (id === "positions" && !positionsRendered) renderPositions();
    if (id === "trust" && !trustRendered) renderTrust();
    if (id === "method" && !methodRendered) renderMethod();
  }
  switchView("board");

  // =========================================================== BOARD VIEW
  // v1.7: tiers are base-rate MULTIPLES ("N times a normal late pick's odds at this
  // position"), computed server-side (src.inference.board_2026.attach_renormalized_probability
  // -- base_rate = historical mean per-season breakouts / players scored at that position
  // this year) and shipped on every board row as r.tier/r.brm -- not recomputed client-side
  // against a fixed absolute cutoff the way v1's TIERS array did, since a fixed cutoff isn't
  // meaningful across positions with very different base rates (QB's is much higher than
  // TE's, historically). TIER_CLASS is just the label->CSS-class lookup.
  const TIER_CLASS = { "Elite target": "t4", "Strong swing": "t3", "Worth a flier": "t2", "Long shot": "t1" };
  function tier(r) {
    const label = r.tier || "Long shot";
    return { cls: TIER_CLASS[label] || "t1", label: label };
  }
  const vets = BOARD.filter(r => !r.rk), rooks = BOARD.filter(r => r.rk);
  const TABS = [
    {id:"QB", label:"QB", rows: vets.filter(r=>r.p==="QB")},
    {id:"RB", label:"RB", rows: vets.filter(r=>r.p==="RB")},
    {id:"WR", label:"WR", rows: vets.filter(r=>r.p==="WR")},
    {id:"TE", label:"TE", rows: vets.filter(r=>r.p==="TE")},
    {id:"RK", label:"ROOKIES", rows: rooks},
  ];
  let cur = "WR", query = "", sortKey = "pr", openKey = null;

  const boardRoot = document.getElementById("viewBoard");
  boardRoot.innerHTML = `
    <div class="howto">
      <div class="step"><span class="k">1 &middot; Draft normally early</span>
        <p>Rounds 1&ndash;6, take your studs and <b>ignore this page</b>. Everyone here is, by definition, available later.</p></div>
      <div class="step"><span class="k">2 &middot; From round 7, open this</span>
        <p>When your pick comes and the options all look the same, <b>take the highest percentage</b> on this board. That's the whole system.</p></div>
      <div class="step"><span class="k">3 &middot; Check the news first</span>
        <p>This board <b>doesn't know about injuries</b>, suspensions, or holdouts. Thirty seconds of news-checking beats the math.</p></div>
    </div>
    <p class="caveat"><b>What the percentage means:</b> the chance this cheap player finishes the season as a true weekly starter anyway (top-12 QB, top-15 RB, top-18 WR, top-8 TE). Each position's percentages are normalized so the total across everyone scored at that position matches how many breakouts actually happen there in a typical season &mdash; so a 25% WR and a 25% QB reflect the same kind of real base-rate comparison, not two calibration curves that happen to output the same number. The tier label below is a multiple of a random late pick's odds at that spot (see Verdict in "How to read this board") &mdash; even an Elite target won't hit every time.</p>
    <nav class="controls" aria-label="Board controls">
      <div class="tabs" role="tablist" id="tabs"></div>
      <input type="search" id="q" placeholder="Search player or team&hellip;" aria-label="Search player or team">
      <select id="sort" aria-label="Sort by">
        <option value="pr">Best bets first</option>
        <option value="e">Cheapest first</option>
        <option value="a">Youngest first</option>
        <option value="drift">Rising fastest</option>
      </select>
    </nav>
    <p class="rookie-note" id="rookieNote"><b>Rookie section &mdash; bigger guesses.</b> Rookies have no NFL history, so a simpler model guesses from draft position and landing spot. Use these only to break ties on your last couple of picks.</p>
    <div class="colhead" id="colhead">
      <div class="num">#</div><div>Player</div><div>Draft cost</div><div>Chance he breaks out</div><div>Market</div><div>Verdict</div>
    </div>
    <ol class="board" id="board"></ol>
    <div class="empty" id="empty" hidden>No players match.</div>
    <section class="reading">
      <h2>How to read this board</h2>
      <div class="glossary">
        <div><b>Draft cost &mdash; "goes ~WR39"</b>Where drafters are currently taking him: the 39th receiver picked, i.e. a bench-round price. Lower number = more expensive.</div>
        <div><b>The percentage</b>The model's calibrated odds he finishes as a weekly starter despite that cheap price, rescaled so each position's percentages add up to how many breakouts that position actually produces in a season. It's the only number you need on draft night.</div>
        <div><b>Verdict</b>That percentage as a multiple of a normal late pick's odds at his position: <b>Elite target</b> (8x or more), <b>Strong swing</b> (4&ndash;8x), <b>Worth a flier</b> (2&ndash;4x), <b>Long shot</b> (under 2x).</div>
        <div><b>Market ▲ / ▼</b>How his preseason ranking has moved across the summer's weekly consensus snapshots. ▲ = the market is warming up on him since June; ▼ = it's cooling.</div>
        <div><b>▲ and ▼ inside a row (tap to open)</b>Why the model thinks so. ▲ = a fact working in his favor (lots of catches last year, new team, cheap price). ▼ = a fact working against him.</div>
        <div><b>Numbers panel (inside a row)</b>Tap "Show the numbers" for the raw feature values and percentiles behind the call, plus the full market-drift chart &mdash; the detail underneath the plain-language verdict.</div>
      </div>
    </section>
  `;

  const tabsEl = document.getElementById("tabs");
  TABS.forEach(t => {
    const b = document.createElement("button");
    b.className = "tab"; b.setAttribute("role", "tab"); b.id = "tab-" + t.id;
    b.innerHTML = esc(t.label) + '<span class="ct">' + t.rows.length + "</span>";
    b.addEventListener("click", () => { cur = t.id; openKey = null; renderBoard(); });
    tabsEl.appendChild(b);
  });
  document.getElementById("q").addEventListener("input", e => { query = e.target.value.trim().toLowerCase(); renderBoard(); });
  document.getElementById("sort").addEventListener("change", e => { sortKey = e.target.value; renderBoard(); });

  function driverChips(s, n) {
    return (s || "").split(", ").filter(Boolean).slice(0, n).map(tok => {
      const i = tok.lastIndexOf(":");
      const f = i > 0 ? tok.slice(0, i).trim() : tok.trim();
      const dir = i > 0 ? tok.slice(i + 1).trim() : "";
      const up = dir.startsWith("+");
      const arrow = dir ? '<span class="dir ' + (up ? "up" : "dn") + '" aria-hidden="true">' + (up ? "▲" : "▼") + "</span>" : "";
      const label = FEAT_LABEL[f] || FEAT_FALLBACK(f);
      return '<span class="drv">' + arrow + esc(label) + "</span>";
    }).join("");
  }
  // FEAT_LABEL is filled in from the first feature-profile lookup seen per
  // code (the payload's `features` map already carries plain labels; this
  // just gives driverChips() a code->label index without duplicating the
  // FEAT translation table client-side).
  const FEAT_LABEL = {};
  Object.values(FEATURES).forEach(list => (list || []).forEach(f => { FEAT_LABEL[f.feat] = f.label; }));

  function costCell(r) {
    if (r.rk) return '<span class="r">' + (r.dr ? "Round " + r.dr + " pick" : "—") + '</span><span class="rr">NFL draft</span>';
    if (r.e == null) return '<span class="r">Undrafted</span><span class="rr">free in most leagues</span>';
    const note = r.e <= 12 ? "goes early" : r.e <= 30 ? "mid-round price" : r.e <= 55 ? "late-round price" : "basically free";
    return '<span class="r">~' + r.p + r.e + '</span><span class="rr">' + note + "</span>";
  }
  function driftCell(r) {
    const d = r.drift;
    if (!d || d.n < 2) return '<span class="drift-cell flat">&mdash;</span>';
    const delta = d.drift;
    if (Math.abs(delta) < 2) return '<span class="drift-cell flat">flat</span>';
    const up = delta > 0;
    return '<span class="drift-cell ' + (up ? "up" : "dn") + '"><span class="arrow">' + (up ? "▲" : "▼") + '</span>' + Math.abs(delta) + " spot" + (Math.abs(delta) === 1 ? "" : "s") + "</span>";
  }
  function driftSentence(r) {
    const d = r.drift;
    if (!d || d.n < 2) return null;
    const delta = d.drift;
    if (Math.abs(delta) < 2) return "The market hasn't moved much on him since early June.";
    const dir = delta > 0 ? "risen" : "fallen";
    return "The market's own consensus rank has " + dir + " ~" + Math.abs(delta) + " spot" + (Math.abs(delta)===1?"":"s") + " at " + r.p + " since early June (" + d.n + " weekly snapshots)" + (delta > 0 ? " &mdash; other drafters are catching on." : " &mdash; other drafters are cooling on him too.") + "";
  }

  function featureProfileHTML(key) {
    const feats = FEATURES[key];
    if (!feats || !feats.length) return "";
    const rows = feats.map(f => {
      const w = Math.max(2, Math.round(f.pctl));
      const valStr = Math.abs(f.value) >= 100 ? Math.round(f.value) : (Math.round(f.value * 100) / 100);
      return '<div class="profile-row"><span class="flbl">' + esc(f.label) + '</span>' +
        '<span class="ftrack"><span class="ffill" style="width:' + w + '%"></span></span>' +
        '<span class="fval">' + esc(String(valStr)) + '</span></div>';
    }).join("");
    return '<div class="profile"><span class="lbl">His feature profile (percentile vs. every ' + esc(feats.length ? "" : "") + 'player at his position this year)</span>' + rows + "</div>";
  }

  function sparklineHTML(key) {
    const d = ECR[key];
    if (!d || d.n < 2) return "";
    const w = 320, h = 56, pad = 6;
    const ranks = d.ranks;
    const lo = Math.min(...ranks), hi = Math.max(...ranks);
    const span = Math.max(1, hi - lo);
    const x = i => pad + (i / (ranks.length - 1)) * (w - 2 * pad);
    // inverted y: lower (better) rank draws higher on the chart
    const y = r => pad + ((r - lo) / span) * (h - 2 * pad);
    const pts = ranks.map((r, i) => x(i) + "," + y(r)).join(" ");
    const lastX = x(ranks.length - 1), lastY = y(ranks[ranks.length - 1]);
    const titles = d.dates.map((dt, i) => dt + ": rank " + ranks[i]).join(" | ");
    const sentence = driftSentence({drift: d, p: ""});
    return '<div class="spark"><span class="lbl">Market drift &mdash; positional rank over the summer (up = rising)</span>' +
      '<div class="spark-wrap" title="' + esc(titles) + '">' +
      '<svg viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" role="img" aria-label="Market rank trend">' +
      '<polyline class="spark-line" points="' + pts + '"></polyline>' +
      '<circle class="spark-dot" cx="' + lastX + '" cy="' + lastY + '" r="4.5"></circle>' +
      "</svg></div>" +
      (sentence ? '<div class="spark-note">' + sentence + "</div>" : "") + "</div>";
  }

  function renderBoard() {
    TABS.forEach(t => document.getElementById("tab-" + t.id).setAttribute("aria-selected", String(t.id === cur)));
    const tab = TABS.find(t => t.id === cur);
    document.getElementById("rookieNote").style.display = cur === "RK" ? "block" : "none";
    let rows = tab.rows.filter(r => !query || r.n.toLowerCase().includes(query) || (r.t || "").toLowerCase().includes(query));
    let dir = (sortKey === "e" || sortKey === "a") ? 1 : -1;
    rows = rows.slice().sort((x, y) => {
      if (sortKey === "drift") {
        const a = x.drift ? x.drift.drift : null, b = y.drift ? y.drift.drift : null;
        if (a == null && b == null) return (y.pr ?? 0) - (x.pr ?? 0);
        if (a == null) return 1; if (b == null) return -1;
        return b - a;
      }
      const a = x[sortKey], b = y[sortKey];
      if (a == null && b == null) return (y.pr ?? 0) - (x.pr ?? 0);
      if (a == null) return 1; if (b == null) return -1;
      return dir * (a - b) || (y.rs ?? 0) - (x.rs ?? 0);
    });
    const maxP = cur === "RK" ? 1 : 0.7;
    const board = document.getElementById("board");
    board.innerHTML = rows.map((r, i) => {
      const key = r.key;
      const p = r.pr == null ? 0 : Math.min(100, Math.round(100 * r.pr / maxP));
      const shown = r.pr == null ? "—" : Math.round(r.pr * 100) + "%";
      const tr = tier(r);
      const sub = [r.t, r.a ? "age " + Math.round(r.a) : null].filter(Boolean).join(" &middot; ");
      const profileHTML = featureProfileHTML(key);
      const sparkHTML = sparklineHTML(key);
      return '<li' + (openKey === key ? ' class="open"' : "") + '>' +
        '<button class="row" data-k="' + esc(key) + '" aria-expanded="' + (openKey === key) + '">' +
          '<span class="rank">' + (i + 1) + "</span>" +
          '<span class="who"><span class="nm">' + esc(r.n) + '</span><span class="sub">' + sub + "</span></span>" +
          '<span class="cost">' + costCell(r) + "</span>" +
          '<span class="meter"><span class="track"><span class="fill" style="width:' + p + '%"></span></span>' +
            '<span class="val">' + shown + "</span></span>" +
          driftCell(r) +
          '<span class="tier ' + tr.cls + '">' + tr.label + "</span>" +
        "</button>" +
        '<div class="detail">' +
          '<div class="say">' + (r.r ? esc(r.r) : esc(r.n) + " — no notes recorded.") + "</div>" +
          (r.s ? '<div class="why"><span class="lbl">What&rsquo;s driving the call</span>' + driverChips(r.s, 9) + "</div>" : "") +
          (!r.rk && r.d != null ? '<div class="also"><span class="lbl">Second opinion</span>A separate model predicts his finish vs his price: ' +
            (r.d > 2 ? "beats his price by ~" + Math.round(r.d) + " spots — the two models agree." :
             r.d < -2 ? "finishes ~" + Math.abs(Math.round(r.d)) + " spots below the hype — the two models disagree, so treat this pick with extra caution." :
             "roughly matches his price.") + "</div>" : "") +
          (profileHTML || sparkHTML ? '<div class="detail-grid">' +
            (profileHTML ? '<div>' + profileHTML + '</div>' : '<div></div>') +
            (sparkHTML ? '<div>' + sparkHTML + '</div>' : '<div></div>') +
            '</div>' : "") +
        "</div></li>";
    }).join("");
    document.getElementById("empty").hidden = rows.length > 0;
    board.querySelectorAll(".row").forEach(b => b.addEventListener("click", () => {
      openKey = openKey === b.dataset.k ? null : b.dataset.k; renderBoard();
    }));
  }
  renderBoard();

  // ======================================================== POSITIONS VIEW
  let positionsRendered = false;
  function renderPositions() {
    positionsRendered = true;
    const root = document.getElementById("viewPositions");
    let html = '<h1 class="vtitle">Positions</h1><p class="vsub">Most players at every position are long shots &mdash; the model just tells you which few are worth the bench slot. The tail on the right is where you shop.</p>';
    POS_ORDER.forEach(pos => {
      const pv = POS_VIEW[pos.toLowerCase()];
      if (!pv) return;
      const base = pv.base_rate != null ? (pv.base_rate * 100).toFixed(1) + "%" : "—";
      const top5 = pv.top5.map(t => '<span class="p5item"><span class="name">' + esc(t.n) + '</span><span class="pv">' + pct(t.pr) + '</span></span>').join("");
      const maxBin = Math.max(1, ...pv.histogram);
      const bins = pv.histogram.map((n, i) => '<div class="bar" data-n="' + n + '" style="height:' + Math.max(2, Math.round(100 * n / maxBin)) + '%" title="' + (i*10) + '–' + (i*10+10) + '%: ' + n + ' player' + (n===1?"":"s") + '"></div>').join("");
      html += '<div class="poscard">' +
        '<div class="hd"><span class="pname">' + esc(POS_NAME[pos]) + '</span>' +
        '<span class="stat">' + pv.count + ' players scored</span>' +
        '<span class="stat">historical breakout base rate <b>' + base + '</b></span></div>' +
        '<div class="top5"><span class="lbl" style="display:block;width:100%;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);font-weight:600;margin-bottom:2px;">Top 5 right now</span>' + top5 + '</div>' +
        '<span class="lbl" style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);font-weight:600;">How conviction is distributed (share of players by chance-of-breakout bucket)</span>' +
        '<div class="hist">' + bins + '</div>' +
        '<div class="hist-labels"><span>0%</span><span>50%</span><span>100%</span></div>' +
        '</div>';
    });
    root.innerHTML = html;
  }

  // ============================================================ TRUST VIEW
  let trustRendered = false;
  function renderTrust() {
    trustRendered = true;
    const root = document.getElementById("viewTrust");
    let html = '<h1 class="vtitle">Trust</h1><p class="vsub">The model finds breakouts the market misses &mdash; tested on real years it never saw during tuning, with no do-overs.</p>';

    html += '<div class="vtable"><table><thead><tr><th>Position</th><th>Its real track record</th></tr></thead><tbody>' +
      '<tr><td><b>QB</b></td><td>Best position. Its 2025 cheap-QB picks included Stafford, Goff and Lawrence — all hit.</td></tr>' +
      '<tr><td><b>RB</b></td><td>Good. Called Chase Brown (\'24) and Etienne (\'25) before they were startable.</td></tr>' +
      '<tr><td><b>WR</b></td><td>Decent. Called Olave and Rice in \'25; missed in \'24 (only one WR broke out league-wide that year).</td></tr>' +
      '<tr><td><b>TE</b></td><td>Thinnest. Called Kyle Pitts in \'25; treat TE calls with extra doubt.</td></tr>' +
      '</tbody></table></div>';

    html += '<h2 style="font-family:\'Barlow Condensed\',sans-serif;font-size:19px;margin:28px 0 4px;">The model vs. just using the market’s own rank</h2>' +
      '<p class="vsub" style="margin-bottom:8px;">Each pair: how well the model’s calibrated score finds real breakouts (PR-AUC, higher = better) on the two holdout years it never trained on, vs. the best of three simple non-model baselines (market rank, last year’s points, age-adjusted market rank).</p>' +
      '<div class="legend"><span><span class="sw model"></span>Model</span><span><span class="sw baseline"></span>Best baseline</span></div>';

    const chartRows = POS_ORDER.map(pos => {
      const t = TRUST.positions[pos.toLowerCase()];
      if (!t) return "";
      const m = t.holdout_pr_auc ?? 0, b = t.best_baseline_pr_auc ?? 0;
      const maxV = Math.max(m, b, 0.05);
      return {pos, t, m, b, maxV};
    }).filter(Boolean);
    const globalMax = Math.max(0.05, ...chartRows.map(r => Math.max(r.m, r.b)));
    html += '<div class="barpair-chart" id="barpairChart">' + chartRows.map(r =>
      '<div class="barpair-row"><span class="lbl">' + r.pos + '</span>' +
      '<span class="track"><span class="fill model" style="width:' + Math.round(100 * r.m / globalMax) + '%"></span></span>' +
      '<span class="val">' + r.m.toFixed(3) + '</span></div>' +
      '<div class="barpair-row"><span class="lbl"></span>' +
      '<span class="track"><span class="fill baseline" style="width:' + Math.round(100 * r.b / globalMax) + '%" title="' + esc(r.t.best_baseline_name || "") + '"></span></span>' +
      '<span class="val">' + r.b.toFixed(3) + '</span></div>'
    ).join("") + '</div>' +
    '<button class="toggle-btn" id="barpairToggle">Show as table</button>' +
    '<div class="vtable" id="barpairTable" hidden><table><thead><tr><th>Position</th><th>Model holdout PR-AUC</th><th>Best baseline</th><th>Baseline PR-AUC</th></tr></thead><tbody>' +
    chartRows.map(r => '<tr><td>' + r.pos + '</td><td class="mono">' + r.m.toFixed(3) + '</td><td>' + esc(r.t.best_baseline_name || "—") + '</td><td class="mono">' + r.b.toFixed(3) + '</td></tr>').join("") +
    '</tbody></table></div>';

    html += '<h2 style="font-family:\'Barlow Condensed\',sans-serif;font-size:19px;margin:28px 0 4px;">Named holdout top-10 lists, 2024 &amp; 2025</h2>' +
      '<p class="vsub" style="margin-bottom:8px;">The model’s actual top-10 cheap-price picks each holdout year, and whether each one really broke out. ✓ = hit.</p>';
    POS_ORDER.forEach(pos => {
      const nl = TRUST.named_lists[pos.toLowerCase()];
      if (!nl) return;
      html += '<h3 style="font-family:\'Barlow Condensed\',sans-serif;font-size:15px;margin:16px 0 4px;">' + pos + '</h3><div class="namedlist">';
      ["2024", "2025"].forEach(yr => {
        const rows = nl[yr] || [];
        html += '<div class="yr"><h3>' + yr + '</h3>' + rows.map(p =>
          '<div class="hitrow' + (p.hit ? " hit" : "") + '"><span>' + (p.hit ? '<span class="chk">✓</span>' : "") + esc(p.player) + '</span><span class="hm">' + pct(p.prob) + '</span></div>'
        ).join("") + '</div>';
      });
      html += '</div>';
    });

    html += '<p class="fine">Tested honestly: the model was trained only on past seasons, then graded on 2024–2025 games it had never seen, with no do-overs. Its ten best ideas per year produce one to three hits &mdash; which is exactly what winning a league on the margins looks like. Vegas betting lines were tested as an ingredient twice (see Method) and added nothing measurable; the market’s own draft prices already contain that information.</p>';

    root.innerHTML = html;
    const toggleBtn = document.getElementById("barpairToggle");
    const chartEl = document.getElementById("barpairChart");
    const legendEl = root.querySelector(".legend");
    const tableEl = document.getElementById("barpairTable");
    toggleBtn.addEventListener("click", () => {
      const showingTable = !tableEl.hidden;
      tableEl.hidden = showingTable;
      chartEl.hidden = !showingTable;
      legendEl.hidden = !showingTable;
      toggleBtn.textContent = showingTable ? "Show as table" : "Show as chart";
    });
  }

  // =========================================================== METHOD VIEW
  let methodRendered = false;
  function renderMethod() {
    methodRendered = true;
    const root = document.getElementById("viewMethod");
    const th = LCFG && LCFG.thresholds ? LCFG.thresholds : {};
    const thLine = POS_ORDER.map(p => th[p] ? (p + " top-" + th[p].finish_top + " (must also have been ranked " + p + (th[p].adp_worse_than) + " or worse preseason)") : null).filter(Boolean).join("; ");
    root.innerHTML = '<h1 class="vtitle">Method</h1><p class="vsub">The plain-language version of how this board is built &mdash; what it knows, what it doesn’t, and why the percentage means what it means.</p>' +
      '<div class="method">' +
      '<h2>What "breakout" means</h2>' +
      '<p>A breakout is a player who (a) finished the season inside a real starter’s range at his position, and (b) wasn’t expected to going in — both conditions have to be true. The exact bars: ' + esc(thLine) + '. A player good enough to finish there but who was already expected to (a first-round pick, say) doesn’t count — this board is about the market’s misses, not just good players.</p>' +
      '<h2>What the percentage means</h2>' +
      '<p>It starts as a <b>calibrated</b> probability (Platt scaling on the seed-ensemble score, v1.7) — among every player the model has ever said "25%" about, roughly 1 in 4 actually broke out. Then it gets <b>renormalized per position</b>: nothing about a calibration curve forces a position\'s probabilities to sum to how many breakouts that position actually produces in a season, so each position is rescaled so they do (see the README\'s v1.7 section). The league-wide base rate for a random cheap/late player is about 3–6% (QB highest, WR/TE thinnest — see the Positions view for the real number at each spot); a tier label is that player\'s displayed percentage as a multiple of his own position\'s base rate this year, not a fixed cutoff.</p>' +
      '<h2>What it trains on</h2>' +
      '<p>Only information that existed <b>before</b> the season started: the player’s own prior-season usage and efficiency, his team’s situation (new coach, new offensive coordinator, teammates who left and freed up targets/carries), his draft pedigree, and the market’s own preseason consensus rank. Every feature is built with a strict N-1 shift — nothing from the season being predicted ever leaks into the inputs used to predict it. That is the one rule the whole pipeline is built around.</p>' +
      '<h2>What it does NOT know</h2>' +
      '<ul>' +
      '<li><b>Injuries, holdouts, suspensions.</b> The manual availability screen this board reads from is empty by design — nobody can responsibly guess next month’s injury report from here. Check the news yourself before you draft anyone on this page.</li>' +
      '<li><b>Anything after the snapshot date</b> shown in the header (the last weekly market-consensus pull this build used).</li>' +
      '<li><b>This year’s actual games.</b> Obviously — none have been played yet.</li>' +
      '</ul>' +
      '<h2>Vegas lines: tested, not used</h2>' +
      '<p>Team-level Vegas win totals and implied scoring were tried as model inputs, twice, across all four positions. Neither pass improved holdout accuracy enough to justify keeping — the market’s own preseason draft price already contains most of what the betting lines would have added. The board still shows Vegas implied points where available, purely as a manual sanity-check column, never as a model input.</p>' +
      '<h2>The rookie section is a different, weaker model</h2>' +
      '<p>Rookies have no NFL history to build the real feature set from, so they’re scored by a much simpler heuristic (draft position, landing-spot opportunity) on its own probability scale — never comparable number-for-number to the veteran percentages above it. Treat rookie calls as tie-breakers on your last couple of picks, not as strong signals.</p>' +
      '</div>';
  }
})();
</script>
"""


def render_html(payload_json: str, meta: dict) -> str:
    return PAGE_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)
