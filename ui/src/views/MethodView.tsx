import type { Payload } from "../types";
import { POS_ORDER } from "../lib/format";

interface Props {
  payload: Payload;
}

/** Ported from template.py's renderMethod(). */
export function MethodView({ payload }: Props) {
  const th = payload.labels_config?.thresholds || {};
  const thLine = POS_ORDER.map((p) =>
    th[p] ? `${p} top-${th[p].finish_top} (must also have been ranked ${p}${th[p].adp_worse_than} or worse preseason)` : null,
  )
    .filter(Boolean)
    .join("; ");

  return (
    <div className="view">
      <h1 className="vtitle">Method</h1>
      <p className="vsub">
        The plain-language version of how this board is built — what it knows, what it doesn't, and why the
        percentage means what it means.
      </p>
      <div className="method">
        <h2>What "breakout" means</h2>
        <p>
          A breakout is a player who (a) finished the season inside a real starter's range at his position, and (b)
          wasn't expected to going in — both conditions have to be true. The exact bars: {thLine}. A player good
          enough to finish there but who was already expected to (a first-round pick, say) doesn't count — this
          board is about the market's misses, not just good players.
        </p>
        <h2>What the percentage means</h2>
        <p>
          It starts as a <b>calibrated</b> probability (Platt scaling on the seed-ensemble score, v1.7) — among
          every player the model has ever said "25%" about, roughly 1 in 4 actually broke out. Then it gets{" "}
          <b>renormalized per position</b>: nothing about a calibration curve forces a position's probabilities to
          sum to how many breakouts that position actually produces in a season, so each position is rescaled so
          they do (see the README's v1.7 section). The league-wide base rate for a random cheap/late player is
          about 3–6% (QB highest, WR/TE thinnest — see the Positions view for the real number at each spot); a
          tier label is that player's displayed percentage as a multiple of his own position's base rate this
          year, not a fixed cutoff.
        </p>
        <h2>What it trains on</h2>
        <p>
          Only information that existed <b>before</b> the season started: the player's own prior-season usage and
          efficiency, his team's situation (new coach, new offensive coordinator, teammates who left and freed up
          targets/carries), his draft pedigree, and the market's own preseason consensus rank. Every feature is
          built with a strict N-1 shift — nothing from the season being predicted ever leaks into the inputs used
          to predict it. That is the one rule the whole pipeline is built around.
        </p>
        <h2>Two models, two lenses (v2.0)</h2>
        <p>
          This board runs <b>two different models</b> side by side, because they answer different questions well.
          The older model (above) directly predicts one rare event — "will this cheap player finish as a starter"
          — which works fine for QB but starves RB/TE of real signal: at those sample sizes its calibrated odds
          for a whole position can collapse toward a flat number that barely distinguishes one player from
          another. The newer model instead predicts each player's full <b>range</b> of likely per-game scoring (a
          floor, an expected value, and a ceiling), and every other number on the page — his predicted finish, his
          odds of being a starter, a flex play, or a bust — is read straight off that one range, for every player,
          whether he is cheap enough to be a "breakout" or not. Neither model throws the other away:{" "}
          <b>Breakout Hunt</b> ranks only the cheap-enough-to-qualify players, by whichever of the two models
          actually tests better at that position (see Trust); <b>Full Projections</b> ranks every scored veteran by
          the newer model's expected points, breakout-eligible or not, with the full range shown as a bar. A player
          already priced as a starter never appears in Breakout Hunt — the newer model still shows where he's
          projected to finish, just without the breakout framing that wouldn't make sense for him.
        </p>
        <h2>What it does NOT know</h2>
        <ul>
          <li>
            <b>Injuries, holdouts, suspensions.</b> The manual availability screen this board reads from is empty by
            design — nobody can responsibly guess next month's injury report from here. Check the news yourself
            before you draft anyone on this page.
          </li>
          <li>
            <b>Anything after the snapshot date</b> shown in the header (the last weekly market-consensus pull this
            build used).
          </li>
          <li>
            <b>This year's actual games.</b> Obviously — none have been played yet.
          </li>
          <li>
            <b>Who plays how many games.</b> Every projection on this page is a PER-GAME rate — a player projected
            for 14 points a game who only plays 8 of them is not the same fantasy asset as one who plays 17, and
            nothing here tells the two apart. Games-played / injury risk is not modeled anywhere in this pipeline.
          </li>
        </ul>
        <h2>Vegas lines: tested, not used</h2>
        <p>
          Team-level Vegas win totals and implied scoring were tried as model inputs, twice, across all four
          positions. Neither pass improved holdout accuracy enough to justify keeping — the market's own preseason
          draft price already contains most of what the betting lines would have added. The board still shows
          Vegas implied points where available, purely as a manual sanity-check column, never as a model input.
        </p>
        <h2>The rookie section is a different, weaker model</h2>
        <p>
          Rookies have no NFL history to build the real feature set from, so they're scored by a much simpler
          heuristic (draft position, landing-spot opportunity) on its own probability scale — never comparable
          number-for-number to the veteran percentages above it. Treat rookie calls as tie-breakers on your last
          couple of picks, not as strong signals.
        </p>
      </div>
    </div>
  );
}
