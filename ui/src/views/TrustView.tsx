import { useState } from "react";
import type { Payload } from "../types";
import { POS_ORDER, pct } from "../lib/format";

interface Props {
  payload: Payload;
}

/** Ported from template.py's renderTrust(). */
export function TrustView({ payload }: Props) {
  const { trust } = payload;
  const [showTable, setShowTable] = useState(false);

  const chartRows = POS_ORDER.map((pos) => {
    const t = trust.positions[pos.toLowerCase()];
    if (!t) return null;
    const m = t.holdout_pr_auc ?? 0;
    const b = t.best_baseline_pr_auc ?? 0;
    return { pos, t, m, b };
  }).filter((r): r is NonNullable<typeof r> => r != null);
  const globalMax = Math.max(0.05, ...chartRows.map((r) => Math.max(r.m, r.b)));

  const qt = trust.quantile || {};
  const anyQuantile = POS_ORDER.some((p) => qt[p.toLowerCase()] && qt[p.toLowerCase()].coverage_q10_q90 != null);
  const gateRows = POS_ORDER.map((p) => {
    const q = qt[p.toLowerCase()];
    return q && q.gate ? { pos: p, g: q.gate } : null;
  }).filter((r): r is NonNullable<typeof r> => r != null);

  return (
    <div className="view">
      <h1 className="vtitle">Trust</h1>
      <p className="vsub">
        The model finds breakouts the market misses — tested on real years it never saw during tuning, with no
        do-overs.
      </p>

      <div className="vtable">
        <table>
          <thead>
            <tr>
              <th>Position</th>
              <th>Its real track record</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <b>QB</b>
              </td>
              <td>Best position. Its 2025 cheap-QB picks included Stafford, Goff and Lawrence — all hit.</td>
            </tr>
            <tr>
              <td>
                <b>RB</b>
              </td>
              <td>Good. Called Chase Brown ('24) and Etienne ('25) before they were startable.</td>
            </tr>
            <tr>
              <td>
                <b>WR</b>
              </td>
              <td>Decent. Called Olave and Rice in '25; missed in '24 (only one WR broke out league-wide that year).</td>
            </tr>
            <tr>
              <td>
                <b>TE</b>
              </td>
              <td>Thinnest. Called Kyle Pitts in '25; treat TE calls with extra doubt.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2 className="section-h2">The model vs. just using the market's own rank</h2>
      <p className="vsub" style={{ marginBottom: 8 }}>
        Each pair: how well the model's calibrated score finds real breakouts (PR-AUC, higher = better) on the two
        holdout years it never trained on, vs. the best of three simple non-model baselines (market rank, last
        year's points, age-adjusted market rank).
      </p>
      <div className="legend" hidden={showTable}>
        <span>
          <span className="sw model" />
          Model
        </span>
        <span>
          <span className="sw baseline" />
          Best baseline
        </span>
      </div>

      {!showTable ? (
        <div className="barpair-chart">
          {chartRows.map((r) => (
            <div key={r.pos}>
              <div className="barpair-row">
                <span className="lbl">{r.pos}</span>
                <span className="track">
                  <span className="fill model" style={{ width: `${Math.round((100 * r.m) / globalMax)}%` }} />
                </span>
                <span className="val">{r.m.toFixed(3)}</span>
              </div>
              <div className="barpair-row">
                <span className="lbl" />
                <span className="track">
                  <span
                    className="fill baseline"
                    style={{ width: `${Math.round((100 * r.b) / globalMax)}%` }}
                    title={r.t.best_baseline_name || ""}
                  />
                </span>
                <span className="val">{r.b.toFixed(3)}</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="vtable">
          <table>
            <thead>
              <tr>
                <th>Position</th>
                <th>Model holdout PR-AUC</th>
                <th>Best baseline</th>
                <th>Baseline PR-AUC</th>
              </tr>
            </thead>
            <tbody>
              {chartRows.map((r) => (
                <tr key={r.pos}>
                  <td>{r.pos}</td>
                  <td className="mono">{r.m.toFixed(3)}</td>
                  <td>{r.t.best_baseline_name || "—"}</td>
                  <td className="mono">{r.b.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <button className="toggle-btn" onClick={() => setShowTable((v) => !v)}>
        {showTable ? "Show as chart" : "Show as table"}
      </button>

      {anyQuantile && (
        <>
          <h2 className="section-h2">The projection ranges (v2.0)</h2>
          <p className="vsub" style={{ marginBottom: 4 }}>
            Every player also gets a floor–expected–ceiling range (his 10th, 50th and 90th percentile per-game
            score). A well-calibrated 80% range should contain the real result about 80% of the time in years the
            model never trained on.
          </p>
          {POS_ORDER.map((pos) => {
            const t = qt[pos.toLowerCase()];
            if (!t || t.coverage_q10_q90 == null) return null;
            return (
              <div className="coverage-stat" key={pos}>
                <b>{pos}:</b> its 80% ranges contained the real result{" "}
                <b>{Math.round(t.coverage_q10_q90 * 100)}%</b> of the time in the 2024–2025 test years
                {t.spearman_q50_actual != null && (
                  <>
                    {" "}
                    (rank correlation of its median projection to actual finish:{" "}
                    <b>{t.spearman_q50_actual.toFixed(2)}</b>)
                  </>
                )}
                .
              </div>
            );
          })}
          {gateRows.length > 0 && (
            <>
              <h3 className="section-h3">Which engine runs Breakout Hunt at each position</h3>
              <p className="vsub" style={{ marginBottom: 8 }}>
                Pre-stated rule: the projection model becomes the primary engine for a position iff its top-10
                precision (ranking eligible players by startable odds) is at least as good as the older
                breakout-odds model's, on the identical 2024–2025 holdout rows. Ties favor the projection model.
              </p>
              <div className="vtable">
                <table>
                  <thead>
                    <tr>
                      <th>Position</th>
                      <th>Projection-model top-10 precision</th>
                      <th>Older-model top-10 precision</th>
                      <th>Primary engine</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gateRows.map((r) => (
                      <tr key={r.pos}>
                        <td>{r.pos}</td>
                        <td className="mono">
                          {r.g.quantile_top10_precision != null ? r.g.quantile_top10_precision.toFixed(3) : "—"}
                        </td>
                        <td className="mono">
                          {r.g.classifier_top10_precision != null ? r.g.classifier_top10_precision.toFixed(3) : "—"}
                        </td>
                        <td>
                          <b>{r.g.primary_engine}</b>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}

      <h2 className="section-h2">Named holdout top-10 lists, 2024 &amp; 2025</h2>
      <p className="vsub" style={{ marginBottom: 8 }}>
        The model's actual top-10 cheap-price picks each holdout year, and whether each one really broke out. ✓ =
        hit.
      </p>
      {POS_ORDER.map((pos) => {
        const nl = trust.named_lists[pos.toLowerCase()];
        if (!nl) return null;
        return (
          <div key={pos}>
            <h3 className="section-h3">{pos}</h3>
            <div className="namedlist">
              {["2024", "2025"].map((yr) => {
                const rows = nl[yr] || [];
                return (
                  <div className="yr" key={yr}>
                    <h3>{yr}</h3>
                    {rows.map((p, i) => (
                      <div className={`hitrow${p.hit ? " hit" : ""}`} key={`${p.player}-${i}`}>
                        <span>
                          {p.hit && <span className="chk">✓</span>}
                          {p.player}
                        </span>
                        <span className="hm">{pct(p.prob)}</span>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      <p className="fine">
        Tested honestly: the model was trained only on past seasons, then graded on 2024–2025 games it had never
        seen, with no do-overs. Its ten best ideas per year produce one to three hits — which is exactly what
        winning a league on the margins looks like. Vegas betting lines were tested as an ingredient twice (see
        Method) and added nothing measurable; the market's own draft prices already contain that information.
      </p>
    </div>
  );
}
