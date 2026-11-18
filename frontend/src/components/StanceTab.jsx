import { useEffect, useState } from "react";
import { api } from "../api.js";
import Sparkline from "./Sparkline.jsx";

const pct = (v) => `${Math.round(v * 100)}%`;

/* Identification: where the two of you agree, where you still diverge, and
   where arguing matters most. Computed by the engine — no model call. */
export default function StanceTab({ state, onFocus }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    api.report(state.session_id).then((r) => live && setReport(r)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [state]);

  return (
    <div className="stage-body">
      <div className="stage-scroll">
        <div className="insight-section">
          <span className="eyebrow">How the position moved</span>
          <div className="trajectory">
            <Sparkline history={state.history} width={220} height={54} />
            <div className="traj-list">
              {state.history.map((h, i) => (
                <span key={i}><span className="num">{pct(h.stance)}</span> — {h.label}</span>
              ))}
            </div>
          </div>
        </div>

        {error && <div className="error">{error}</div>}
        {!report && !error && <div className="insight-empty">Computing…</div>}
        {report && (
          <>
            <div className="insight-section">
              <span className="eyebrow">Where you agree</span>
              {report.agreements.length === 0 && <div className="insight-empty">Nothing yet — make a point.</div>}
              {report.agreements.map((a) => (
                <div key={a.id} className="insight">
                  <button className="ref" onClick={() => onFocus(a.id)}>{a.id}</button>
                  <span>“{a.text}” <span className="insight-note">— {a.basis}.</span></span>
                </div>
              ))}
            </div>
            <div className="insight-section">
              <span className="eyebrow">Where you still diverge</span>
              {report.divergences.length === 0 && (
                <div className="insight-empty">No recorded divergence — contest a confidence, or hold a point firmly that it objects to.</div>
              )}
              {report.divergences.map((d) => (
                <div key={d.id} className="insight">
                  <button className="ref" onClick={() => onFocus(d.id)}>{d.id}</button>
                  <span>
                    “{d.text}”{" "}
                    <span className="insight-note">
                      — {d.basis}; a <b>{d.kind}</b> disagreement{d.kind === "value" ? " (evidence cannot settle it)" : " (evidence could settle it)"}.
                    </span>
                  </span>
                </div>
              ))}
            </div>
            <div className="insight-section">
              <span className="eyebrow">Where arguing matters most</span>
              {report.trade_offs.length === 0 && <div className="insight-empty">Nothing on the table yet.</div>}
              {report.trade_offs.map((t) => (
                <div key={t.id} className="insight">
                  <button className="ref" onClick={() => onFocus(t.id)}>{t.id}</button>
                  <span>
                    “{t.text}”{" "}
                    <span className="insight-note">
                      — moves the position <span className="num">{t.sensitivity >= 0 ? "+" : ""}{t.sensitivity.toFixed(2)}</span> per unit of confidence; {t.kind}.
                    </span>
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
