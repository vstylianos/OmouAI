import { useEffect, useState } from "react";
import { api } from "../api.js";

const pct = (v) => `${Math.round(v * 100)}%`;

/* "[a3]" in a note becomes a mono chip so ids stand out from prose. */
function Note({ text }) {
  const parts = text.split(/(\[[a-z]+\d*\])/g);
  return (
    <>
      {parts.map((p, i) =>
        /^\[[a-z]+\d*\]$/.test(p) ? (
          <span key={i} className="ref static">{p.slice(1, -1)}</span>
        ) : (
          <span key={i}>{p}</span>
        )
      )}
    </>
  );
}

function Section({ title, items, empty }) {
  return (
    <section>
      <h2>{title}</h2>
      {items.length === 0 ? (
        <p className="report-empty">{empty}</p>
      ) : (
        <ul>
          {items.map((line, i) => (
            <li key={i}><Note text={line} /></li>
          ))}
        </ul>
      )}
    </section>
  );
}

/* Documentation stage as meeting notes: one neutral model call, worded
   from the engine's record only (trajectory, graph, computed agreements,
   divergences, trade-offs, goal verdicts) and cached until the state
   changes. Replaces the app view while open; printable. */
export default function SummaryView({ sessionId, onClose }) {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.summary(sessionId).then(setSummary).catch((e) => setError(e.message));
  }, [sessionId]);

  const notes = summary?.notes;
  return (
    <div className="report">
      <div className="report-bar">
        <button className="btn" onClick={onClose}>← Back to deliberation</button>
        <button className="btn primary" onClick={() => window.print()}>Print / save PDF</button>
      </div>

      {error && <div className="error">{error}</div>}
      {!summary && !error && <p className="lede">Writing up the notes…</p>}

      {summary && (
        <article className="report-page">
          <header>
            <span className="eyebrow">Meeting notes</span>
            <h1>{summary.topic}</h1>
            <p className="report-meta">
              With <b>{summary.personas.map((p) => p.name).join(", ")}</b> ·{" "}
              {new Date().toLocaleDateString()} · {summary.turns} turn{summary.turns === 1 ? "" : "s"} ·{" "}
              {summary.arguments} argument{summary.arguments === 1 ? "" : "s"} · position{" "}
              <span className="num">{pct(summary.opening_stance ?? 0.5)} → {pct(summary.stance)}</span>
            </p>
          </header>

          <section>
            <h2>Summary</h2>
            <p><Note text={notes.summary} /></p>
          </section>

          <Section title="Key points raised" items={notes.key_points} empty="Nothing was argued yet." />
          <Section title="Where you agreed" items={notes.agreements} empty="No agreement recorded." />
          <Section title="Where you still disagree" items={notes.disagreements} empty="No divergence recorded." />
          <Section title="What would move the discussion" items={notes.next_steps} empty="Nothing to suggest yet." />

          {summary.lens.length > 0 && (
            <section>
              <h2>Goals, as the deliberation stands</h2>
              <ul>
                {summary.lens.map((e) => (
                  <li key={e.id}>
                    <b>{e.label}</b>: {e.verdict}{" "}
                    <span className="report-note">(σ {e.sigma.toFixed(2)})</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <p className="report-note">
            Written by a neutral note-taker from the computed record; every figure comes
            from the argumentation engine.
          </p>
        </article>
      )}
    </div>
  );
}
