import { useEffect, useRef, useState } from "react";

const SEMANTICS = [
  ["modular", "Modular (linear)"],
  ["dfquad", "DF-QuAD"],
  ["quadratic_energy", "Quadratic energy"],
];

/* Researcher knobs, out of the participant's way: how the position is
   computed, how readily the persona updates, expert mode, theme. */
export default function SettingsDrawer({
  state, busy, expert, theme, onSettings, onToggleExpert, onTheme, onReset, onClose,
}) {
  const [k, setK] = useState(state.conservativeness);
  const sent = useRef(state.conservativeness);
  const keyTimer = useRef(null);
  useEffect(() => { setK(state.conservativeness); sent.current = state.conservativeness; }, [state.conservativeness]);
  useEffect(() => () => clearTimeout(keyTimer.current), []);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const commit = () => {
    clearTimeout(keyTimer.current);
    if (Math.abs(k - sent.current) > 1e-6) { sent.current = k; onSettings({ semantics: "modular", conservativeness: k }); }
  };
  const commitSoon = () => { clearTimeout(keyTimer.current); keyTimer.current = setTimeout(commit, 600); };

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label="Settings">
        <div className="drawer-head">
          <h2>Settings</h2>
          <button className="btn ghost" onClick={onClose}>Close</button>
        </div>
        <div className="drawer-body">
          <div className="sect">
            <span className="eyebrow">How the position is computed</span>
            <div className="chips">
              {SEMANTICS.map(([id, label]) => (
                <button key={id} className="chip" aria-pressed={state.semantics === id} disabled={busy} onClick={() => onSettings({ semantics: id })}>{label}</button>
              ))}
            </div>
            <span className="plain">The gradual semantics that turns the argument map into a position. The trajectory recomputes when you switch.</span>
          </div>

          {state.semantics === "modular" && (
            <div className="sect">
              <div className="row"><span className="eyebrow">How readily the persona updates</span><span className="num" style={{ fontSize: 12 }}>k {k.toFixed(1)}</span></div>
              <div className="slider-bar">
                <div className="fill" style={{ width: `${((k - 1) / 3) * 100}%`, background: "var(--accent)" }} />
                <input type="range" min="1" max="4" step="0.1" value={k} disabled={busy} aria-label="How readily the persona updates"
                  onChange={(e) => setK(parseFloat(e.target.value))} onPointerUp={commit} onKeyUp={commitSoon} onBlur={commit} />
              </div>
              <div className="scale-ends"><span>Moves easily</span><span>Holds firm</span></div>
              <span className="plain">The table's epistemic disposition: how far a given weight of counter-evidence moves the position. Default is the personas' mean, {(state.personas.reduce((a, p) => a + p.conservativeness, 0) / state.personas.length).toFixed(1)}.</span>
            </div>
          )}

          <div className="sect">
            <div className="toggle">
              <span>Expert mode</span>
              <button className="switch" role="switch" aria-checked={expert} onClick={onToggleExpert} aria-label="Expert mode" />
            </div>
            <span className="plain">Shows τ, σ, node types and polarity on every card.</span>
          </div>

          <div className="sect">
            <span className="eyebrow">Theme</span>
            <div className="chips">
              {["light", "dark", "system"].map((t) => (
                <button key={t} className="chip" aria-pressed={theme === t} onClick={() => onTheme(t)}>{t[0].toUpperCase() + t.slice(1)}</button>
              ))}
            </div>
          </div>

          <div className="sect">
            <span className="eyebrow">This session</span>
            <span className="plain mono">{state.usage.calls} model calls · {state.usage.prompt_tokens.toLocaleString()} prompt + {state.usage.completion_tokens.toLocaleString()} completion tokens</span>
            <button className="btn danger" style={{ alignSelf: "flex-start", marginTop: 6 }} disabled={busy} onClick={onReset}>New deliberation</button>
          </div>
        </div>
      </aside>
    </>
  );
}
