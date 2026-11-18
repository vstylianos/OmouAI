import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

const pct = (v) => `${Math.round(v * 100)}%`;
const goalShort = (goalId, goals) => {
  if (goalId.startsWith("sdg_")) return `SDG ${goalId.slice(4)}`;
  const label = goals?.find((g) => g.id === goalId)?.label ?? goalId;
  return label.length > 22 ? `${label.slice(0, 21)}…` : label;
};

/* Confidence slider with the same commit discipline as before: local value
   while dragging, one PATCH per gesture (pointerup and the blur that follows
   it are deduped), keyboard taps debounced. */
function ConfidenceSlider({ node, busy, colour, onRescore }) {
  const [tau, setTau] = useState(node.base_score);
  const sent = useRef(node.base_score);
  const keyTimer = useRef(null);
  useEffect(() => { setTau(node.base_score); sent.current = node.base_score; }, [node.id, node.base_score]);
  useEffect(() => () => clearTimeout(keyTimer.current), []);
  const commit = () => {
    clearTimeout(keyTimer.current);
    if (Math.abs(tau - sent.current) > 1e-6) { sent.current = tau; onRescore(node.id, tau); }
  };
  const commitSoon = () => { clearTimeout(keyTimer.current); keyTimer.current = setTimeout(commit, 600); };
  return (
    <div className="slider-bar">
      <div className="fill" style={{ width: `${tau * 100}%`, background: colour }} />
      <input
        type="range" min="0" max="1" step="0.01" value={tau} disabled={busy}
        aria-label={`Confidence for ${node.id}`}
        onChange={(e) => setTau(parseFloat(e.target.value))}
        onPointerUp={commit} onKeyUp={commitSoon} onBlur={commit}
      />
    </div>
  );
}

export default function Inspector({
  state, selectedId, expert, busy, levels,
  onSelect, onRescore, onRetag, onDrop, onOpenMenu, onToggleExpert,
}) {
  const node = state.nodes.find((n) => n.id === selectedId) ?? null;
  const [leverage, setLeverage] = useState([]);
  const [pickOpen, setPickOpen] = useState(false);
  const [whyOpen, setWhyOpen] = useState(false);

  useEffect(() => {
    let live = true;
    api.leverage(state.session_id).then((r) => live && setLeverage(r.leverage)).catch(() => {});
    return () => { live = false; };
  }, [state]);
  useEffect(() => { setPickOpen(false); setWhyOpen(false); }, [selectedId]);

  const foot = (
    <div className="inspector-foot">
      <div className="toggle">
        <span>Expert mode</span>
        <button className="switch" role="switch" aria-checked={expert} onClick={onToggleExpert} aria-label="Expert mode" />
      </div>
      <span className="help">Shows τ, σ, node types and polarity on every card.</span>
    </div>
  );

  if (!node) {
    return (
      <aside className="inspector">
        <div className="inspector-head"><span className="eyebrow">Inspector</span></div>
        <div className="inspector-empty">
          Click an argument in the map to see how it holds, what pushes on it, and to contest
          its confidence. Right-click one to make your own point on it or to ask a persona for a reason.
        </div>
        <div className="inspector-body" style={{ justifyContent: "flex-end" }}>{foot}</div>
      </aside>
    );
  }

  const mine = node.provenance === "user";
  const colour = node.is_topic ? "var(--ink-3)" : mine ? "var(--user)" : node.relation_type === "attack" ? "var(--attack)" : "var(--support)";
  const parent = state.nodes.find((n) => n.id === node.parent_id);
  const children = state.nodes.filter((n) => n.parent_id === node.id);
  const objections = children.filter((c) => c.relation_type === "attack");
  const supports = children.filter((c) => c.relation_type === "support");
  const lev = leverage.find((l) => l.id === node.id);
  const rank = leverage.findIndex((l) => l.id === node.id);
  const tags = node.goal_impacts ?? [];
  const untagged = (state.goals ?? []).filter((g) => !tags.some((t) => t.goal_id === g.id));
  const removeTag = (goalId) =>
    onRetag(node.id, tags.filter((t) => t.goal_id !== goalId).map((t) => ({ goal_id: t.goal_id, direction: t.direction })));
  const addTag = (goalId, direction) => {
    setPickOpen(false);
    onRetag(node.id, [...tags.map((t) => ({ goal_id: t.goal_id, direction: t.direction })), { goal_id: goalId, direction }]);
  };
  const levelLabel = mine && levels?.length
    ? levels.reduce((b, l) => (Math.abs(l.tau - node.base_score) < Math.abs(b.tau - node.base_score) ? l : b)).label.toLowerCase()
    : null;

  return (
    <aside className="inspector">
      <div className="inspector-head">
        <span className="eyebrow">Selected</span>
        <span className="who" style={{ color: colour }}>
          {node.is_topic ? "the claim" : `${node.id} · ${mine ? "yours" : node.author_name ?? "persona"}`}
        </span>
        <span className="acts">
          <button className="btn small" disabled={busy} title="Make your own point on this, or ask a persona for a reason"
            onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); onOpenMenu(node.id, r.left - 300, r.bottom + 6); }}>Add to this…</button>
          {!node.is_topic && (
            <button className="btn small danger" disabled={busy} title="Remove this argument and everything that hangs on it" onClick={() => onDrop(node.id)}>×</button>
          )}
        </span>
      </div>

      <div className="inspector-body">
        <div style={{ lineHeight: 1.45 }}>{node.text}</div>

        <div className="sect">
          <div className="row">
            <span className="eyebrow">{node.is_topic ? "Prior on the claim" : mine ? "Your confidence" : "Its confidence"}</span>
            <span className="num">{pct(node.base_score)}{levelLabel ? ` · ${levelLabel}` : ""}</span>
          </div>
          <ConfidenceSlider node={node} busy={busy} colour={colour} onRescore={onRescore} />
          {!node.is_topic && (
            <>
              <div className="row"><span className="eyebrow">Holds at</span><span className="num">{pct(node.strength)}</span></div>
              <div className="bar"><div className="fill" style={{ width: `${node.strength * 100}%`, background: colour }} /></div>
            </>
          )}
          <span className="help">
            {node.is_topic
              ? "The deliberation starts neutral. Dragging this sets a prior for the claim itself."
              : mine
              ? "Drag the handle to change your confidence. No model call."
              : "Drag the handle to contest its confidence. The gap is recorded as a disagreement."}
          </span>
          {!mine && !node.is_topic && node.rationale && (
            <>
              <button className="btn small ghost" style={{ alignSelf: "flex-start" }} aria-expanded={whyOpen} onClick={() => setWhyOpen((v) => !v)}>
                {whyOpen ? "Hide why" : `Why ${pct(node.elicited_tau ?? node.base_score)}?`}
              </button>
              {whyOpen && (
                <div className="why">
                  <span className="eyebrow">Its reasoning for the confidence</span>{node.rationale}
                </div>
              )}
            </>
          )}
        </div>

        {parent && (
          <div className="sect">
            <span className="eyebrow">{node.relation_type === "attack" ? "Objects to" : "Supports"}</span>
            <button className="related" onClick={() => onSelect(parent.id)}>
              <span className="id">{parent.is_topic ? "claim" : parent.id}</span><span className="text">{parent.text}</span>
              <span className="val">{pct(parent.strength)}</span>
            </button>
            {node.relation_rationale && <span className="help">{node.relation_rationale}</span>}
          </div>
        )}

        {objections.length > 0 && (
          <div className="sect">
            <span className="eyebrow">Objected to by</span>
            {objections.map((c) => (
              <button key={c.id} className="related" onClick={() => onSelect(c.id)}>
                <span className="id">{c.id}</span><span className="text">{c.text}</span>
                <span className="val" style={{ color: "var(--attack)" }}>{pct(c.strength)}</span>
              </button>
            ))}
          </div>
        )}
        {supports.length > 0 && (
          <div className="sect">
            <span className="eyebrow">Supported by</span>
            {supports.map((c) => (
              <button key={c.id} className="related" onClick={() => onSelect(c.id)}>
                <span className="id">{c.id}</span><span className="text">{c.text}</span>
                <span className="val" style={{ color: "var(--support)" }}>{pct(c.strength)}</span>
              </button>
            ))}
          </div>
        )}

        {!node.is_topic && (state.goals?.length > 0) && (
          <div className="sect">
            <span className="eyebrow">Goals it bears on</span>
            <div className="chips" style={{ gap: 4 }}>
              {tags.map((t) => (
                <span
                  key={t.goal_id}
                  className={`tag goal ${t.source === "user" ? "user-src" : t.source ?? "judged"}`}
                  title={`${t.direction === "positive" ? "Helps" : "Strains"} ${t.goal_id} — ${t.source === "user" ? "asserted by you" : "model-judged"}`}
                >
                  {t.direction === "positive" ? "↑" : "↓"} {goalShort(t.goal_id, state.goals)}
                  <button className="tag-x" aria-label={`Remove ${t.goal_id} tag`} disabled={busy} onClick={() => removeTag(t.goal_id)}>×</button>
                </span>
              ))}
              {untagged.length > 0 && (
                <button className="tag goal judged" style={{ color: "var(--ink-3)", cursor: "pointer" }} aria-expanded={pickOpen} disabled={busy} onClick={() => setPickOpen((v) => !v)}>+ goal</button>
              )}
            </div>
            {pickOpen && (
              <div className="tag-picker">
                {untagged.map((g) => (
                  <div key={g.id} className="tag-pick-row">
                    <span className="tag-pick-label" title={g.label}>{g.label}</span>
                    <button className="btn small" disabled={busy} onClick={() => addTag(g.id, "positive")}>↑ helps</button>
                    <button className="btn small" disabled={busy} onClick={() => addTag(g.id, "negative")}>↓ strains</button>
                  </div>
                ))}
              </div>
            )}
            {tags.length === 0 && !pickOpen && <span className="help">Not tagged.</span>}
          </div>
        )}

        {lev && (
          <div className="sect">
            <span className="eyebrow">If you pushed on this</span>
            <span style={{ fontSize: 12.5, color: "var(--ink-2)" }}>
              The position would move <b className="mono" style={{ color: lev.sensitivity < 0 ? "var(--attack)" : "var(--support)" }}>{lev.sensitivity >= 0 ? "+" : ""}{lev.sensitivity.toFixed(2)}</b> per unit of confidence
              {rank === 0 ? " — the most leverage on the table." : "."}{" "}
              {lev.kind === "value" ? "It rests on values; evidence cannot settle it." : "Evidence could settle it."}
            </span>
          </div>
        )}

        {foot}
      </div>
    </aside>
  );
}
