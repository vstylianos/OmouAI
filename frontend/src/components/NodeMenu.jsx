import { useEffect, useRef, useState } from "react";
import { ReadingCard } from "./ChatPanel.jsx";

const initials = (name) =>
  name.split(/[\s-]+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase();

/* The right-click menu on an argument. Two things a person can do with a
   node: make their own point on it, or ask a persona for a reason for or
   against it. The Devil's Advocate only ever argues against. */
export function NodeMenu({ menu, personas, busy, onOwnPoint, onAsk, onDrop, onClose }) {
  const ref = useRef(null);
  const [asking, setAsking] = useState(null); // persona id whose directions are open
  useEffect(() => {
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onKey); };
  }, [onClose]);
  if (!menu) return null;
  const label = menu.nodeId === "topic" ? "the claim" : menu.nodeId;

  return (
    <div ref={ref} className="node-menu" style={{ left: menu.x, top: menu.y }} role="menu" aria-label={`Actions on ${label}`}>
      <div className="node-menu-head eyebrow">On {label}</div>
      <button className="node-menu-item" role="menuitem" disabled={busy} onClick={() => onOwnPoint(menu.nodeId, menu)}>
        <span className="mi-icon you">You</span>
        <span><b>Make my own point</b><small>Write an argument for or against it</small></span>
      </button>
      <div className="node-menu-sep eyebrow">Ask a persona for a reason</div>
      {personas.map((p) => {
        const contrarian = p.kind === "contrarian";
        const open = asking === p.id || personas.length === 1;
        return (
          <div key={p.id} className="node-menu-persona">
            <button className="node-menu-item" role="menuitem" aria-expanded={open} disabled={busy}
              onClick={() => setAsking((cur) => (cur === p.id ? null : p.id))}>
              <span className="mi-icon">{initials(p.name)}</span>
              <span><b>{p.name}</b><small>{contrarian ? "always argues against" : "a reason for or against"}</small></span>
            </button>
            {open && (
              <div className="node-menu-dirs">
                {!contrarian && (
                  <button className="btn small" disabled={busy} onClick={() => onAsk(menu.nodeId, "support", p.id)}>+ a reason for</button>
                )}
                <button className="btn small" disabled={busy} onClick={() => onAsk(menu.nodeId, "attack", p.id)}>− a reason against</button>
              </div>
            )}
          </div>
        );
      })}
      {menu.nodeId !== "topic" && (
        <>
          <div className="node-menu-sep" />
          <button className="node-menu-item" role="menuitem" disabled={busy} onClick={() => { onClose(); onDrop(menu.nodeId); }}>
            <span className="mi-icon danger">×</span>
            <span><b>Remove this argument</b><small>Takes everything that hangs on it</small></span>
          </button>
        </>
      )}
    </div>
  );
}

/* The popover for writing your own point on a chosen node. Stage one is
   the text; stage two, once the reading comes back, is the same
   check-my-reading card the chat uses, in place. */
export function PointPopover({ popover, nodes, levels, personas, pending, busy, onSubmit, onConfirm, onDiscard, onClose }) {
  const [text, setText] = useState("");
  const ref = useRef(null);
  useEffect(() => { setText(""); }, [popover?.nodeId]);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);
  if (!popover) return null;
  const target = nodes.find((n) => n.id === popover.nodeId);
  const label = target?.is_topic ? "the claim" : popover.nodeId;
  const submit = () => { const t = text.trim(); if (t && !busy) onSubmit(popover.nodeId, t); };

  return (
    <div ref={ref} className="point-popover" style={{ left: popover.x, top: popover.y }} role="dialog" aria-label={`Your point on ${label}`}>
      {pending?.attachment ? (
        <ReadingCard
          key={pending.attachment.restated}
          pending={{ ...pending.attachment, original: pending.text }}
          nodes={nodes}
          levels={levels}
          personaName={personas.length === 1 ? personas[0].name : "The personas"}
          busy={busy}
          onConfirm={onConfirm}
          onDiscard={onDiscard}
        />
      ) : (
        <>
          <div className="eyebrow">Your point on {label}</div>
          {target && <div className="point-target">“{target.text}”</div>}
          <textarea
            rows={3} value={text} autoFocus disabled={busy}
            placeholder="Say why this holds, or why it doesn't…"
            aria-label="Write your point"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); } }}
          />
          <div className="pending-actions">
            <button className="btn user" disabled={busy || !text.trim()} onClick={submit}>{busy ? "Reading…" : "Continue"}</button>
            <button className="btn" disabled={busy} onClick={onClose}>Cancel</button>
          </div>
        </>
      )}
    </div>
  );
}
