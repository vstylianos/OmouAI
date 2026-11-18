import { useEffect, useRef, useState } from "react";

const pct = (v) => `${Math.round(v * 100)}%`;
const initials = (name) =>
  name.split(/[\s-]+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
const HINTS = { unsure: "a hunch", somewhat: "plausible", moderately: "likely", very: "good grounds", certain: "no doubt" };

function Move({ label, delta }) {
  const unchanged = Math.abs(delta.change) < 1e-9;
  const colour = unchanged ? "var(--ink-3)" : delta.change > 0 ? "var(--support)" : "var(--attack)";
  return (
    <div>
      <span className="eyebrow">{label}</span>
      <span className="val" style={{ color: colour }}>
        {pct(delta.before)}→{pct(delta.after)}
      </span>
    </div>
  );
}

/* One stance card per turn: the steps (your point, its response) and the
   net, plus the driver in words. */
function StepsCard({ delta, steps, onFocus }) {
  if (!delta) return null;
  const rows = steps && steps.length > 1 ? steps : [{ label: "stance", delta }];
  const driver = (steps?.length ? steps[steps.length - 1].delta : delta).driver;
  return (
    <div className="steps">
      {rows.map((s, i) => <Move key={i} label={s.label} delta={s.delta} />)}
      {steps && steps.length > 1 && <Move label="net" delta={delta} />}
      {driver && (
        <div className="steps-why">
          {driver.status === "added" ? "New argument " : "Most affected: "}
          <button className="ref" onClick={() => onFocus(driver.id)}>{driver.id}</button>{" "}
          {driver.status === "added" ? `holds at ${pct(driver.after)}.` : `moved ${pct(driver.before)} → ${pct(driver.after)}.`}
        </div>
      )}
    </div>
  );
}

/* Every point of the user's stops here before it enters the graph. */
export function ReadingCard({ pending, nodes, levels, personaName, onConfirm, onDiscard, busy }) {
  const mine = pending.provenance !== "generated";
  const shaky = pending.confidence < 75;
  const [editing, setEditing] = useState(shaky && mine);
  const [targetId, setTargetId] = useState(
    nodes.some((n) => n.id === pending.target_id) ? pending.target_id : nodes.find((n) => n.is_topic)?.id
  );
  const [relation, setRelation] = useState(pending.relation_type);
  const [confidence, setConfidence] = useState(null);
  const changed = targetId !== pending.target_id || relation !== pending.relation_type;
  const overrides = changed ? { target_id: targetId, relation_type: relation } : null;
  const ready = !busy && (!mine || confidence !== null);
  const target = nodes.find((n) => n.id === targetId);
  const targetLabel = target?.is_topic ? "claim" : targetId;

  if (!mine) {
    return (
      <div className="pending">
        <div className="pending-q">
          At your request I drafted this argument, and I'd attach it{" "}
          <b>{pending.relation_type}ing {targetLabel}</b> (<b className="num">{pending.confidence}%</b> sure of the placement) — {pending.rationale}
          <br /><br /><i>“{pending.restated}”</i>
        </div>
        {editing && (
          <div className="placement-edit">
            <select value={targetId} aria-label="Argument this bears on" onChange={(e) => setTargetId(e.target.value)}>
              {nodes.map((n) => <option key={n.id} value={n.id}>[{n.id}] {n.text.length > 80 ? `${n.text.slice(0, 79)}…` : n.text}</option>)}
            </select>
            <div className="chips">
              <button className="chip" aria-pressed={relation === "support"} onClick={() => setRelation("support")}>supports it</button>
              <button className="chip" aria-pressed={relation === "attack"} onClick={() => setRelation("attack")}>attacks it</button>
            </div>
          </div>
        )}
        <div className="pending-actions">
          <button className="btn primary" disabled={busy} onClick={() => onConfirm(overrides, null)}>Yes, add it</button>
          {!editing && <button className="btn" disabled={busy} onClick={() => setEditing(true)}>Attach it elsewhere…</button>}
          <button className="btn" disabled={busy} onClick={onDiscard}>Discard</button>
        </div>
      </div>
    );
  }

  return (
    <div className="reading" role="group" aria-label="Check my reading">
      <div className="reading-cols">
        <div><span className="eyebrow">You said</span><div className="said">{pending.original}</div></div>
        <div><span className="eyebrow">Read as</span><div className="read">{pending.restated}</div></div>
      </div>
      <div className="placement">
        <span>
          Classified as <b className={`rel ${relation}`}>{relation === "attack" ? "an attack" : "a support"} to {targetLabel}</b>
          {shaky && <> — only <b className="num">{pending.confidence}%</b> sure of that</>}
        </span>
        {!editing && <button className="btn small ghost change" onClick={() => setEditing(true)}>Change</button>}
        {editing && (
          <div className="placement-edit" style={{ width: "100%" }}>
            <select value={targetId} aria-label="Argument this point bears on" onChange={(e) => setTargetId(e.target.value)}>
              {nodes.map((n) => <option key={n.id} value={n.id}>[{n.id}] {n.text.length > 80 ? `${n.text.slice(0, 79)}…` : n.text}</option>)}
            </select>
            <div className="chips">
              <button className="chip" aria-pressed={relation === "support"} onClick={() => setRelation("support")}>supports it</button>
              <button className="chip" aria-pressed={relation === "attack"} onClick={() => setRelation("attack")}>objects to it</button>
            </div>
          </div>
        )}
      </div>
      <div className="sect">
        <span className="eyebrow">How sure are you of this point?</span>
        <div className="seg-row" role="radiogroup" aria-label="How sure are you of this point?">
          {levels.map((l) => (
            <button key={l.key} className="seg" role="radio" aria-checked={confidence === l.key} disabled={busy} onClick={() => setConfidence(l.key)}>
              <b>{l.label}</b><span>{HINTS[l.key] ?? ""}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="reading-actions">
        <button className="btn user" disabled={!ready} title={confidence === null ? "Pick how sure you are first" : undefined} onClick={() => onConfirm(overrides, confidence)}>Add my point</button>
        <button className="btn" disabled={busy} onClick={onDiscard}>Discard</button>
      </div>
    </div>
  );
}

function IntentCard({ request, onResolve, onDiscard, busy }) {
  return (
    <div className="pending">
      <div className="pending-q">
        Did you mean this as <b>a request</b> — <i>“{request}”</i> — or as a point of your own?
      </div>
      <div className="pending-actions">
        <button className="btn primary" disabled={busy} onClick={() => onResolve("request")}>It's a request</button>
        <button className="btn" disabled={busy} onClick={() => onResolve("argument")}>It's my point</button>
        <button className="btn" disabled={busy} onClick={onDiscard}>Discard</button>
      </div>
    </div>
  );
}

export default function ChatPanel({
  messages, personas, levels, nodes, busy, pending, onSend,
  onConfirmPending, onResolveIntent, onDiscardPending, onFocus,
}) {
  const [draft, setDraft] = useState("");
  /* Who a question or request is addressed to, when several personas sit
     at the table. */
  const [to, setTo] = useState(personas[0]?.id);
  useEffect(() => { if (!personas.some((p) => p.id === to)) setTo(personas[0]?.id); }, [personas, to]);
  const nameOf = (id) => personas.find((p) => p.id === id)?.name ?? "Persona";
  const addressee = nameOf(to);
  const streamRef = useRef(null);
  const labelOf = (key) => levels?.find((c) => c.key === key)?.label?.toLowerCase() ?? key;

  useEffect(() => {
    const el = streamRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, pending, busy]);

  const send = () => {
    const text = draft.trim();
    if (!text || busy || pending) return;
    setDraft("");
    onSend(text, to);
  };

  return (
    <section className="chat">
      <div className="stream" ref={streamRef}>
        {messages.map((m) => (
          <div key={m.id} id={`msg-${m.id}`} className={`msg ${m.who}`}>
            {m.who === "agent" && <span className="eyebrow">{nameOf(m.speaker)}</span>}
            {m.who === "you" && <span className="eyebrow">You{m.confidence ? ` · ${labelOf(m.confidence)}` : ""}</span>}
            {m.text && <p className="msg-body">{m.text}</p>}
            {m.question && <div className="msg-question">{m.question}</div>}
            {m.delta && <StepsCard delta={m.delta} steps={m.steps} onFocus={onFocus} />}
          </div>
        ))}

        {pending && (
          <div className="msg">
            {pending.intent ? (
              <IntentCard request={pending.intent} busy={busy} onResolve={onResolveIntent} onDiscard={onDiscardPending} />
            ) : (
              <ReadingCard
                key={pending.attachment.restated}
                pending={{ ...pending.attachment, original: pending.text }}
                nodes={nodes}
                levels={levels ?? []}
                personaName={personas.length === 1 ? personas[0].name : "The personas"}
                busy={busy}
                onConfirm={onConfirmPending}
                onDiscard={onDiscardPending}
              />
            )}
          </div>
        )}

        {busy && (
          <div className="msg system">
            <p className="msg-body thinking">Working</p>
          </div>
        )}
      </div>

      <div className="composer">
        {personas.length > 1 && (
          <div className="addressee" role="radiogroup" aria-label="Who are you talking to?">
            <span className="eyebrow">To</span>
            <div className="chips">
              {personas.map((p) => (
                <button key={p.id} className="chip" role="radio" aria-checked={to === p.id} aria-pressed={to === p.id} disabled={busy} onClick={() => setTo(p.id)}>{p.name}</button>
              ))}
            </div>
          </div>
        )}
        <div className="composer-row">
          <textarea
            rows={2}
            value={draft}
            placeholder={pending ? "Resolve the open card first…" : `Ask ${addressee} where it stands, or for a reason…`}
            aria-label="Your message"
            disabled={busy || !!pending}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); }
            }}
          />
          <button className="btn primary" style={{ height: 40 }} onClick={send} disabled={busy || !!pending || !draft.trim()}>Send</button>
        </div>
        <span className="hint">Enter sends · Shift+Enter for a new line · to make a point on an argument, right-click it in the map</span>
      </div>
    </section>
  );
}
