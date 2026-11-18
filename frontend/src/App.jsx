import { useCallback, useEffect, useRef, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { api } from "./api.js";
import ChatPanel from "./components/ChatPanel.jsx";
import GraphPane from "./components/GraphPane.jsx";
import GoalsTab from "./components/GoalsTab.jsx";
import Inspector from "./components/Inspector.jsx";
import PositionMeter from "./components/PositionMeter.jsx";
import SettingsDrawer from "./components/SettingsDrawer.jsx";
import StanceTab from "./components/StanceTab.jsx";
import StartScreen from "./components/StartScreen.jsx";
import SummaryView from "./components/SummaryView.jsx";
import TopBar from "./components/TopBar.jsx";

/* The deliberation survives a reload: the session id, the chat, and any open
   card are kept in the browser, and the live session is fetched back from
   the server. A backend restart still drops sessions. */
const STORE = "deliberation.session";
const loadSaved = () => { try { return JSON.parse(localStorage.getItem(STORE)); } catch { return null; } };
const save = (value) => {
  try { value ? localStorage.setItem(STORE, JSON.stringify(value)) : localStorage.removeItem(STORE); } catch { /* fine */ }
};
const pref = (key, fallback) => { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } };
const setPref = (key, value) => { try { localStorage.setItem(key, value); } catch { /* fine */ } };

export default function App() {
  const [catalogue, setCatalogue] = useState(null);
  const [state, setState] = useState(null);
  const [messages, setMessages] = useState([]);
  /* A pending card: {attachment, text, followup, origin: "chat"|"node"} or
     {intent, text}. Node-origin cards render in the popover on the map. */
  const [pending, setPending] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [focusedId, setFocusedId] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [menu, setMenu] = useState(null);        // {nodeId, x, y} — the right-click menu
  const [popover, setPopover] = useState(null);  // {nodeId, x, y} — writing your own point
  const [tab, setTab] = useState("arguments");
  const [showSummary, setShowSummary] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [expert, setExpert] = useState(() => pref("deliberation.expert", "0") === "1");
  const [theme, setTheme] = useState(() => pref("deliberation.theme", "light"));
  const [gate, setGate] = useState(null);
  const msgId = useRef(0);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    setPref("deliberation.theme", theme);
  }, [theme]);
  useEffect(() => { setPref("deliberation.expert", expert ? "1" : "0"); }, [expert]);

  const restore = useCallback(async () => {
    const saved = loadSaved();
    if (!saved?.id) return;
    try {
      const next = await api.session(saved.id);
      const msgs = saved.messages ?? [];
      msgId.current = msgs.reduce((m, x) => Math.max(m, x.id), 0);
      setMessages(msgs);
      setPending(saved.pending?.origin === "node" ? null : saved.pending ?? null);
      setState(next);
    } catch {
      save(null);
    }
  }, []);

  const loadCatalogue = useCallback((retry = false) => {
    api.catalogue()
      .then((c) => { setGate(null); setCatalogue(c); restore(); })
      .catch((e) => { if (e.status === 401) setGate(retry ? "wrong" : "ask"); else setError(e.message); });
  }, [restore]);

  useEffect(() => { loadCatalogue(); }, [loadCatalogue]);
  useEffect(() => { if (state) save({ id: state.session_id, messages, pending }); }, [state, messages, pending]);

  const push = useCallback((msg) => {
    msgId.current += 1;
    setMessages((cur) => [...cur, { id: msgId.current, ...msg }]);
  }, []);

  const nameOf = useCallback((id) => state?.personas.find((p) => p.id === id)?.name ?? "Persona", [state]);

  const apply = useCallback((next, { speak = true, text = "", origin = "chat" } = {}) => {
    setState(next);
    const turn = next.turn;
    if (!turn) return;
    if (turn.pending) { setPending({ attachment: turn.pending, text, origin, followup: turn.followup_request ?? null }); return; }
    if (turn.pending_intent) { setPending({ intent: turn.pending_intent, text }); return; }
    setPending(null);
    setPopover(null);
    if (speak) {
      if (turn.reply) {
        const moved = turn.added_nodes?.length > 0 || Math.abs(turn.delta?.change ?? 0) > 1e-9;
        push({ who: "agent", speaker: turn.speaker, text: turn.reply, question: turn.question, delta: moved ? turn.delta : null, steps: turn.steps, nodes: turn.added_nodes });
      } else if (turn.added_nodes?.length && turn.steps?.length) {
        const id = turn.added_nodes[0];
        const node = next.nodes.find((n) => n.id === id);
        const where = node?.parent_id === "topic" ? "the claim" : node?.parent_id;
        push({ who: "system", text: `Added your point ${id}, ${node?.relation_type === "attack" ? "against" : "for"} ${where}.`, delta: turn.delta, steps: turn.steps, nodes: turn.added_nodes });
      }
    }
    if (turn.added_nodes?.length) { setSelectedId(turn.added_nodes[0]); setTab("arguments"); }
  }, [push]);

  const guard = useCallback(async (fn) => {
    setBusy(true); setError("");
    try { return await fn(); }
    catch (e) { setError(e.message); push({ who: "system", text: `Something went wrong: ${e.message}` }); return null; }
    finally { setBusy(false); }
  }, [push]);

  const start = (domainId, personaIds, topic, goals) =>
    guard(async () => {
      const next = await api.createSession(domainId, personaIds, topic, goals);
      setMessages([]); msgId.current = 0; setSelectedId(null); setTab("arguments");
      setState(next);
      for (const intro of next.turn.intros) push({ who: "agent", speaker: intro.persona_id, text: intro.text });
    });

  /* Chat: questions and requests, addressed to one persona. */
  const send = (text, personaId) =>
    guard(async () => {
      push({ who: "you", text });
      apply(await api.contribute(state.session_id, text, { persona_id: personaId }), { text });
    });

  /* Right-click → your own point on a node: the text goes for a direction
     check with the target fixed; the reading card comes back into the popover. */
  const pointSubmit = (nodeId, text) =>
    guard(async () => {
      push({ who: "you", text });
      apply(await api.contribute(state.session_id, text, { target_id: nodeId }), { text, origin: "node" });
    });

  const confirmPending = (overrides = null, confidence = null) =>
    guard(async () => {
      const { attachment, text, followup, origin } = pending;
      const final = overrides ? { ...attachment, ...overrides } : attachment;
      if (confidence) {
        setMessages((cur) => {
          const i = cur.map((m) => m.who === "you" && m.text === text).lastIndexOf(true);
          return i < 0 ? cur : cur.map((m, j) => (j === i ? { ...m, confidence } : m));
        });
      }
      apply(await api.contribute(state.session_id, text, { attachment: final, confidence }), { text, origin });
      if (followup) apply(await api.contribute(state.session_id, followup, { force: "request" }), { text: followup });
    });

  const resolveIntent = (kind) =>
    guard(async () => {
      const { text } = pending;
      apply(await api.contribute(state.session_id, text, { force: kind }), { text });
    });

  const discardPending = () => {
    const followup = pending?.followup;
    setPending(null); setPopover(null);
    push({ who: "system", text: "Discarded — that point was not added." });
    if (followup) guard(async () => apply(await api.contribute(state.session_id, followup, { force: "request" }), { text: followup }));
  };

  /* Ask a named persona for a reason for or against a node. */
  const ask = useCallback((nodeId, relationType, personaId) =>
    guard(async () => {
      setMenu(null);
      const next = await api.expand(state.session_id, nodeId, relationType, personaId);
      apply(next, { speak: false });
      const direction = relationType === "support" ? "for" : "against";
      const where = nodeId === "topic" ? "the claim" : nodeId;
      const who = nameOf(personaId ?? next.turn.speaker);
      push({ who: "system", text: next.turn.added_nodes.length
        ? `${who} gave a reason ${direction} ${where}: ${next.turn.added_nodes[0]}.`
        : `${who} has nothing new ${direction} ${where} — nothing was added.`, delta: next.turn.delta, nodes: next.turn.added_nodes });
    }), [guard, apply, push, state, nameOf]);

  const rescore = useCallback((nodeId, value) =>
    guard(async () => apply(await api.rescore(state.session_id, nodeId, value), { speak: false })), [guard, apply, state]);

  const retag = useCallback((nodeId, tags) =>
    guard(async () => { apply(await api.retag(state.session_id, nodeId, tags), { speak: false }); }), [guard, apply, state]);

  const drop = useCallback((nodeId) =>
    guard(async () => {
      apply(await api.drop(state.session_id, nodeId), { speak: false });
      setSelectedId(null);
      push({ who: "system", text: `Removed ${nodeId} from the map.` });
    }), [guard, apply, push, state]);

  const settings = (body) => guard(async () => setState(await api.settings(state.session_id, body)));

  const focus = useCallback((nodeId) => {
    setTab("arguments");
    setSelectedId(nodeId);
    setFocusedId(nodeId);
    setTimeout(() => setFocusedId((cur) => (cur === nodeId ? null : cur)), 2400);
  }, []);

  const select = useCallback((nodeId) => setSelectedId(nodeId), []);

  /* Overlay positions are relative to the stage; the menu opens where the
     pointer is, the popover where the menu was. */
  const openMenu = useCallback((nodeId, clientX, clientY) => {
    const host = document.querySelector(".stage-body")?.getBoundingClientRect();
    const x = Math.max(8, Math.min(clientX - (host?.left ?? 0), (host?.width ?? 800) - 300));
    const y = Math.max(8, Math.min(clientY - (host?.top ?? 0), (host?.height ?? 600) - 240));
    setPopover(null);
    setMenu({ nodeId, x, y });
  }, []);
  const openPopover = useCallback((nodeId, at) => {
    setMenu(null);
    setPopover({ nodeId, x: Math.min(at.x, 9999), y: at.y });
  }, []);
  const closeOverlays = useCallback(() => { setMenu(null); if (!pending || pending.origin !== "node") setPopover(null); }, [pending]);

  const reset = () => {
    save(null);
    setState(null); setMessages([]); setPending(null); setError("");
    setSelectedId(null); setMenu(null); setPopover(null);
    setShowSummary(false); setSettingsOpen(false); setTab("arguments");
  };

  if (!catalogue) {
    return (
      <div className="app"><div className="center">
        {gate ? (
          <form className="gate" onSubmit={(e) => { e.preventDefault(); api.setPassphrase(new FormData(e.target).get("passphrase").trim()); loadCatalogue(true); }}>
            <span className="wordmark">OmouAI</span>
            <p style={{ margin: 0, color: "var(--ink-3)" }}>This demo is shared by invitation.</p>
            <input name="passphrase" type="password" placeholder="Passphrase" autoFocus />
            <button type="submit" className="btn primary">Enter</button>
            {gate === "wrong" && <div className="error">That passphrase was not accepted.</div>}
          </form>
        ) : error ? <div className="error">{error}</div> : <p style={{ color: "var(--ink-3)" }}>Loading…</p>}
      </div></div>
    );
  }

  if (!state) {
    return <div className="app"><StartScreen domains={catalogue.domains} sdgs={catalogue.sdgs} busy={busy} error={error} onStart={start} /></div>;
  }

  if (showSummary) {
    return <SummaryView sessionId={state.session_id} onClose={() => setShowSummary(false)} />;
  }

  const levels = catalogue.confidence_levels;
  const chatPending = pending && pending.origin !== "node" ? pending : null;

  return (
    <div className="app">
      <TopBar state={state} onOpenSummary={() => setShowSummary(true)} onOpenSettings={() => setSettingsOpen(true)} />
      <div className="body">
        <ChatPanel
          messages={messages} personas={state.personas} levels={levels} nodes={state.nodes}
          busy={busy} pending={chatPending} onSend={send} onConfirmPending={confirmPending}
          onResolveIntent={resolveIntent} onDiscardPending={discardPending} onFocus={focus}
        />
        <section className="stage">
          <PositionMeter state={state} />
          <div className="tabs" role="tablist">
            <button className="tab" role="tab" aria-selected={tab === "arguments"} onClick={() => setTab("arguments")}>
              Arguments <span className="count">{state.nodes.length - 1}</span>
            </button>
            <button className="tab" role="tab" aria-selected={tab === "goals"} onClick={() => setTab("goals")}>
              Goal evaluation
              {state.lens.length > 0 && (
                <span className="count">
                  {[
                    [state.lens.filter((e) => e.verdict === "supported").length, "↑"],
                    [state.lens.filter((e) => e.verdict === "undermined").length, "↓"],
                  ].filter(([n]) => n > 0).map(([n, a]) => `${n}${a}`).join(" ") || state.lens.length}
                </span>
              )}
            </button>
            <button className="tab" role="tab" aria-selected={tab === "stance"} onClick={() => setTab("stance")}>Where we stand</button>
            {tab === "arguments" && (
              <div className="legend">
                <span><i className="sw" />supports</span>
                <span><i className="sw att" />attacks</span>
                <span><i className="dot" />yours</span>
              </div>
            )}
          </div>
          {tab === "arguments" && (
            <ReactFlowProvider>
              <GraphPane
                state={state} levels={levels} expert={expert} focusedId={focusedId} selectedId={selectedId} onSelect={select}
                menu={menu} popover={popover} pending={pending} busy={busy}
                onMenu={openMenu} onOwnPoint={openPopover} onAsk={ask} onDrop={drop} onPointSubmit={pointSubmit}
                onConfirmPending={confirmPending} onDiscardPending={discardPending} onCloseOverlays={closeOverlays}
              />
            </ReactFlowProvider>
          )}
          {tab === "goals" && <GoalsTab state={state} />}
          {tab === "stance" && <StanceTab state={state} onFocus={focus} />}
        </section>
        <Inspector
          state={state} selectedId={selectedId} expert={expert} busy={busy} levels={levels}
          onSelect={select} onRescore={rescore} onRetag={retag} onDrop={drop}
          onOpenMenu={(nodeId, x, y) => { setTab("arguments"); openMenu(nodeId, x, y); }}
          onToggleExpert={() => setExpert((v) => !v)}
        />
      </div>
      {settingsOpen && (
        <SettingsDrawer
          state={state} busy={busy} expert={expert} theme={theme}
          onSettings={settings} onToggleExpert={() => setExpert((v) => !v)} onTheme={setTheme}
          onReset={reset} onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  );
}
