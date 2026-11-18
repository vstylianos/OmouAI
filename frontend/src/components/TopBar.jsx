const KIND_TAG = { stakeholder: "PERSONA", expert: "EXPERT", contrarian: "DEVIL'S ADVOCATE" };
const KIND_CLASS = { stakeholder: "", expert: "expert", contrarian: "con" };
const initials = (name) =>
  name.split(/[\s-]+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase();

function Gear() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <circle cx="8" cy="8" r="2.5" />
      <path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4" />
    </svg>
  );
}

function Lines() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path d="M2 3h12M2 8h12M2 13h8" />
    </svg>
  );
}

/* Identity and navigation only. Engine settings live in the drawer. */
export default function TopBar({ state, onOpenSummary, onOpenSettings }) {
  const { personas, usage } = state;
  return (
    <header className="topbar">
      <span className="wordmark">OmouAI</span>
      <div className="topbar-claim">
        <span className="tag claim">CLAIM</span>
        <span className="text" title={state.topic}>{state.topic}</span>
      </div>
      <div className="persona-pills">
        {personas.map((persona) => (
          <div key={persona.id} className="persona-pill" title={persona.one_line}>
            <span className="avatar">{initials(persona.name)}</span>
            <span className="name">{persona.name}</span>
            {personas.length === 1 && <span className={`tag ${KIND_CLASS[persona.kind] ?? ""}`}>{KIND_TAG[persona.kind] ?? "PERSONA"}</span>}
          </div>
        ))}
      </div>
      <button className="btn" onClick={onOpenSummary}><Lines /> Summary</button>
      <button className="btn icon" onClick={onOpenSettings} aria-label="Settings" title="Settings"><Gear /></button>
      <span className="usage" title="Model calls and tokens this session">
        {usage.calls} calls · {((usage.prompt_tokens + usage.completion_tokens) / 1000).toFixed(1)}k tok
      </span>
    </header>
  );
}
