import { useMemo, useState } from "react";

const KIND_LABEL = { stakeholder: "Stakeholder", expert: "Domain expert", contrarian: "Devil's advocate" };
const KIND_TAG = { stakeholder: "STAKEHOLDER", expert: "EXPERT", contrarian: "CONTRARIAN" };
const KIND_CLASS = { stakeholder: "", expert: "expert", contrarian: "con" };
const initials = (name) =>
  name.split(/[\s-]+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase();

function Check() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M2 6l3 3 5-6" />
    </svg>
  );
}

/* Three steps: case & persona → claim → goals. The persona card follows you
   down the rail. Nothing is generated until Start; the setup costs no calls. */
export default function StartScreen({ domains, sdgs, busy, error, onStart }) {
  const [step, setStep] = useState(1);
  const [domainId, setDomainId] = useState(domains[0]?.id ?? "");
  const domain = domains.find((d) => d.id === domainId) ?? domains[0];
  const [personaIds, setPersonaIds] = useState(() => (domain?.partners[0] ? [domain.partners[0].id] : []));
  const chosen = (domain?.partners ?? []).filter((p) => personaIds.includes(p.id));
  const persona = chosen[0];
  const togglePersona = (id) =>
    setPersonaIds((cur) => (cur.includes(id) ? (cur.length > 1 ? cur.filter((x) => x !== id) : cur) : [...cur, id]));
  const [claim, setClaim] = useState(domain?.claims[0] ?? "");
  const [custom, setCustom] = useState("");
  const [selSdgs, setSelSdgs] = useState(null); // null = not yet suggested
  const [selInds, setSelInds] = useState(new Set());
  const [customs, setCustoms] = useState([]);
  const [customDraft, setCustomDraft] = useState("");

  /* Suggested goals: the SDGs the case's indicators map to. */
  const suggested = useMemo(
    () => new Set((domain?.indicators ?? []).flatMap((i) => i.sdg)),
    [domain]
  );
  const sdgSet = selSdgs ?? suggested;

  const pickDomain = (d) => {
    setDomainId(d.id);
    setPersonaIds(d.partners[0] ? [d.partners[0].id] : []);
    setClaim(d.claims[0] ?? "");
    setSelInds(new Set());
    setSelSdgs(null);
  };
  const toggle = (setter, current) => (id) => {
    const next = new Set(current);
    next.has(id) ? next.delete(id) : next.add(id);
    setter(next);
  };
  const addCustom = () => {
    const text = customDraft.trim();
    if (text && !customs.includes(text)) setCustoms((c) => [...c, text]);
    setCustomDraft("");
  };
  const chosenClaim = custom.trim() || claim;
  const goalCount = sdgSet.size + selInds.size + customs.length;

  const start = () => {
    if (personaIds.length && chosenClaim && !busy)
      onStart(domainId, personaIds, chosenClaim, { sdgs: [...sdgSet], indicators: [...selInds], custom: customs });
  };

  const steps = [
    { n: 1, title: "Case & personas", sub: `${domain?.name ?? ""} · ${chosen.map((p) => p.name).join(", ")}` },
    { n: 2, title: "The claim", sub: step > 2 ? chosenClaim : "Pick one, or write your own" },
    { n: 3, title: "Goals to measure against", sub: step >= 3 ? `${goalCount} selected` : "Suggested from the case; adjust if you like" },
  ];

  return (
    <div className="start">
      <div className="start-top">
        <span className="wordmark">OmouAI</span>
        <span className="tagline">Deliberate a policy claim with various personas.</span>
      </div>

      <div className="start-grid">
        <div className="rail">
          <span className="eyebrow" style={{ marginBottom: 8 }}>Set up the deliberation</span>
          {steps.map((s) => (
            <button
              key={s.n}
              className={`step ${s.n < step ? "done" : s.n === step ? "current" : "todo"}`}
              onClick={() => s.n < step && setStep(s.n)}
              disabled={s.n > step}
            >
              <span className="n">{s.n < step ? <Check /> : s.n}</span>
              <span className="t"><b>{s.title}</b><span title={s.sub}>{s.sub}</span></span>
            </button>
          ))}
          {persona && (
            <div className="persona-card">
              <span className="eyebrow">{chosen.length === 1 ? "Persona" : `${chosen.length} personas at the table`}</span>
              {chosen.map((p) => (
                <div key={p.id} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span className="avatar" style={{ width: 32, height: 32, fontSize: 11 }}>{initials(p.name)}</span>
                  <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.2 }}>
                    <b>{p.name}</b>
                    <span style={{ fontSize: 12, color: "var(--ink-3)" }}>{KIND_LABEL[p.kind] ?? "Stakeholder"}</span>
                  </div>
                </div>
              ))}
              {chosen.length === 1 && <p>“{persona.values[0]}”</p>}
              {step > 1 && <button className="btn ghost small" style={{ alignSelf: "flex-start" }} onClick={() => setStep(1)}>Change personas</button>}
            </div>
          )}
        </div>

        <div className="start-main">
          {step === 1 && (
            <>
              <div>
                <h1>Who do you want to deliberate with?</h1>
                <p className="lede">Pick a case study, then one or more personas: stakeholders with commitments, an expert who argues from the evidence, or the Devil's Advocate, who only ever argues against. Each argument you ask for is made by the persona you choose.</p>
              </div>
              <div className="chips">
                {domains.map((d) => (
                  <button key={d.id} className="chip" aria-pressed={d.id === domainId} onClick={() => pickDomain(d)}>{d.name}</button>
                ))}
              </div>
              {domain?.blurb && <p className="lede" style={{ fontSize: 13 }}>{domain.blurb}</p>}
              <div className="pick-grid">
                {domain?.partners.map((p) => (
                  <button key={p.id} className="pick" aria-pressed={personaIds.includes(p.id)} onClick={() => togglePersona(p.id)}>
                    <div className="meta">
                      <span className="avatar">{initials(p.name)}</span>
                      <span className={`tag ${KIND_CLASS[p.kind] ?? ""}`}>{KIND_TAG[p.kind] ?? "STAKEHOLDER"}</span>
                    </div>
                    <span className="title">{p.name}</span>
                    <span className="sub">{p.one_line}</span>
                  </button>
                ))}
              </div>
              <div className="start-actions">
                <button className="btn primary" onClick={() => setStep(2)}>Continue to the claim →</button>
                <span className="hint">{chosen.length} selected · click to add or remove</span>
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <div>
                <h1>Which claim would you like to deliberate about?</h1>
              </div>
              <div className="pick-grid">
                {domain?.claims.map((c) => (
                  <button key={c} className="pick" aria-pressed={!custom.trim() && c === claim} onClick={() => { setClaim(c); setCustom(""); }}>
                    <span className="title">{c}</span>
                  </button>
                ))}
              </div>
              <div className="sect">
                <span className="eyebrow">Or write your own</span>
                <input type="text" value={custom} placeholder="A policy claim, stated so that one can be for or against it…" onChange={(e) => setCustom(e.target.value)} />
              </div>
              <div className="start-actions">
                <button className="btn" onClick={() => setStep(1)}>← Back</button>
                <button className="btn primary" disabled={!chosenClaim} onClick={() => setStep(3)}>Continue to goals →</button>
              </div>
            </>
          )}

          {step === 3 && (
            <>
              <div>
                <h1>What should the deliberation be measured against?</h1>
                <p className="lede">Goals never enter the argument map. Each one is evaluated from the arguments that bear on it, so you can see whether the case for it is supported, contested, or undermined.</p>
              </div>
              <div className="goal-group">
                <div className="head"><span className="eyebrow">Sustainable Development Goals (United Nations)</span></div>
                <div className="chips">
                  {sdgs.map((s) => (
                    <button key={s.goal} className="chip" aria-pressed={sdgSet.has(s.goal)} onClick={() => toggle(setSelSdgs, sdgSet)(s.goal)}>{s.goal} · {s.name}</button>
                  ))}
                </div>
              </div>
              {domain?.indicators.length > 0 && (
                <div className="goal-group">
                  <div className="head"><span className="eyebrow">Indicators from the law's monitoring framework (Catalan homelessness law)</span></div>
                  <div className="chips">
                    {domain.indicators.map((i) => (
                      <button key={i.id} className="chip" aria-pressed={selInds.has(i.id)} title={i.objective} onClick={() => toggle(setSelInds, selInds)(i.id)}>{i.name}</button>
                    ))}
                  </div>
                </div>
              )}
              <div className="goal-group">
                <div className="head"><span className="eyebrow">In your own words</span></div>
                {customs.length > 0 && (
                  <div className="chips">
                    {customs.map((c) => (
                      <span key={c} className="chip" aria-pressed="true">{c}<button className="tag-x" aria-label={`Remove ${c}`} onClick={() => setCustoms((cur) => cur.filter((x) => x !== c))}>×</button></span>
                    ))}
                  </div>
                )}
                <div className="custom-row">
                  <input type="text" value={customDraft} placeholder="e.g. Families off the waiting list within a year" onChange={(e) => setCustomDraft(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.nativeEvent.isComposing) { e.preventDefault(); addCustom(); } }} />
                  <button className="btn" disabled={!customDraft.trim()} onClick={addCustom}>Add</button>
                </div>
              </div>
              {error && <div className="error">{error}</div>}
              <div className="start-actions">
                <button className="btn" onClick={() => setStep(2)}>← Back</button>
                <button className="btn primary" disabled={busy || !chosenClaim} onClick={start}>{busy ? "Starting…" : "Start the deliberation"}</button>
                <span className="hint">{goalCount} goal{goalCount === 1 ? "" : "s"} · starts neutral, no model call</span>
              </div>
            </>
          )}
        </div>
      </div>

      <footer className="start-foot">
        <img src="/logos/UNU_logo.png" alt="United Nations University" />
        <img src="/logos/NMSU_logo.png" alt="New Mexico State University" />
        <img className="kcl" src="/logos/KCL_logo.webp" alt="King's College London" />
        <img src="/logos/washu_logo.jpeg" alt="Washington University in St. Louis" />
      </footer>
    </div>
  );
}
