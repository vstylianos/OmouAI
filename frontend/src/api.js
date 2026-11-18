/* Hosted demos sit behind a shared passphrase (the backend's DEMO_PASSPHRASE
   gate). The passphrase rides on every request as a header; locally the gate
   is off and the header is simply absent. */
let passphrase = null;
try {
  passphrase = localStorage.getItem("demo_passphrase");
} catch {
  /* storage unavailable — the user just retypes it */
}

const auth = () => (passphrase ? { "X-Demo-Passphrase": passphrase } : {});

const json = async (res) => {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const err = new Error(body.detail || `${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
};

const get = (url) => fetch(url, { headers: auth() }).then(json);

const send = (method) => (url, body) =>
  fetch(url, {
    method,
    headers: { "Content-Type": "application/json", ...auth() },
    body: JSON.stringify(body),
  }).then(json);

const post = send("POST");
const patch = send("PATCH");
const put = send("PUT");

export const api = {
  setPassphrase: (value) => {
    passphrase = value;
    try {
      localStorage.setItem("demo_passphrase", value);
    } catch {
      /* fine — kept in memory for this tab */
    }
  },
  catalogue: () => get("/api/personas"),
  createSession: (domain_id, persona_ids, topic, goals) =>
    post("/api/session", { domain_id, persona_ids, topic, goals }),
  session: (id) => get(`/api/session/${id}`),
  lens: (id) => get(`/api/session/${id}/lens`),
  summary: (id) => get(`/api/session/${id}/summary`),
  retag: (id, node_id, tags) =>
    put(`/api/session/${id}/node/${node_id}/tags`, { tags }),
  expand: (id, node_id, relation_type, persona_id = null) =>
    post(`/api/session/${id}/expand`, { node_id, relation_type, persona_id }),
  contribute: (id, text, { attachment = null, force = null, confidence = null, persona_id = null, target_id = null } = {}) =>
    post(`/api/session/${id}/contribute`, { text, attachment, force, confidence, persona_id, target_id }),
  report: (id) => get(`/api/session/${id}/report`),
  leverage: (id) => get(`/api/session/${id}/leverage`),
  rescore: (id, node_id, base_score) =>
    patch(`/api/session/${id}/node/${node_id}`, { base_score }),
  drop: (id, node_id) =>
    fetch(`/api/session/${id}/node/${node_id}`, {
      method: "DELETE",
      headers: auth(),
    }).then(json),
  settings: (id, body) => patch(`/api/session/${id}/settings`, body),
};
