import { Handle, Position } from "@xyflow/react";

const TYPE_LABEL = { empirical: "EMP", predictive: "PRD", normative: "NRM" };
const LEVEL_LABEL = { unsure: "unsure", somewhat: "somewhat sure", moderately: "moderately sure", very: "very sure", certain: "certain" };

const goalShort = (goalId, goals) => {
  if (goalId.startsWith("sdg_")) return `SDG ${goalId.slice(4)}`;
  const label = goals?.find((g) => g.id === goalId)?.label ?? goalId;
  return label.length > 18 ? `${label.slice(0, 17)}…` : label;
};

/* Nearest scale level for a user node's tau, for the plain-language footer. */
const levelOf = (tau, levels) => {
  if (!levels?.length) return null;
  return levels.reduce((best, l) => (Math.abs(l.tau - tau) < Math.abs(best.tau - tau) ? l : best));
};

/* Compact by default: who, the text, one bar. Symbols and types only in
   expert mode. Everything else lives in the inspector once selected. */
export default function ArgumentNode({ id, data }) {
  const { node, goals, levels, expert, selected, focused, pinned, onSelect, onMenu } = data;
  const mine = node.provenance === "user";
  const fill = node.is_topic
    ? node.strength > 0.55 ? "var(--support)" : node.strength < 0.45 ? "var(--attack)" : "var(--ink-3)"
    : mine ? "var(--user)"
    : node.relation_type === "attack" ? "var(--attack)" : "var(--support)";
  const gap = node.strength - node.base_score;
  const rel = node.is_topic ? null
    : node.parent_id === "topic"
      ? node.relation_type === "attack" ? "against the claim" : "for the claim"
      : node.relation_type === "attack" ? `objects to ${node.parent_id}` : `supports ${node.parent_id}`;
  const level = mine ? levelOf(node.base_score, levels) : null;

  return (
    <div
      className={[
        "node",
        node.is_topic && "topic",
        mine && "from-user",
        selected && "selected",
        focused && "focused",
      ].filter(Boolean).join(" ")}
      onClick={() => onSelect(id)}
      onContextMenu={(e) => { e.preventDefault(); onSelect(id); onMenu(id, e.clientX, e.clientY); }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(id); } }}
    >
      <Handle type="source" position={Position.Top} isConnectable={false} />
      <Handle type="target" position={Position.Bottom} isConnectable={false} />

      <div className="node-head">
        {node.is_topic ? <span className="tag claim">CLAIM</span>
          : mine ? <span className="tag you">YOURS</span>
          : <span className="tag">{(node.author_name ?? "persona").toUpperCase()}</span>}
        {rel && <span className={`rel ${node.relation_type}`}>{rel}</span>}
        {expert && !node.is_topic && <span className={`tag ${node.polarity}`}>{node.polarity.toUpperCase()}</span>}
        {expert && <span className="tag">{TYPE_LABEL[node.node_type]}</span>}
        {pinned && <span className="pin-dot" title="Pinned — auto-layout leaves it here" />}
        <span className="id">{node.is_topic ? `${Math.round(node.strength * 100)}%` : id}</span>
      </div>

      <div className="node-text">{node.text}</div>

      <div className="bar" title={node.is_topic ? "The position, computed from every argument" : "How strongly this holds now; the tick is the confidence it was given"}>
        <div className="fill" style={{ width: `${node.strength * 100}%`, background: fill }} />
        {node.is_topic ? <div className="neutral" /> : <div className="tick" style={{ left: `${node.base_score * 100}%` }} />}
      </div>

      {!node.is_topic && (
        <div className="node-foot">
          {mine
            ? <span>Rated <b>{level ? LEVEL_LABEL[level.key] : `${Math.round(node.base_score * 100)}%`}</b></span>
            : <span>Its confidence <b className="mono">{Math.round(node.base_score * 100)}%</b></span>}
          <span>holds at <b className="mono">{Math.round(node.strength * 100)}%</b></span>
          {gap < -0.05 && <span className="contested">contested</span>}
          {gap > 0.05 && <span className="lifted">reinforced</span>}
        </div>
      )}

      {expert && !node.is_topic && (
        <div className="node-foot mono">
          <span>τ {node.base_score.toFixed(2)}</span>
          <span>σ {node.strength.toFixed(2)}</span>
          {node.goal_impacts?.map((t) => (
            <span key={t.goal_id} className={`tag goal ${t.source === "user" ? "user-src" : t.source ?? "judged"}`}>
              {t.direction === "positive" ? "↑" : "↓"} {goalShort(t.goal_id, goals)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
