import { useEffect, useMemo, useRef, useState } from "react";
import {
  Background, Controls, Handle, Position, ReactFlow, ReactFlowProvider,
  useEdgesState, useNodesInitialized, useNodesState, useReactFlow,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import { api } from "../api.js";

/* The evaluation lens, materialised inside the stage.

   Default view: one small tree per goal — the goal on top, the arguments
   that bear on it beneath, an argument repeated under every goal it bears
   on. No edge ever crosses another, which is what makes each verdict
   readable. The shared view draws each argument once, with every goal it
   feeds; hovering a goal there fades everything else. Read-only: to change a
   verdict, contest the arguments in the map. */

const verdictColour = (v) => (v === "supported" ? "var(--support)" : v === "undermined" ? "var(--attack)" : "var(--goal)");

function GoalNode({ data }) {
  const { entry, dim } = data;
  const colour = verdictColour(entry.verdict);
  return (
    <div className={`lens-goal ${entry.verdict}`} style={{ opacity: dim ? 0.25 : 1 }}>
      <Handle type="target" position={Position.Bottom} isConnectable={false} />
      <div className="node-head">
        <span className="tag goal">{entry.kind.toUpperCase()}</span>
        <span className="rel" style={{ color: colour }}>{entry.verdict}</span>
        <span className="id">{Math.round(entry.sigma * 100)}%</span>
      </div>
      <div className="node-text" style={{ fontWeight: 500 }}>{entry.label}</div>
      <div className="bar"><div className="fill" style={{ width: `${entry.sigma * 100}%`, background: colour }} /><div className="neutral" /></div>
      <div className="node-foot">
        <span>{entry.contributions.length} argument{entry.contributions.length === 1 ? "" : "s"} bear on it</span>
      </div>
    </div>
  );
}

function LensArgNode({ data }) {
  const { arg, dim } = data;
  return (
    <div className="lens-arg" style={{ opacity: dim ? 0.25 : 1 }}>
      <Handle type="source" position={Position.Top} isConnectable={false} />
      <div className="node-head">
        <span className="id" style={{ marginLeft: 0 }}>{arg.id}</span>
        <span className="id">holds at {Math.round(arg.sigma * 100)}%</span>
      </div>
      <div className="node-text">{arg.text}</div>
    </div>
  );
}

const nodeTypes = { goal: GoalNode, lensArg: LensArgNode };

/* Edge styling; `dim` fades edges outside the hovered goal's neighbourhood. */
function styleEdge(edge, dim) {
  const positive = edge.data.direction === "positive";
  const colour = positive ? "var(--support)" : "var(--attack)";
  return {
    ...edge,
    label: positive ? "+" : "−",
    labelStyle: { fill: colour, fontFamily: "var(--mono)", fontWeight: 700, fontSize: 14, opacity: dim ? 0.25 : 1 },
    labelBgStyle: { fill: "var(--surface)", stroke: colour, strokeWidth: 1.5, opacity: dim ? 0.25 : 1 },
    labelBgPadding: [6, 3],
    labelBgBorderRadius: 4,
    style: { stroke: colour, strokeWidth: 2.25, opacity: dim ? 0.15 : 1 },
  };
}

/* Card widths are fixed by CSS; heights grow with the text, so the layout
   runs on measured sizes once the cards exist (these are the first-paint
   fallbacks, as in the argument map). */
const GOAL_W = 260, GOAL_H = 118, ARG_W = 220, ARG_H = 88;
const sizeOf = (dims, id, w, h) => dims.get(id) ?? { width: w, height: h };

/* One tree per goal: goal on top, its arguments beneath, arguments repeated.
   Each tree is laid out on its own, then the trees are packed into rows so
   the whole reads at a useful zoom rather than as one long strip. */
function layoutGrouped(lens, dims) {
  const trees = lens.map((entry) => {
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "BT", nodesep: 22, ranksep: 80, marginx: 0, marginy: 0 });
    g.setDefaultEdgeLabel(() => ({}));
    const gid = `goal:${entry.id}`;
    g.setNode(gid, sizeOf(dims, gid, GOAL_W, GOAL_H));
    const nodes = [{ id: gid, type: "goal", position: { x: 0, y: 0 }, data: { entry, goalId: entry.id, dim: false } }];
    const edges = [];
    for (const c of entry.contributions) {
      const aid = `${entry.id}|${c.id}`;
      g.setNode(aid, sizeOf(dims, aid, ARG_W, ARG_H));
      nodes.push({ id: aid, type: "lensArg", position: { x: 0, y: 0 }, data: { arg: c, goalIds: [entry.id], dim: false } });
      g.setEdge(aid, gid);
      edges.push(styleEdge({ id: `${aid}->${gid}`, source: aid, target: gid, data: { direction: c.direction, goalId: entry.id } }, false));
    }
    dagre.layout(g);
    let maxX = 0, maxY = 0;
    const placed = nodes.map((n) => {
      const d = g.node(n.id);
      const x = d.x - d.width / 2, y = d.y - d.height / 2;
      maxX = Math.max(maxX, x + d.width); maxY = Math.max(maxY, y + d.height);
      return { ...n, position: { x, y } };
    });
    return { nodes: placed, edges, w: maxX, h: maxY };
  });
  /* Pack into rows: roughly square, so a handful of goals read as a grid. */
  const cols = Math.max(1, Math.ceil(Math.sqrt(trees.length)));
  const gapX = 70, gapY = 90;
  const nodes = [], edges = [];
  let rowY = 0;
  for (let i = 0; i < trees.length; i += cols) {
    const row = trees.slice(i, i + cols);
    let x = 0;
    const rowH = Math.max(...row.map((t) => t.h));
    for (const t of row) {
      for (const n of t.nodes) nodes.push({ ...n, position: { x: n.position.x + x, y: n.position.y + rowY } });
      edges.push(...t.edges);
      x += t.w + gapX;
    }
    rowY += rowH + gapY;
  }
  return { nodes, edges };
}

/* Each argument once, every goal it feeds. */
function layoutShared(lens, dims) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "BT", nodesep: 40, ranksep: 140, marginx: 24, marginy: 24 });
  g.setDefaultEdgeLabel(() => ({}));
  const args = new Map();
  for (const entry of lens) for (const c of entry.contributions) {
    if (!args.has(c.id)) args.set(c.id, { arg: c, goalIds: [] });
    args.get(c.id).goalIds.push(entry.id);
  }
  const nodes = [];
  const edges = [];
  for (const entry of lens) {
    const gid = `goal:${entry.id}`;
    g.setNode(gid, sizeOf(dims, gid, GOAL_W, GOAL_H));
    nodes.push({ id: gid, type: "goal", position: { x: 0, y: 0 }, data: { entry, goalId: entry.id, dim: false } });
  }
  for (const { arg, goalIds } of args.values()) {
    g.setNode(arg.id, sizeOf(dims, arg.id, ARG_W, ARG_H));
    nodes.push({ id: arg.id, type: "lensArg", position: { x: 0, y: 0 }, data: { arg, goalIds, dim: false } });
  }
  for (const entry of lens)
    for (const c of entry.contributions) {
      g.setEdge(c.id, `goal:${entry.id}`);
      edges.push(styleEdge({ id: `${c.id}->${entry.id}`, source: c.id, target: `goal:${entry.id}`, data: { direction: c.direction, goalId: entry.id } }, false));
    }
  dagre.layout(g);
  return {
    nodes: nodes.map((n) => { const d = g.node(n.id); return { ...n, position: { x: d.x - d.width / 2, y: d.y - d.height / 2 } }; }),
    edges,
  };
}

const layout = (lens, mode, dims) => (mode === "grouped" ? layoutGrouped(lens, dims) : layoutShared(lens, dims));

/* The flow itself. Lays out once on fallback sizes so the first paint is
   sensible, then again on measured sizes so no card overlaps another.
   Hovering a goal in the shared view only changes what is dimmed, never
   where anything sits. */
function LensFlow({ grounded, mode }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [activeGoal, setActiveGoal] = useState(null);
  const nodesInitialized = useNodesInitialized();
  const { fitView, getNodes } = useReactFlow();
  const measuredFor = useRef("");
  const structure = useMemo(
    () => mode + "|" + grounded.map((e) => `${e.id}:${e.contributions.map((c) => c.id).join(",")}`).join(";"),
    [grounded, mode]
  );

  useEffect(() => {
    const { nodes, edges } = layout(grounded, mode, new Map());
    setNodes(nodes); setEdges(edges);
    measuredFor.current = "";
  }, [grounded, mode, setNodes, setEdges]);

  useEffect(() => {
    if (!nodesInitialized || measuredFor.current === structure) return;
    measuredFor.current = structure;
    const dims = new Map(getNodes().map((n) => [n.id, {
      width: n.measured?.width ?? n.width ?? ARG_W, height: n.measured?.height ?? n.height ?? ARG_H,
    }]));
    const { nodes, edges } = layout(grounded, mode, dims);
    setNodes(nodes); setEdges(edges);
    const id = setTimeout(() => fitView({ padding: 0.08, maxZoom: 1.1 }), 0);
    return () => clearTimeout(id);
  }, [nodesInitialized, structure, grounded, mode, getNodes, setNodes, setEdges, fitView]);

  useEffect(() => {
    setNodes((nds) => nds.map((n) => {
      const dim = !!activeGoal && (n.type === "goal" ? n.data.goalId !== activeGoal : !n.data.goalIds.includes(activeGoal));
      return dim === n.data.dim ? n : { ...n, data: { ...n.data, dim } };
    }));
    setEdges((eds) => eds.map((e) => styleEdge(e, !!activeGoal && e.data.goalId !== activeGoal)));
  }, [activeGoal, setNodes, setEdges]);

  return (
    <ReactFlow
      nodes={nodes} edges={edges} nodeTypes={nodeTypes}
      onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
      nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}
      onNodeMouseEnter={(_, n) => mode === "shared" && n.type === "goal" && setActiveGoal(n.data.goalId)}
      onNodeMouseLeave={() => setActiveGoal(null)}
      minZoom={0.2} maxZoom={1.5} proOptions={{ hideAttribution: true }}
    >
      <Background gap={22} size={1} color="var(--line-strong)" />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

export default function GoalsTab({ state }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("grouped");
  useEffect(() => {
    let live = true;
    api.lens(state.session_id).then((d) => live && setData(d)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [state]);
  const grounded = useMemo(() => (data ? data.lens.filter((e) => e.contributions.length > 0) : []), [data]);

  if (!state.goals?.length) {
    return <div className="stage-body"><div className="inspector-empty">No goals were selected for this deliberation.</div></div>;
  }
  return (
    <div className="stage-body">
      <div className="verdicts">
        {state.lens.map((e) => (
          <span key={e.id} className={`verdict ${e.verdict}`} title={`σ ${e.sigma.toFixed(2)}`}>
            <span className="lbl">{e.label}</span>
            <span className="v">{e.verdict === "supported" ? "↑" : e.verdict === "undermined" ? "↓" : "·"} {e.verdict}</span>
          </span>
        ))}
        {grounded.length > 0 && (
          <div className="chips" style={{ marginLeft: "auto" }}>
            <button className="chip" aria-pressed={mode === "grouped"} onClick={() => setMode("grouped")} title="One tree per goal; an argument is repeated under every goal it bears on">By goal</button>
            <button className="chip" aria-pressed={mode === "shared"} onClick={() => setMode("shared")} title="Each argument once; hover a goal to see only what bears on it">Shared</button>
          </div>
        )}
      </div>
      {error && <div className="error" style={{ padding: 16 }}>{error}</div>}
      {data && grounded.length === 0 && (
        <div className="inspector-empty">
          No argument bears on a goal yet, so every goal reads as contested at a neutral 50%. Tag an
          argument with a goal in the inspector, or ask a persona for reasons — its arguments come tagged.
        </div>
      )}
      {grounded.length > 0 && (
        <div className="flow-wrap">
          <ReactFlowProvider>
            <LensFlow key={mode} grounded={grounded} mode={mode} />
          </ReactFlowProvider>
        </div>
      )}
      <div className="stage-note">
        Each goal is evaluated as a virtual node under the same semantics as the position: the arguments
        tagged with it support (+, green) or strain (−, red) it at their computed strength, read against a
        neutral 50%. Tags are judged by the persona that made the argument, or asserted by you. To change
        a verdict, contest the arguments or their tags in the map.
      </div>
    </div>
  );
}
