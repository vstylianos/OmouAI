import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background, Controls, ReactFlow, useEdgesState, useNodesInitialized,
  useNodesState, useReactFlow,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import ArgumentNode from "./ArgumentNode.jsx";

const nodeTypes = { argument: ArgumentNode };

/* Dagre over measured heights. rankdir BT because our edges run child -> parent,
   so the claim ends up at the top with its reasons beneath it. */
function runDagre(rfNodes, rfEdges, pinned, dims) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "BT", nodesep: 36, ranksep: 64, marginx: 24, marginy: 24 });
  g.setDefaultEdgeLabel(() => ({}));
  rfNodes.forEach((n) => {
    const d = dims.get(n.id);
    g.setNode(n.id, { width: d?.width ?? 280, height: d?.height ?? 120 });
  });
  rfEdges.forEach((e) => g.setEdge(e.source, e.target));
  dagre.layout(g);
  return rfNodes.map((n) => {
    if (pinned.has(n.id)) return n;
    const d = g.node(n.id);
    if (!d) return n;
    return { ...n, position: { x: d.x - d.width / 2, y: d.y - d.height / 2 } };
  });
}

import { NodeMenu, PointPopover } from "./NodeMenu.jsx";

export default function GraphPane({
  state, levels, expert, focusedId, selectedId, onSelect,
  menu, popover, pending, busy, onMenu, onOwnPoint, onAsk, onDrop, onPointSubmit, onConfirmPending, onDiscardPending, onCloseOverlays,
}) {
  const hostRef = useRef(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [pinned, setPinned] = useState(() => new Set());
  const nodesInitialized = useNodesInitialized();
  const { fitView, getNodes } = useReactFlow();
  const lastStructure = useRef("");
  const pendingFit = useRef(false);

  const structure = useMemo(
    () =>
      state.nodes.map((n) => n.id).sort().join(",") + "|" +
      state.edges.map((e) => `${e.source}>${e.target}`).sort().join(","),
    [state.nodes, state.edges]
  );

  useEffect(() => {
    setNodes((current) => {
      const byId = new Map(current.map((n) => [n.id, n]));
      return state.nodes.map((node) => {
        const existing = byId.get(node.id);
        return {
          ...existing,
          id: node.id,
          type: "argument",
          position: existing?.position ?? { x: 0, y: 0 },
          zIndex: selectedId === node.id ? 20 : focusedId === node.id ? 10 : 1,
          data: {
            node,
            goals: state.goals,
            levels,
            expert,
            selected: selectedId === node.id,
            focused: focusedId === node.id,
            pinned: pinned.has(node.id),
            onSelect,
            onMenu,
          },
        };
      });
    });
    setEdges(
      state.edges.map((e) => {
        const attack = e.relation_type === "attack";
        return {
          id: `${e.source}->${e.target}`,
          source: e.source,
          target: e.target,
          type: "smoothstep",
          label: attack ? "−" : "+",
          labelStyle: { fill: attack ? "var(--attack)" : "var(--support)", fontFamily: "var(--mono)", fontWeight: 700, fontSize: 14 },
          labelBgStyle: { fill: "var(--surface)", stroke: attack ? "var(--attack)" : "var(--support)", strokeWidth: 1.5 },
          labelBgPadding: [6, 3],
          labelBgBorderRadius: 4,
          style: { stroke: attack ? "var(--attack)" : "var(--support)", strokeWidth: 2.25 },
        };
      })
    );
  }, [state, levels, expert, focusedId, selectedId, pinned, onSelect, onMenu, setNodes, setEdges]);

  const measuredDims = useCallback(
    () => new Map(getNodes().map((n) => [n.id, { width: n.measured?.width ?? n.width ?? 280, height: n.measured?.height ?? n.height ?? 120 }])),
    [getNodes]
  );

  useEffect(() => {
    if (!nodesInitialized || structure === lastStructure.current) return;
    lastStructure.current = structure;
    setNodes((nds) => runDagre(nds, state.edges, pinned, measuredDims()));
    pendingFit.current = true;
  }, [nodesInitialized, structure, state.edges, pinned, setNodes, measuredDims]);

  /* Expert mode changes node heights: re-run the layout once measured. */
  useEffect(() => {
    lastStructure.current = "";
  }, [expert]);

  useEffect(() => {
    if (!pendingFit.current || !nodesInitialized) return;
    pendingFit.current = false;
    const id = setTimeout(() => fitView({ padding: 0.2, duration: 340, maxZoom: 1 }), 0);
    return () => clearTimeout(id);
  }, [nodes, nodesInitialized, fitView]);

  const onNodeDragStop = useCallback((_, node) => setPinned((cur) => new Set(cur).add(node.id)), []);

  const tidy = useCallback(() => {
    setPinned(new Set());
    setNodes((nds) => runDagre(nds, state.edges, new Set(), measuredDims()));
    pendingFit.current = true;
  }, [state.edges, setNodes, measuredDims]);

  useEffect(() => {
    if (!focusedId) return;
    if (nodes.some((x) => x.id === focusedId))
      fitView({ nodes: [{ id: focusedId }], padding: 0.55, duration: 420, maxZoom: 1.2 });
  }, [focusedId]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="stage-body" ref={hostRef}>
      <div className="flow-wrap">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeDragStop={onNodeDragStop}
          onPaneClick={() => { onSelect(null); onCloseOverlays(); }}
          onPaneContextMenu={(e) => e.preventDefault()}
          nodesConnectable={false}
          elementsSelectable={false}
          minZoom={0.25}
          maxZoom={1.6}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={22} size={1} color="var(--line-strong)" />
          <Controls showInteractive={false} />
        </ReactFlow>
        <button className="btn small" style={{ position: "absolute", right: 12, top: 10 }} onClick={tidy}>
          Tidy up{pinned.size ? ` · ${pinned.size} pinned` : ""}
        </button>
        <NodeMenu menu={menu} personas={state.personas} busy={busy} onOwnPoint={onOwnPoint} onAsk={onAsk} onDrop={onDrop} onClose={onCloseOverlays} />
        <PointPopover
          popover={popover} nodes={state.nodes} levels={levels} personas={state.personas}
          pending={pending?.origin === "node" ? pending : null} busy={busy}
          onSubmit={onPointSubmit} onConfirm={onConfirmPending} onDiscard={onDiscardPending} onClose={onCloseOverlays}
        />
      </div>
      {!state.is_tree && (
        <div className="stage-note warn">
          An argument now reaches the claim by more than one path, so it cuts both ways —
          contesting it has no guaranteed direction of effect.
        </div>
      )}
      <div className="stage-note">
        <b>Right-click an argument</b> to make your own point on it or to ask a persona for a reason
        for or against it. Click to inspect. Bars: how strongly each argument holds now; the dark
        tick is the confidence it was given.
      </div>
    </div>
  );
}
