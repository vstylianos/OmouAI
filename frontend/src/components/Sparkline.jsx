/* Stance across turns. The trajectory is the point of the demo — a single
   current reading undersells that you moved it. */
export default function Sparkline({ history, width = 164, height = 46 }) {
  const pts = history.map((h) => h.stance);
  if (pts.length === 0) return <svg width={width} height={height} />;

  const pad = 7;
  const x = (i) => pad + (i / Math.max(1, pts.length - 1)) * (width - pad * 2);
  const y = (v) => height - pad - v * (height - pad * 2);

  const line = pts.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `${line} L${x(pts.length - 1).toFixed(1)},${height - pad} L${x(0).toFixed(1)},${height - pad} Z`;
  const last = pts[pts.length - 1];

  return (
    <svg width={width} height={height} role="img" aria-label="Stance across the conversation">
      <line
        x1={pad} y1={y(0.5)} x2={width - pad} y2={y(0.5)}
        stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="2 3" opacity=".45"
      />
      {pts.length > 1 && <path d={area} fill="var(--accent)" opacity=".1" />}
      {pts.length > 1 && (
        <path d={line} fill="none" stroke="var(--accent)" strokeWidth="1.8"
              strokeLinejoin="round" strokeLinecap="round" />
      )}
      {pts.map((v, i) => (
        <circle
          key={i} cx={x(i)} cy={y(v)} r={i === pts.length - 1 ? 3.4 : 2}
          fill={i === pts.length - 1
            ? last > 0.5 ? "var(--support)" : "var(--attack)"
            : "var(--accent)"}
        >
          <title>{history[i].label}: {Math.round(v * 100)}%</title>
        </circle>
      ))}
    </svg>
  );
}
