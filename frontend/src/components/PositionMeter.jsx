const pct = (v) => `${Math.round(v * 100)}%`;

/* The persona's position on the claim: the current reading, the previous
   turn's reading as an outline, and the neutral tick. The number is coloured
   by side; the caption says which way. */
export default function PositionMeter({ state }) {
  const stance = state.stance;
  const history = state.history ?? [];
  const prev = history.length > 1 ? history[history.length - 2].stance : null;
  const side = stance > 0.55 ? "support" : stance < 0.45 ? "attack" : null;
  const colour = side === "support" ? "var(--support)" : side === "attack" ? "var(--attack)" : "var(--ink-2)";
  const delta = Math.round((stance - 0.5) * 100);

  return (
    <div className="meter-row">
      <span className="eyebrow">Position on the claim</span>
      <span className="meter-val" style={{ color: colour }}>{pct(stance)}</span>
      <div className="meter-wrap">
        <div className="meter" role="img" aria-label={`Position ${pct(stance)} for the claim`}>
          <div className="axis" />
          <div className="ticks">{Array.from({ length: 11 }, (_, i) => <i key={i} />)}</div>
          <div className="neutral" />
          {prev !== null && Math.abs(prev - stance) > 0.005 && (
            <div className="mark prev" style={{ left: `${prev * 100}%`, color: colour }} title={`Before the last turn: ${pct(prev)}`} />
          )}
          <div className="mark" style={{ left: `${stance * 100}%`, background: colour }} />
        </div>
        <div className="meter-labels"><span>against</span><span>neutral</span><span>for</span></div>
      </div>
      <span className="meter-swing">
        {delta === 0 ? "at neutral" : `${delta > 0 ? "▲ +" : "▼ "}${delta} pts from neutral`}
      </span>
    </div>
  );
}
