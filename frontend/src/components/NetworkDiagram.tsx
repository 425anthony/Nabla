import { useMemo, useState } from "react";
import type { LayerSnapshot } from "../hooks/useTraining";

interface Props {
  layerSizes: number[];
  snapshot: LayerSnapshot[] | null;
  showGradients: boolean;
  isTraining?: boolean;
}

const W = 700;
const H = 340;
const NODE_R = 10;
const MAX_DISPLAY_NODES = 8; // cap per layer for visual clarity

// Endpoints of the gradient heatmap scale (also used by the legend):
// purple (low) → amber (high), which pops on the dark surface.
const GRAD_LOW = "#7c6ff7";
const GRAD_HIGH = "#f59e0b";

// Map a value in [0,1] to a purple→amber color for the gradient heatmap
function gradientColor(t: number): string {
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(124, 245)},${lerp(111, 158)},${lerp(247, 11)})`;
}

// Map a weight to an edge color: positive=teal, negative=coral
function weightColor(w: number): string {
  return w >= 0 ? "#00d4aa" : "#ff7a5c";
}

// Per-layer neuron base color
function neuronColor(li: number, layers: number): string {
  if (li === 0) return "#5b6b8c";           // input: dim gray-blue
  if (li === layers - 1) return "#00d4aa";  // output: teal
  return "#7c6ff7";                          // hidden: purple/violet
}

export function NetworkDiagram({ layerSizes, snapshot, showGradients, isTraining }: Props) {
  const [inspected, setInspected] = useState<{
    layer: number; neuron: number
  } | null>(null);

  const layers = layerSizes.length;
  const xStep = W / (layers + 1);

  // Positions for each neuron in each layer (capped at MAX_DISPLAY_NODES)
  const positions = useMemo(() => {
    return layerSizes.map((size, li) => {
      const display = Math.min(size, MAX_DISPLAY_NODES);
      const yStep = H / (display + 1);
      const x = xStep * (li + 1);
      return Array.from({ length: display }, (_, ni) => ({
        x,
        y: yStep * (ni + 1),
        neuronIdx: ni,
      }));
    });
  }, [layerSizes, xStep]);

  // Sample a subset of edges between adjacent layers (cap for perf).
  // snapshot[li] is the weight matrix FROM visual layer li TO visual layer li+1.
  const edges = useMemo(() => {
    const result: {
      x1: number; y1: number; x2: number; y2: number;
      weight: number; li: number; ni: number; nj: number;
    }[] = [];
    for (let li = 0; li < layers - 1; li++) {
      const from = positions[li];
      const to = positions[li + 1];
      // Sample at most 5 source neurons × 5 target neurons
      const srcSample = from.slice(0, 5);
      const dstSample = to.slice(0, 5);
      for (const src of srcSample) {
        for (const dst of dstSample) {
          const w = snapshot?.[li]?.W?.[src.neuronIdx]?.[dst.neuronIdx] ?? 0;
          result.push({
            x1: src.x, y1: src.y,
            x2: dst.x, y2: dst.y,
            weight: w, li, ni: src.neuronIdx, nj: dst.neuronIdx,
          });
        }
      }
    }
    return result;
  }, [positions, snapshot, layers]);

  // A neuron at visual layer `layer` (>= 1) is produced by weight-layer
  // snapshot[layer - 1]; its incoming weights are column `neuron` of that matrix.
  const inspectedInfo = useMemo(() => {
    if (!inspected || !snapshot) return null;
    const { layer, neuron } = inspected;
    const wl = layer >= 1 ? snapshot[layer - 1] : null;
    if (!wl) {
      return { isInput: true, bias: null, dzMean: null,
        topWeights: [] as { src: number; w: number }[] };
    }
    const colWeights = wl.W.map((row, src) => ({ src, w: row[neuron] ?? 0 }));
    const topWeights = [...colWeights]
      .sort((a, b) => Math.abs(b.w) - Math.abs(a.w))
      .slice(0, 5);
    return {
      isInput: false,
      bias: wl.b?.[0]?.[neuron] ?? null,
      dzMean: wl.dz_mean,
      topWeights,
    };
  }, [inspected, snapshot]);

  // Max abs weight for edge normalisation
  const maxW = useMemo(() => {
    let m = 0.001;
    edges.forEach((e) => { if (Math.abs(e.weight) > m) m = Math.abs(e.weight); });
    return m;
  }, [edges]);

  // Max dz_mean for gradient normalisation. Use a tiny floor only to avoid
  // divide-by-zero so the blue→red scale auto-fits the actual gradient range.
  const maxGrad = useMemo(() => {
    if (!snapshot || snapshot.length === 0) return 1e-9;
    return Math.max(1e-9, ...snapshot.map((s) => s.dz_mean));
  }, [snapshot]);

  return (
    <div style={{ position: "relative" }}>
      <style>{`
        @keyframes nd-pulse {
          0%   { transform: scale(1);   opacity: 0.45; }
          100% { transform: scale(2.1); opacity: 0;    }
        }
        .nd-pulse-ring {
          transform-box: fill-box;
          transform-origin: center;
          animation: nd-pulse 1.7s ease-out infinite;
          pointer-events: none;
        }
      `}</style>

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
        {/* Edges — stroke/width/opacity set via style so they animate (CSS
            transition) smoothly as the scrubber moves between epochs. */}
        {edges.map((e, i) => {
          const opacity = Math.min(1, Math.abs(e.weight) / maxW);
          const col = weightColor(e.weight);
          return (
            <line
              key={i}
              x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
              style={{
                stroke: col,
                strokeWidth: 0.6 + 1.8 * opacity,
                opacity: 0.18 + 0.6 * opacity,
                filter: `drop-shadow(0 0 ${1 + 2 * opacity}px ${col})`,
                transition: "stroke 0.45s ease, stroke-width 0.45s ease, opacity 0.45s ease",
              }}
            />
          );
        })}

        {/* Neurons */}
        {positions.map((layer, li) =>
          layer.map(({ x, y, neuronIdx }) => {
            const isInspected =
              inspected?.layer === li && inspected?.neuron === neuronIdx;
            // Visual layer li (>= 1) is colored by weight-layer li-1's gradient.
            const gradLayer = li >= 1 ? snapshot?.[li - 1] : undefined;
            const gradIntensity = gradLayer
              ? Math.min(1, gradLayer.dz_mean / maxGrad)
              : 0;
            const fill = showGradients && snapshot
              ? gradientColor(gradIntensity)
              : neuronColor(li, layers);
            // Soft colored glow; intensifies while training.
            const glow = isTraining
              ? `drop-shadow(0 0 6px ${fill}) drop-shadow(0 0 13px ${fill})`
              : `drop-shadow(0 0 5px ${fill})`;

            return (
              <g key={`${li}-${neuronIdx}`}>
                {/* "Alive" pulse during active training — glows outward */}
                {isTraining && (
                  <circle
                    cx={x} cy={y} r={NODE_R}
                    className="nd-pulse-ring"
                    fill="none"
                    stroke={fill}
                    strokeWidth={2.5}
                    style={{
                      animationDelay: `${li * 0.15 + neuronIdx * 0.05}s`,
                      filter: `drop-shadow(0 0 6px ${fill})`,
                    }}
                  />
                )}
                <circle
                  cx={x} cy={y} r={isInspected ? NODE_R + 3 : NODE_R}
                  fill={fill}
                  stroke={isInspected ? "#f5b942" : "#0a0a0f"}
                  strokeWidth={isInspected ? 2.5 : 1.5}
                  style={{
                    cursor: "pointer",
                    filter: glow,
                    transition: "fill 0.45s ease, r 0.15s ease, stroke 0.15s ease, filter 0.3s ease",
                  }}
                  onClick={() =>
                    setInspected(
                      isInspected ? null : { layer: li, neuron: neuronIdx }
                    )
                  }
                />
              </g>
            );
          })
        )}

        {/* Layer labels */}
        {layerSizes.map((size, li) => (
          <text
            key={li}
            x={xStep * (li + 1)}
            y={H - 8}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text-tertiary)"
          >
            {li === 0 ? "input" : li === layers - 1 ? "output" : `hidden ${li}`}
            {"\n"}({size})
          </text>
        ))}
      </svg>

      {/* Gradient heatmap legend */}
      {showGradients && (
        <div style={{
          display: "flex", alignItems: "center", gap: 8, marginTop: 8,
          fontSize: 11, color: "var(--color-text-tertiary)",
        }}>
          <span>mean |∂L/∂z| per layer</span>
          <span>low</span>
          <div style={{
            width: 120, height: 10, borderRadius: 3,
            background: `linear-gradient(to right, ${GRAD_LOW}, ${GRAD_HIGH})`,
            border: "0.5px solid var(--color-border-tertiary)",
          }} />
          <span>high</span>
          <span style={{ fontVariantNumeric: "tabular-nums" }}>
            (0 → {maxGrad.toExponential(1)})
          </span>
        </div>
      )}

      {/* Inspect panel */}
      {inspected && inspectedInfo && (
        <div style={{
          marginTop: 8,
          padding: "10px 12px",
          background: "var(--color-background-secondary)",
          borderRadius: 8,
          fontSize: 12,
          color: "var(--color-text-secondary)",
        }}>
          <div style={{
            display: "flex", gap: 16, alignItems: "center",
            marginBottom: inspectedInfo.isInput ? 0 : 8,
          }}>
            <span style={{ fontWeight: 500 }}>
              Layer {inspected.layer}, neuron {inspected.neuron}
            </span>
            {!inspectedInfo.isInput && (
              <>
                <span>bias: {inspectedInfo.bias?.toFixed(4) ?? "n/a"}</span>
                <span>mean |∂L/∂z|: {inspectedInfo.dzMean?.toFixed(5)}</span>
              </>
            )}
            <span
              style={{ marginLeft: "auto", cursor: "pointer" }}
              onClick={() => setInspected(null)}
            >
              ✕
            </span>
          </div>

          {inspectedInfo.isInput ? (
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
              Input neuron — no incoming weights.
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 6 }}>
                Top 5 incoming weights (by magnitude)
              </div>
              {(() => {
                const top = inspectedInfo.topWeights;
                const maxAbs = Math.max(...top.map((t) => Math.abs(t.w)), 1e-9);
                return top.map(({ src, w }) => (
                  <div key={src} style={{
                    display: "flex", alignItems: "center", gap: 8, marginBottom: 3,
                  }}>
                    <span style={{
                      width: 52, fontSize: 11, textAlign: "right",
                      color: "var(--color-text-tertiary)",
                      fontVariantNumeric: "tabular-nums",
                    }}>
                      ← n{src}
                    </span>
                    <div style={{
                      flex: 1, height: 12, borderRadius: 3, overflow: "hidden",
                      background: "var(--color-background-primary)",
                    }}>
                      <div style={{
                        width: `${(Math.abs(w) / maxAbs) * 100}%`,
                        height: "100%",
                        background: weightColor(w),
                        borderRadius: 3,
                        transition: "width 0.3s ease, background 0.3s ease",
                      }} />
                    </div>
                    <span style={{
                      width: 64, fontSize: 11, fontVariantNumeric: "tabular-nums",
                    }}>
                      {w >= 0 ? "+" : ""}{w.toFixed(4)}
                    </span>
                  </div>
                ));
              })()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
