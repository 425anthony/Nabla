import { useMemo, useState } from "react";
import type { LayerSnapshot } from "../hooks/useTraining";

interface Props {
  layerSizes: number[];
  snapshot: LayerSnapshot[] | null;
  showGradients: boolean;
}

const W = 700;
const H = 340;
const NODE_R = 10;
const MAX_DISPLAY_NODES = 8; // cap per layer for visual clarity

// Map a value in [0,1] to a blue→red color for gradient heatmap
function gradientColor(intensity: number): string {
  const r = Math.round(55 + intensity * 200);
  const b = Math.round(255 - intensity * 200);
  return `rgb(${r},80,${b})`;
}

// Map a weight to a stroke color: positive=blue, negative=coral
function weightColor(w: number): string {
  return w >= 0 ? "#378ADD" : "#D85A30";
}

export function NetworkDiagram({ layerSizes, snapshot, showGradients }: Props) {
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

  // Sample a subset of edges between adjacent layers (cap for perf)
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

  const inspectedInfo = useMemo(() => {
    if (!inspected || !snapshot) return null;
    const { layer, neuron } = inspected;
    if (layer >= snapshot.length) return null;
    const s = snapshot[layer];
    return {
      dz_mean: s.dz_mean,
      bias: s.b?.[0]?.[neuron]?.toFixed(4) ?? "n/a",
      dw_mean: s.dW
        ? (s.dW[neuron]?.reduce((a: number, v: number) => a + Math.abs(v), 0) /
            (s.dW[neuron]?.length || 1)).toFixed(5)
        : "n/a",
    };
  }, [inspected, snapshot]);

  // Max abs weight for normalisation
  const maxW = useMemo(() => {
    let m = 0.001;
    edges.forEach((e) => { if (Math.abs(e.weight) > m) m = Math.abs(e.weight); });
    return m;
  }, [edges]);

  // Max dz_mean for gradient normalisation
  const maxGrad = useMemo(() => {
    if (!snapshot) return 0.001;
    return Math.max(0.001, ...snapshot.map((s) => s.dz_mean));
  }, [snapshot]);

  return (
    <div style={{ position: "relative" }}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
        {/* Edges */}
        {edges.map((e, i) => {
          const opacity = Math.min(1, Math.abs(e.weight) / maxW);
          return (
            <line
              key={i}
              x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
              stroke={weightColor(e.weight)}
              strokeWidth={0.5 + 1.5 * opacity}
              opacity={0.15 + 0.6 * opacity}
            />
          );
        })}

        {/* Neurons */}
        {positions.map((layer, li) =>
          layer.map(({ x, y, neuronIdx }) => {
            const isInspected =
              inspected?.layer === li && inspected?.neuron === neuronIdx;
            const gradIntensity = snapshot
              ? Math.min(1, (snapshot[li]?.dz_mean ?? 0) / maxGrad)
              : 0;
            const fill = showGradients && snapshot
              ? gradientColor(gradIntensity)
              : "#378ADD";

            return (
              <circle
                key={`${li}-${neuronIdx}`}
                cx={x} cy={y} r={isInspected ? NODE_R + 3 : NODE_R}
                fill={fill}
                stroke={isInspected ? "#EF9F27" : "white"}
                strokeWidth={isInspected ? 2.5 : 1}
                style={{ cursor: "pointer", transition: "all 0.15s" }}
                onClick={() =>
                  setInspected(
                    isInspected ? null : { layer: li, neuron: neuronIdx }
                  )
                }
              />
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

      {/* Inspect panel */}
      {inspected && inspectedInfo && (
        <div style={{
          marginTop: 8,
          padding: "8px 12px",
          background: "var(--color-background-secondary)",
          borderRadius: 8,
          fontSize: 12,
          color: "var(--color-text-secondary)",
          display: "flex",
          gap: 24,
        }}>
          <span>Layer {inspected.layer}, neuron {inspected.neuron}</span>
          <span>bias: {inspectedInfo.bias}</span>
          <span>mean |∂L/∂z|: {inspectedInfo.dz_mean.toFixed(5)}</span>
          <span>mean |∂L/∂W|: {inspectedInfo.dw_mean}</span>
          <span
            style={{ marginLeft: "auto", cursor: "pointer" }}
            onClick={() => setInspected(null)}
          >
            ✕
          </span>
        </div>
      )}
    </div>
  );
}
