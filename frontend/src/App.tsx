import { useState, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { useTraining, type TrainConfig } from "./hooks/useTraining";
import { NetworkDiagram } from "./components/NetworkDiagram";

const DEFAULT_CONFIG: TrainConfig = {
  layer_sizes: [784, 128, 64, 10],
  hidden_activation: "relu",
  epochs: 15,
  batch_size: 256,
  learning_rate: 0.01,
  momentum: 0.9,
  data_fraction: 0.1,
};

export default function App() {
  const { snapshots, isTraining, isDone, error, startTraining, stopTraining } =
    useTraining();

  const [config, setConfig] = useState<TrainConfig>(DEFAULT_CONFIG);
  const [epochIdx, setEpochIdx] = useState(0);
  const [showGradients, setShowGradients] = useState(false);
  const [playing, setPlaying] = useState(false);

  // Keep epoch scrubber at latest epoch while training
  const displayEpoch = isTraining ? snapshots.length - 1 : epochIdx;
  const currentSnap = snapshots[displayEpoch] ?? null;

  const chartData = useMemo(
    () =>
      snapshots.map((s) => ({
        epoch: s.epoch,
        "train loss": s.train_loss,
        "train acc": s.train_acc,
        "test acc": s.test_acc,
      })),
    [snapshots]
  );

  // Play through epochs
  const togglePlay = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    setPlaying(true);
    let i = epochIdx;
    const interval = setInterval(() => {
      i++;
      if (i >= snapshots.length) {
        clearInterval(interval);
        setPlaying(false);
      } else {
        setEpochIdx(i);
      }
    }, 300);
  };

  return (
    <div style={{
      maxWidth: 900, margin: "0 auto", padding: "24px 20px",
      fontFamily: "var(--font-sans, system-ui)",
      color: "var(--color-text-primary, #1a1a1a)",
    }}>
      <h1 style={{ fontSize: 22, fontWeight: 500, marginBottom: 4 }}>
        Neural network visualizer
      </h1>
      <p style={{ fontSize: 14, color: "var(--color-text-secondary)", marginBottom: 24 }}>
        Pure NumPy backprop on MNIST — watch weights and gradients evolve during training
      </p>

      {/* Config panel */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
        gap: 12, marginBottom: 20,
        padding: "16px", background: "var(--color-background-secondary)",
        borderRadius: 12, border: "0.5px solid var(--color-border-tertiary)",
      }}>
        {(["epochs", "batch_size", "learning_rate", "data_fraction"] as const).map((key) => (
          <label key={key} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
              {key.replace("_", " ")}
            </span>
            <input
              type="number"
              value={config[key] as number}
              step={key === "learning_rate" ? 0.001 : key === "data_fraction" ? 0.05 : 1}
              min={key === "data_fraction" ? 0.05 : 1}
              max={key === "data_fraction" ? 1 : undefined}
              onChange={(e) =>
                setConfig((c) => ({ ...c, [key]: parseFloat(e.target.value) }))
              }
              disabled={isTraining}
              style={{ padding: "4px 8px", borderRadius: 6, fontSize: 13,
                border: "0.5px solid var(--color-border-secondary)",
                background: "var(--color-background-primary)" }}
            />
          </label>
        ))}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 10, marginBottom: 24, alignItems: "center" }}>
        <button
          onClick={() => isTraining ? stopTraining() : startTraining(config)}
          style={{
            padding: "8px 20px", borderRadius: 8, fontSize: 14, cursor: "pointer",
            background: isTraining ? "var(--color-background-danger)" : "var(--color-background-info)",
            color: isTraining ? "var(--color-text-danger)" : "var(--color-text-info)",
            border: "0.5px solid",
            borderColor: isTraining ? "var(--color-border-danger)" : "var(--color-border-info)",
          }}
        >
          {isTraining ? "Stop training" : "Start training"}
        </button>

        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13,
          color: "var(--color-text-secondary)", cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={showGradients}
            onChange={(e) => setShowGradients(e.target.checked)}
          />
          Gradient heatmap
        </label>

        {isDone && (
          <span style={{ fontSize: 12, color: "var(--color-text-success)",
            marginLeft: "auto" }}>
            Training complete — {snapshots.length} epochs recorded
          </span>
        )}
        {error && (
          <span style={{ fontSize: 12, color: "var(--color-text-danger)" }}>
            Error: {error}
          </span>
        )}
      </div>

      {/* Network diagram */}
      {snapshots.length > 0 && (
        <>
          <div style={{
            padding: 16, borderRadius: 12,
            border: "0.5px solid var(--color-border-tertiary)",
            background: "var(--color-background-primary)",
            marginBottom: 16,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between",
              alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontSize: 13, fontWeight: 500 }}>
                Epoch {(currentSnap?.epoch ?? 0)} — loss:{" "}
                {currentSnap?.train_loss?.toFixed(4)} — test acc:{" "}
                {((currentSnap?.test_acc ?? 0) * 100).toFixed(1)}%
              </span>
              {showGradients && (
                <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>
                  blue → red = low → high |∂L/∂z|
                </span>
              )}
            </div>

            <NetworkDiagram
              layerSizes={config.layer_sizes}
              snapshot={currentSnap?.layers ?? null}
              showGradients={showGradients}
            />
          </div>

          {/* Epoch scrubber */}
          {!isTraining && snapshots.length > 1 && (
            <div style={{ display: "flex", alignItems: "center", gap: 12,
              marginBottom: 20 }}>
              <button onClick={togglePlay} style={{
                padding: "4px 12px", borderRadius: 6, fontSize: 12,
                cursor: "pointer", border: "0.5px solid var(--color-border-secondary)",
                background: "transparent", color: "var(--color-text-secondary)",
              }}>
                {playing ? "⏸ Pause" : "▶ Play"}
              </button>
              <input
                type="range" min={0} max={snapshots.length - 1}
                value={epochIdx}
                onChange={(e) => { setPlaying(false); setEpochIdx(+e.target.value); }}
                style={{ flex: 1 }}
              />
              <span style={{ fontSize: 12, color: "var(--color-text-tertiary)",
                minWidth: 60 }}>
                epoch {epochIdx + 1} / {snapshots.length}
              </span>
            </div>
          )}

          {/* Loss / accuracy chart */}
          <div style={{
            padding: 16, borderRadius: 12,
            border: "0.5px solid var(--color-border-tertiary)",
            background: "var(--color-background-primary)",
          }}>
            <p style={{ fontSize: 13, fontWeight: 500, marginBottom: 12, marginTop: 0 }}>
              Loss and accuracy over epochs
            </p>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData}>
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="train loss" stroke="#D85A30"
                  dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="train acc" stroke="#378ADD"
                  dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
                <Line type="monotone" dataKey="test acc" stroke="#1D9E75"
                  dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
