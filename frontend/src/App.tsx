import { useState, useMemo, useEffect, useRef } from "react";
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
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPlaying = () => {
    if (playRef.current !== null) {
      clearInterval(playRef.current);
      playRef.current = null;
    }
    setPlaying(false);
  };

  // Clear any running playback interval when the component unmounts.
  useEffect(() => () => {
    if (playRef.current !== null) clearInterval(playRef.current);
  }, []);

  // Keep epoch scrubber at latest epoch while training
  const displayEpoch = isTraining ? snapshots.length - 1 : epochIdx;
  const currentSnap = snapshots[displayEpoch] ?? null;

  // When training finishes, jump the scrubber to the final (best) epoch
  // instead of leaving it at epoch 0.
  useEffect(() => {
    if (isDone && snapshots.length > 0) setEpochIdx(snapshots.length - 1);
  }, [isDone, snapshots.length]);

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
      stopPlaying();
      return;
    }
    setPlaying(true);
    let i = epochIdx;
    playRef.current = setInterval(() => {
      i++;
      if (i >= snapshots.length) {
        stopPlaying();
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
      <h1 style={{
        fontSize: 34, fontWeight: 700, marginBottom: 2, letterSpacing: "-0.02em",
        color: "#f3f1ff",
        textShadow: "0 0 24px rgba(124,111,247,0.45)",
      }}>
        <span style={{ color: "var(--accent-purple)" }}>∇</span> Nabla
      </h1>
      <p style={{ fontSize: 14, color: "var(--color-text-secondary)", marginBottom: 24 }}>
        Neural network visualizer — pure NumPy backprop on MNIST, watching weights and gradients evolve during training
      </p>

      {/* Config panel */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
        gap: 14, marginBottom: 20,
        padding: "18px", background: "var(--color-background-primary)",
        borderRadius: 12, border: "1px solid var(--color-border-secondary)",
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
              className="nabla-input"
            />
          </label>
        ))}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 10, marginBottom: 24, alignItems: "center" }}>
        <button
          onClick={() => isTraining ? stopTraining() : startTraining(config)}
          className={`nabla-btn${isTraining ? " nabla-btn--stop" : ""}`}
        >
          {isTraining ? "Stop training" : "Start training"}
        </button>

        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13,
          color: "var(--color-text-secondary)", cursor: "pointer" }}>
          <input
            type="checkbox"
            className="nabla-toggle-input"
            checked={showGradients}
            onChange={(e) => setShowGradients(e.target.checked)}
          />
          <span className="nabla-toggle" />
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
          <div className="nabla-diagram-card" style={{
            padding: 16, borderRadius: 12,
            background: "var(--color-background-primary)",
            marginBottom: 16,
          }}>
            <div style={{
              display: "flex", alignItems: "baseline", gap: 28,
              flexWrap: "wrap", marginBottom: 16,
            }}>
              <span style={{
                fontSize: 12, color: "var(--color-text-tertiary)",
                textTransform: "uppercase", letterSpacing: "0.09em",
              }}>
                Epoch {currentSnap?.epoch ?? 0}
              </span>

              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{
                  fontSize: 40, fontWeight: 700, lineHeight: 1, color: "#00d4aa",
                  textShadow: "0 0 22px rgba(0,212,170,0.45)",
                  fontVariantNumeric: "tabular-nums",
                }}>
                  {((currentSnap?.test_acc ?? 0) * 100).toFixed(1)}%
                </span>
                <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
                  test accuracy
                </span>
              </div>

              <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                <span style={{
                  fontSize: 22, fontWeight: 600, color: "#f87171",
                  fontVariantNumeric: "tabular-nums",
                }}>
                  {currentSnap?.train_loss?.toFixed(4)}
                </span>
                <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
                  loss
                </span>
              </div>

              <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                <span style={{
                  fontSize: 15, fontWeight: 500, color: "var(--color-text-secondary)",
                  fontVariantNumeric: "tabular-nums",
                }}>
                  {((currentSnap?.train_acc ?? 0) * 100).toFixed(1)}%
                </span>
                <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
                  train acc
                </span>
              </div>
            </div>

            <NetworkDiagram
              layerSizes={config.layer_sizes}
              snapshot={currentSnap?.layers ?? null}
              showGradients={showGradients}
              isTraining={isTraining}
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
                onChange={(e) => { stopPlaying(); setEpochIdx(+e.target.value); }}
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
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} stroke="#2a2a3a" />
                <YAxis tick={{ fontSize: 11 }} stroke="#2a2a3a" />
                <Tooltip
                  contentStyle={{
                    background: "#101019", border: "1px solid #2a2a3a",
                    borderRadius: 8, fontSize: 12,
                  }}
                  labelStyle={{ color: "#ececf4" }}
                  itemStyle={{ color: "#9b9bb2" }}
                  cursor={{ stroke: "#2a2a3a" }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="train loss" stroke="#f87171"
                  dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="train acc" stroke="#a78bfa"
                  dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
                <Line type="monotone" dataKey="test acc" stroke="#00d4aa"
                  dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
