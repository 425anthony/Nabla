import { useState, useMemo, useEffect, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { useTraining, type TrainConfig } from "./hooks/useTraining";
import { NetworkDiagram } from "./components/NetworkDiagram";
import { InfoTip, BeginnerContext } from "./components/InfoTip";

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
  const { snapshots, isTraining, isDone, error, startTraining, stopTraining, reset } =
    useTraining();

  const [config, setConfig] = useState<TrainConfig>(DEFAULT_CONFIG);
  const [epochIdx, setEpochIdx] = useState(0);
  const [showGradients, setShowGradients] = useState(false);
  const [beginner, setBeginner] = useState(true); // ML-beginner mode (ⓘ tips + plain-English panel)
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1); // playback multiplier: 0.5x | 1x | 2x
  const rafRef = useRef<number | null>(null);
  const speedRef = useRef(speed);        // read live inside the rAF loop
  useEffect(() => { speedRef.current = speed; }, [speed]);

  // One epoch advances every BASE_STEP_MS / speed; the diagram's CSS transition
  // is tied to this so colors morph continuously across the whole step.
  const BASE_STEP_MS = 700;
  const stepMs = BASE_STEP_MS / speed;

  const stopPlaying = () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setPlaying(false);
  };

  // Cancel any running playback frame when the component unmounts.
  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
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

  // Play through epochs with requestAnimationFrame for smooth, jitter-free timing.
  const togglePlay = () => {
    if (playing) {
      stopPlaying();
      return;
    }
    // Restart from the beginning if we're already at the end.
    let i = epochIdx >= snapshots.length - 1 ? 0 : epochIdx;
    setEpochIdx(i);
    setPlaying(true);

    let last: number | null = null;
    let acc = 0;
    const tick = (now: number) => {
      if (last !== null) acc += now - last;
      last = now;
      const step = BASE_STEP_MS / speedRef.current; // live speed
      while (acc >= step) {
        acc -= step;
        i++;
        if (i >= snapshots.length) {
          stopPlaying();
          return;
        }
        setEpochIdx(i);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  // ── Architecture builder ───────────────────────────────────────────────
  // layer_sizes = [input(784), ...hidden, output(10)]; only hidden is editable.
  const MAX_HIDDEN = 6;     // keeps total layers ≤ 8 (matches backend validator)
  const MIN_NEURONS = 1;
  const MAX_NEURONS = 256;
  const hidden = config.layer_sizes.slice(1, -1);

  // Any architecture edit invalidates the prior run's snapshots.
  const setLayerSizes = (next: number[]) => {
    setConfig((c) => ({ ...c, layer_sizes: next }));
    reset();
  };
  const addHiddenLayer = () => {
    if (hidden.length >= MAX_HIDDEN) return;
    const sizes = config.layer_sizes;
    setLayerSizes([...sizes.slice(0, -1), 32, sizes[sizes.length - 1]]);
  };
  const removeHiddenLayer = (idx: number) => {
    const next = config.layer_sizes.filter((_, i) => i !== idx + 1); // +1: skip input
    setLayerSizes(next);
  };
  const setHiddenSize = (idx: number, value: number) => {
    const v = Math.max(MIN_NEURONS, Math.min(MAX_NEURONS, value || MIN_NEURONS));
    const next = config.layer_sizes.map((s, i) => (i === idx + 1 ? v : s));
    setLayerSizes(next);
  };

  // ── Plain-English status for beginner mode (updates each epoch) ─────────
  const lastSnap = snapshots[snapshots.length - 1] ?? null;
  const beginnerStatus = (() => {
    if (error) return `Something went wrong while training: ${error}`;
    if (isTraining) {
      if (!currentSnap) {
        return "Warming up — loading the handwritten-digit images and starting the network off with random guesses…";
      }
      const pct = Math.round(currentSnap.test_acc * 100);
      const prev = snapshots[snapshots.length - 2];
      const better = !prev || currentSnap.test_acc >= prev.test_acc;
      return `Epoch ${currentSnap.epoch}: The network is adjusting its weights to reduce mistakes${better ? " and getting better" : ""} — it now correctly identifies ${pct}% of digits it has never seen before.`;
    }
    if (isDone && lastSnap) {
      return `Done! After ${snapshots.length} epochs the network correctly identifies ${Math.round(lastSnap.test_acc * 100)}% of unseen digits. Drag the slider below to replay how it learned, epoch by epoch.`;
    }
    if (snapshots.length > 0 && currentSnap) {
      return `Replaying epoch ${currentSnap.epoch}: at this point in training the network gets ${Math.round(currentSnap.test_acc * 100)}% of unseen digits right.`;
    }
    return "This network starts out guessing randomly. Press “Start training” and watch the diagram below come alive as it learns to read handwritten digits.";
  })();

  // Overfitting: training accuracy meaningfully ahead of test accuracy.
  const overfitGap =
    currentSnap && currentSnap.train_acc != null && currentSnap.test_acc != null
      ? currentSnap.train_acc - currentSnap.test_acc
      : 0;
  const showOverfit = beginner && overfitGap > 0.05;

  return (
    <BeginnerContext.Provider value={beginner}>
    <div style={{
      maxWidth: 900, margin: "0 auto", padding: "24px 20px",
      fontFamily: "var(--font-sans, system-ui)",
      color: "var(--color-text-primary, #1a1a1a)",
    }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
        <h1 style={{
          fontSize: 34, fontWeight: 700, marginBottom: 2, letterSpacing: "-0.02em",
          color: "#f3f1ff",
          textShadow: "0 0 24px rgba(124,111,247,0.45)",
        }}>
          <span style={{ color: "var(--accent-purple)" }}>∇</span> Nabla
        </h1>

        {/* Beginner / Advanced mode toggle */}
        <div style={{
          display: "flex", gap: 2, padding: 3, borderRadius: 9, marginTop: 6,
          background: "var(--color-background-secondary)",
          border: "1px solid var(--color-border-secondary)",
        }}>
          {([["beginner", "Beginner"], ["advanced", "Advanced"]] as const).map(([val, label]) => {
            const active = (val === "beginner") === beginner;
            return (
              <button
                key={val}
                onClick={() => setBeginner(val === "beginner")}
                style={{
                  padding: "5px 14px", borderRadius: 7, fontSize: 12, cursor: "pointer",
                  border: "none", fontWeight: active ? 600 : 400,
                  background: active ? "rgba(124,111,247,0.22)" : "transparent",
                  color: active ? "#c9c2ff" : "var(--color-text-tertiary)",
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <p style={{ fontSize: 14, color: "var(--color-text-secondary)", marginBottom: 24 }}>
        Neural network visualizer — pure NumPy backprop
        <InfoTip term="backpropagation" /> on MNIST, watching weights and gradients evolve during training
      </p>

      {/* Config panel */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
        gap: 14, marginBottom: 20,
        padding: "18px", background: "var(--color-background-primary)",
        borderRadius: 12, border: "1px solid var(--color-border-secondary)",
      }}>
        {(["epochs", "batch_size", "learning_rate", "data_fraction"] as const).map((key) => {
          const tip = key === "epochs" ? "epoch"
            : key === "batch_size" ? "batch_size"
            : key === "learning_rate" ? "learning_rate"
            : "data_fraction";
          return (
          <label key={key} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
              {key.replace("_", " ")}
              <InfoTip term={tip} />
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
          );
        })}
      </div>

      {/* Architecture builder */}
      <div style={{
        marginBottom: 20, padding: "18px",
        background: "var(--color-background-primary)",
        borderRadius: 12, border: "1px solid var(--color-border-secondary)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
          <span style={{ fontSize: 13, fontWeight: 600 }}>Architecture</span>
          <span style={{ fontSize: 12, color: "var(--color-text-tertiary)", fontVariantNumeric: "tabular-nums" }}>
            {config.layer_sizes.join(" → ")}
          </span>
          <button
            onClick={addHiddenLayer}
            disabled={isTraining || hidden.length >= MAX_HIDDEN}
            className="nabla-btn"
            style={{ marginLeft: "auto", padding: "5px 14px", fontSize: 13, opacity: (isTraining || hidden.length >= MAX_HIDDEN) ? 0.4 : 1 }}
          >
            + Add hidden layer
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {/* Fixed input layer */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 13, color: "var(--color-text-tertiary)" }}>
            <span style={{ width: 78 }}>input</span>
            <span style={{ fontVariantNumeric: "tabular-nums" }}>784 (fixed)</span>
          </div>

          {/* Editable hidden layers */}
          {hidden.map((n, idx) => (
            <div key={idx} style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ width: 78, fontSize: 13, color: "var(--color-text-secondary)" }}>
                hidden {idx + 1}
              </span>
              <input
                type="range"
                min={MIN_NEURONS} max={MAX_NEURONS} value={n}
                onChange={(e) => setHiddenSize(idx, +e.target.value)}
                disabled={isTraining}
                style={{ flex: 1 }}
              />
              <input
                type="number"
                min={MIN_NEURONS} max={MAX_NEURONS} value={n}
                onChange={(e) => setHiddenSize(idx, parseInt(e.target.value, 10))}
                disabled={isTraining}
                className="nabla-input"
                style={{ width: 72 }}
              />
              <span style={{ fontSize: 12, color: "var(--color-text-tertiary)", width: 52 }}>neurons</span>
              <button
                onClick={() => removeHiddenLayer(idx)}
                disabled={isTraining}
                title="Remove this layer"
                style={{
                  width: 26, height: 26, borderRadius: 6, cursor: isTraining ? "default" : "pointer",
                  background: "transparent", color: "var(--color-text-danger)",
                  border: "1px solid var(--color-border-secondary)",
                  opacity: isTraining ? 0.4 : 1, lineHeight: 1, fontSize: 16,
                }}
              >
                −
              </button>
            </div>
          ))}

          {/* Fixed output layer */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 13, color: "var(--color-text-tertiary)" }}>
            <span style={{ width: 78 }}>output</span>
            <span style={{ fontVariantNumeric: "tabular-nums" }}>10 (fixed)</span>
          </div>
        </div>

        {hidden.length >= MAX_HIDDEN && (
          <p style={{ fontSize: 11, color: "var(--color-text-tertiary)", margin: "10px 0 0" }}>
            Maximum {MAX_HIDDEN} hidden layers.
          </p>
        )}
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
          <InfoTip term="gradient_heatmap" />
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

      {/* Beginner mode: plain-English annotation panel (live status) */}
      {beginner && (
        <div style={{
          display: "flex", alignItems: "flex-start", gap: 10,
          marginBottom: 20, padding: "12px 14px 12px 18px", borderRadius: 10,
          background: "rgba(124,111,247,0.08)",
          border: "1px solid rgba(124,111,247,0.30)",
          borderLeft: "3px solid var(--accent-purple)",
          fontSize: 13, lineHeight: 1.5, color: "var(--color-text-secondary)",
        }}>
          <span style={{ fontSize: 15, lineHeight: 1.4 }}>💡</span>
          <span>{beginnerStatus}</span>
        </div>
      )}

      {/* Beginner mode: overfitting warning (train acc well above test acc) */}
      {showOverfit && (
        <div style={{
          display: "flex", alignItems: "flex-start", gap: 10,
          marginBottom: 20, padding: "12px 14px 12px 18px", borderRadius: 10,
          background: "rgba(245,158,11,0.08)",
          border: "1px solid rgba(245,158,11,0.30)",
          borderLeft: "3px solid var(--accent-amber)",
          fontSize: 13, lineHeight: 1.5, color: "var(--color-text-secondary)",
        }}>
          <span style={{ fontSize: 15, lineHeight: 1.4 }}>⚠️</span>
          <span>
            <strong>Possible overfitting.</strong> The network gets{" "}
            {Math.round((currentSnap?.train_acc ?? 0) * 100)}% of the images it trained on
            right, but only {Math.round((currentSnap?.test_acc ?? 0) * 100)}% of unseen ones
            — a gap of {Math.round(overfitGap * 100)}%. That usually means it's starting to
            memorize the training images instead of learning general patterns that work on
            new digits. Training on more data (higher data fraction), for fewer epochs, or
            with a smaller network typically helps.
          </span>
        </div>
      )}

      {/* Network diagram — always visible as a live architecture preview */}
      <div className="nabla-diagram-card" style={{
        padding: 16, borderRadius: 12,
        background: "var(--color-background-primary)",
        marginBottom: 16,
      }}>
        {currentSnap ? (
            <div style={{
              display: "flex", alignItems: "baseline", gap: 28,
              flexWrap: "wrap", marginBottom: 16,
            }}>
              <span style={{
                fontSize: 12, color: "var(--color-text-tertiary)",
                textTransform: "uppercase", letterSpacing: "0.09em",
              }}>
                Epoch {currentSnap?.epoch ?? 0}<InfoTip term="epoch" />
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
                  test accuracy<InfoTip term="test_acc" />
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
                  loss<InfoTip term="loss" />
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
                  train acc<InfoTip term="train_acc" />
                </span>
              </div>
            </div>
        ) : (
          <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 16 }}>
            Architecture preview: {config.layer_sizes.join(" → ")} — press “Start training” to run.
          </div>
        )}

        <NetworkDiagram
          layerSizes={config.layer_sizes}
          snapshot={currentSnap?.layers ?? null}
          showGradients={showGradients}
          isTraining={isTraining}
          transitionMs={playing ? stepMs : 400}
        />
      </div>

      {/* Epoch playback + loss chart (after a run) */}
      {snapshots.length > 0 && (
        <>
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

              {/* Playback speed */}
              <div style={{ display: "flex", gap: 4 }}>
                {[0.5, 1, 2].map((s) => (
                  <button
                    key={s}
                    onClick={() => setSpeed(s)}
                    style={{
                      padding: "4px 8px", borderRadius: 6, fontSize: 11,
                      cursor: "pointer", fontVariantNumeric: "tabular-nums",
                      border: "0.5px solid",
                      borderColor: speed === s ? "var(--accent-purple)" : "var(--color-border-secondary)",
                      background: speed === s ? "rgba(124,111,247,0.15)" : "transparent",
                      color: speed === s ? "#c9c2ff" : "var(--color-text-tertiary)",
                    }}
                  >
                    {s}x
                  </button>
                ))}
              </div>

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
    </BeginnerContext.Provider>
  );
}
