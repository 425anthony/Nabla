import { useState, useRef, useCallback } from "react";

export interface LayerSnapshot {
  W: number[][];
  b: number[][];
  dW: number[][];
  dz_mean: number;  // mean |∂L/∂z| — used for gradient heatmap
}

export interface EpochSnapshot {
  epoch: number;
  train_loss: number;
  train_acc: number;
  test_acc: number;
  layers: LayerSnapshot[];
}

export interface TrainConfig {
  layer_sizes: number[];
  hidden_activation: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  momentum: number;
  data_fraction: number;
}

// Backend base URL — baked in at build time via VITE_API_URL so the same
// build runs locally and in production; falls back to localhost for dev.
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// The backend emits train_accuracy/test_accuracy and per-layer grad_magnitude.
// Map those onto the field names the components consume.
function normalizeSnapshot(raw: any): EpochSnapshot {
  return {
    epoch: raw.epoch,
    train_loss: raw.train_loss,
    train_acc: raw.train_accuracy,
    test_acc: raw.test_accuracy,
    layers: (raw.layers ?? []).map((l: any) => ({
      W: l.W,
      b: l.b,
      dW: l.dW,
      dz_mean: l.grad_magnitude,
    })),
  };
}

export function useTraining() {
  const [snapshots, setSnapshots] = useState<EpochSnapshot[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader | null>(null);

  const startTraining = useCallback(async (config: TrainConfig) => {
    setSnapshots([]);
    setIsDone(false);
    setError(null);
    setIsTraining(true);

    try {
      // 1. Kick off the job — backend returns a job_id immediately.
      const startRes = await fetch(`${API_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!startRes.ok) throw new Error(`HTTP ${startRes.status}`);
      const { job_id } = await startRes.json();

      // 2. Open the SSE stream for that job and read epoch snapshots.
      const res = await fetch(`${API_URL}/train/${job_id}/stream`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (!res.body) throw new Error("No response body");

      const reader = res.body
        .pipeThrough(new TextDecoderStream())
        .getReader();
      readerRef.current = reader;

      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;

        // SSE lines: "data: {...}\n\n"
        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.replace(/^data: /, "").trim();
          if (!trimmed) continue;
          const parsed = JSON.parse(trimmed);
          if (parsed.event === "complete") {
            setIsDone(true);
          } else if (parsed.event === "error") {
            throw new Error(parsed.detail ?? "Training failed");
          } else {
            setSnapshots((prev) => [...prev, normalizeSnapshot(parsed)]);
          }
        }
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsTraining(false);
      readerRef.current = null;
    }
  }, []);

  const stopTraining = useCallback(() => {
    readerRef.current?.cancel();
    setIsTraining(false);
  }, []);

  return { snapshots, isTraining, isDone, error, startTraining, stopTraining };
}
