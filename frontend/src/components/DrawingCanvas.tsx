import { useRef, useState, useEffect, useContext } from "react";
import { API_URL } from "../hooks/useTraining";
import { BeginnerContext } from "./InfoTip";

const SIZE = 280;   // on-canvas drawing resolution (also the displayed size)
const STROKE = 20;  // pen width in canvas pixels

interface Prediction {
  probabilities: number[];
  prediction: number;
}

// Convert the 280x280 drawing (white ink on black) to the 784-vector MNIST
// expects: crop to the ink's bounding box, scale to fit a 20x20 box, then center
// by center-of-mass inside a 28x28 frame, normalized to [0, 1].
function preprocess(canvas: HTMLCanvasElement): number[] | null {
  const ctx = canvas.getContext("2d")!;
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);

  // Bounding box of drawn pixels (red channel = intensity, white-on-black).
  let minX = SIZE, minY = SIZE, maxX = -1, maxY = -1;
  const THRESH = 20;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (data[(y * SIZE + x) * 4] > THRESH) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null; // nothing drawn

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;

  // Scale the cropped digit to fit within 20x20 (preserving aspect ratio).
  const scale = 20 / Math.max(bw, bh);
  const sw = Math.max(1, Math.round(bw * scale));
  const sh = Math.max(1, Math.round(bh * scale));

  const tmp = document.createElement("canvas");
  tmp.width = sw;
  tmp.height = sh;
  const tctx = tmp.getContext("2d")!;
  tctx.imageSmoothingEnabled = true; // antialiased downscale, like MNIST
  tctx.drawImage(canvas, minX, minY, bw, bh, 0, 0, sw, sh);

  // Center of mass of the scaled digit.
  const sdata = tctx.getImageData(0, 0, sw, sh).data;
  let mass = 0, cx = 0, cy = 0;
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const v = sdata[(y * sw + x) * 4];
      mass += v;
      cx += x * v;
      cy += y * v;
    }
  }
  if (mass === 0) return null;
  cx /= mass;
  cy /= mass;

  // Paste into a 28x28 frame so the center of mass sits at (14, 14).
  const out = document.createElement("canvas");
  out.width = 28;
  out.height = 28;
  const octx = out.getContext("2d")!;
  octx.fillStyle = "#000";
  octx.fillRect(0, 0, 28, 28);
  let px = Math.round(14 - cx);
  let py = Math.round(14 - cy);
  px = Math.max(0, Math.min(28 - sw, px));
  py = Math.max(0, Math.min(28 - sh, py));
  octx.drawImage(tmp, px, py);

  const odata = octx.getImageData(0, 0, 28, 28).data;
  const pixels = new Array(784);
  for (let i = 0; i < 784; i++) pixels[i] = odata[i * 4] / 255; // red channel → [0,1]
  return pixels;
}

export function DrawingCanvas({ trainExamples }: { trainExamples: number }) {
  const beginner = useContext(BeginnerContext);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const dirty = useRef(false);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fillBlack = () => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, SIZE, SIZE);
    }
  };
  useEffect(fillBlack, []);

  const at = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (SIZE / rect.width),
      y: (e.clientY - rect.top) * (SIZE / rect.height),
    };
  };

  const start = (e: React.PointerEvent) => {
    e.preventDefault();
    drawing.current = true;
    dirty.current = true;
    const ctx = canvasRef.current!.getContext("2d")!;
    const { x, y } = at(e);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = STROKE;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.01, y + 0.01); // dot for a single tap
    ctx.stroke();
  };

  const move = (e: React.PointerEvent) => {
    if (!drawing.current) return;
    e.preventDefault();
    const ctx = canvasRef.current!.getContext("2d")!;
    const { x, y } = at(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const end = () => {
    if (!drawing.current) return;
    drawing.current = false;
    if (dirty.current) predict();
  };

  const clear = () => {
    fillBlack();
    dirty.current = false;
    setPred(null);
    setError(null);
  };

  const predict = async () => {
    const pixels = preprocess(canvasRef.current!);
    if (!pixels) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      setPred(await res.json());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const maxProb = pred ? Math.max(...pred.probabilities) : 1;

  return (
    <div className="nabla-diagram-card" style={{
      padding: 16, borderRadius: 12,
      background: "var(--color-background-primary)",
      marginBottom: 16,
    }}>
      <p style={{ fontSize: 13, fontWeight: 600, margin: "0 0 4px" }}>Draw a digit</p>
      {beginner && (
        <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", margin: "0 0 14px", lineHeight: 1.5 }}>
          Draw any digit (0–9) and the network will guess what it is using what it
          learned from {trainExamples.toLocaleString()} handwritten examples.
        </p>
      )}

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
        {/* Canvas + clear */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10, alignItems: "center" }}>
          <canvas
            ref={canvasRef}
            width={SIZE}
            height={SIZE}
            onPointerDown={start}
            onPointerMove={move}
            onPointerUp={end}
            onPointerLeave={end}
            style={{
              width: SIZE, height: SIZE, borderRadius: 12,
              background: "#000", cursor: "crosshair", touchAction: "none",
              border: "1px solid rgba(124,111,247,0.40)",
              boxShadow: "0 0 24px rgba(124,111,247,0.25)",
            }}
          />
          <button onClick={clear} className="nabla-btn">
            Clear
          </button>
        </div>

        {/* Prediction results */}
        <div style={{ flex: 1, minWidth: 240 }}>
          {pred ? (
            <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
              {/* Winning digit, large in teal */}
              <div style={{ textAlign: "center" }}>
                <div style={{
                  fontSize: 64, fontWeight: 700, lineHeight: 1, color: "#00d4aa",
                  textShadow: "0 0 22px rgba(0,212,170,0.45)", fontVariantNumeric: "tabular-nums",
                }}>
                  {pred.prediction}
                </div>
                <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 4 }}>
                  {Math.round(pred.probabilities[pred.prediction] * 100)}% sure
                </div>
              </div>

              {/* Per-digit probability bars */}
              <div style={{ flex: 1 }}>
                {pred.probabilities.map((p, d) => {
                  const win = d === pred.prediction;
                  return (
                    <div key={d} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                      <span style={{
                        width: 12, marginRight: 6, fontSize: 12, textAlign: "right",
                        color: win ? "#00d4aa" : "var(--color-text-tertiary)",
                        fontWeight: win ? 700 : 400, fontVariantNumeric: "tabular-nums",
                      }}>
                        {d}
                      </span>
                      <div style={{
                        flex: 1, height: 12, borderRadius: 3, overflow: "hidden",
                        background: "var(--color-background-input)",
                      }}>
                        <div style={{
                          width: `${(p / maxProb) * 100}%`, height: "100%", borderRadius: 3,
                          background: win ? "#00d4aa" : "#7c6ff7",
                          transition: "width 0.25s ease, background 0.25s ease",
                        }} />
                      </div>
                      <span style={{
                        width: 46, fontSize: 11, textAlign: "right",
                        color: "var(--color-text-tertiary)", fontVariantNumeric: "tabular-nums",
                      }}>
                        {(p * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div style={{ fontSize: 13, color: "var(--color-text-tertiary)", paddingTop: 8 }}>
              {loading
                ? "Reading your drawing…"
                : error
                  ? `Error: ${error}`
                  : "Draw a digit on the left — the prediction appears here when you lift your pen."}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
