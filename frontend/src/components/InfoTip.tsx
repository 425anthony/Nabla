import { createContext, useContext, useState } from "react";
import { GLOSSARY, type GlossaryKey } from "../glossary";

// True in beginner mode — controls whether ⓘ icons render at all.
export const BeginnerContext = createContext(false);

/**
 * Small ⓘ icon next to a technical term. Hover or click shows a plain-English
 * definition from the glossary. Renders nothing outside beginner mode.
 */
export function InfoTip({ term }: { term: GlossaryKey }) {
  const beginner = useContext(BeginnerContext);
  const [hover, setHover] = useState(false);
  const [pinned, setPinned] = useState(false);

  if (!beginner) return null;

  const entry = GLOSSARY[term];
  const open = hover || pinned;

  return (
    <span style={{ position: "relative", display: "inline-flex", verticalAlign: "middle" }}>
      <button
        type="button"
        aria-label={`What is ${entry.title}?`}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setPinned((p) => !p); }}
        style={{
          width: 15, height: 15, borderRadius: "50%", marginLeft: 5,
          fontSize: 10, fontStyle: "italic", fontWeight: 700, lineHeight: 1,
          fontFamily: "Georgia, 'Times New Roman', serif",
          cursor: "pointer", padding: 0, flex: "none",
          display: "inline-flex", alignItems: "center", justifyContent: "center",
          border: "1px solid",
          borderColor: open ? "var(--accent-purple)" : "var(--color-border-secondary)",
          background: open ? "rgba(124,111,247,0.2)" : "transparent",
          color: open ? "#c9c2ff" : "var(--color-text-tertiary)",
        }}
      >
        i
      </button>
      {open && (
        <span
          role="tooltip"
          style={{
            position: "absolute", top: "calc(100% + 6px)", left: "50%",
            transform: "translateX(-50%)", width: 250, zIndex: 100,
            padding: "10px 12px", borderRadius: 8,
            background: "#1b1b27",
            border: "1px solid var(--color-border-secondary)",
            boxShadow: "0 8px 28px rgba(0,0,0,0.55)",
            // reset inherited text styling (some labels are uppercase / spaced)
            fontSize: 12, fontWeight: 400, lineHeight: 1.5,
            letterSpacing: "normal", textTransform: "none", textAlign: "left",
            whiteSpace: "normal", color: "var(--color-text-secondary)",
          }}
        >
          <span style={{
            display: "block", fontWeight: 600, marginBottom: 4,
            color: "var(--color-text-primary)",
          }}>
            {entry.title}
          </span>
          {entry.text}
        </span>
      )}
    </span>
  );
}
