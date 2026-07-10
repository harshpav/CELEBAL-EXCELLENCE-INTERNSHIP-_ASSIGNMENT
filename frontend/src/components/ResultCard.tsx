import AgentTrace from "./AgentTrace";
import type { HistoryEntry } from "../types";

interface Props {
  entry: HistoryEntry;
}

export default function ResultCard({ entry }: Props) {
  const borderColor = entry.success
    ? "rgba(16,185,129,0.3)"
    : "rgba(239,68,68,0.3)";

  return (
    <div
      className="glass fade-in"
      style={{
        padding: 20,
        borderColor,
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      {/* question */}
      <div>
        <p
          style={{
            fontSize: 14,
            fontWeight: 600,
            color: "var(--text)",
            marginBottom: 4,
          }}
        >
          {entry.question}
        </p>
        <p style={{ fontSize: 11, color: "var(--text3)" }}>
          {entry.success ? "✓ resolved" : "✗ unresolved"} ·{" "}
          {entry.attempts} attempt{entry.attempts === 1 ? "" : "s"}
        </p>
      </div>

      {/* trace — only show if more than 1 attempt */}
      {entry.trace.length > 1 && (
        <AgentTrace trace={entry.trace} pending={false} />
      )}

      {/* answer */}
      <p
        style={{
          fontSize: 13,
          color: entry.success ? "var(--text)" : "var(--red)",
          lineHeight: 1.7,
          whiteSpace: "pre-wrap",
        }}
      >
        {entry.answer}
      </p>

      {/* chart */}
      {entry.chart && (
        <div
          style={{
            borderRadius: 8,
            overflow: "hidden",
            border: "1px solid var(--border)",
            background: "#fff",
          }}
        >
          <img
            src={`data:image/png;base64,${entry.chart}`}
            alt={`Chart for: ${entry.question}`}
            style={{ width: "100%", display: "block" }}
          />
        </div>
      )}

      {/* generated code */}
      {entry.code && (
        <details style={{ marginTop: 4 }}>
          <summary
            style={{
              fontSize: 12,
              color: "var(--text3)",
              cursor: "pointer",
              userSelect: "none",
              padding: "4px 0",
            }}
          >
            view generated code
          </summary>
          <pre
            style={{
              marginTop: 8,
              padding: "12px 14px",
              background: "var(--bg3)",
              borderRadius: 6,
              border: "1px solid var(--border)",
              fontSize: 11,
              fontFamily: "'JetBrains Mono', monospace",
              color: "var(--text)",
              overflowX: "auto",
              whiteSpace: "pre",
              lineHeight: 1.6,
            }}
          >
            {entry.code}
          </pre>
        </details>
      )}
    </div>
  );
}
