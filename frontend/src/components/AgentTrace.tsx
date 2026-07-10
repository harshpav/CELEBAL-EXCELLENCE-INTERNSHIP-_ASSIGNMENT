import type { TraceStep } from "../types";

interface Props {
  trace: TraceStep[];
  pending: boolean;
}

type StatusColor = { dot: string; text: string; bg: string };

function colors(status: TraceStep["status"]): StatusColor {
  if (status === "success")
    return { dot: "var(--green)", text: "var(--green)", bg: "rgba(16,185,129,0.08)" };
  if (status === "error" || status === "llm_error")
    return { dot: "var(--red)", text: "var(--red)", bg: "rgba(239,68,68,0.06)" };
  return { dot: "var(--text3)", text: "var(--text2)", bg: "transparent" };
}

function label(status: TraceStep["status"]): string {
  if (status === "success") return "resolved ✓";
  if (status === "llm_error") return "generation failed";
  return "raised error · retrieving docs";
}

export default function AgentTrace({ trace, pending }: Props) {
  if (trace.length === 0 && !pending) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {trace.map((step, i) => {
        const c = colors(step.status);
        return (
          <div
            key={i}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 6,
              padding: "10px 12px",
              borderRadius: 8,
              background: c.bg,
              border: `1px solid ${c.dot}30`,
              fontSize: 12,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: c.dot,
                  flexShrink: 0,
                  display: "inline-block",
                }}
              />
              <span style={{ color: "var(--text2)", fontWeight: 600 }}>
                attempt {step.attempt}
              </span>
              <span style={{ color: c.text, marginLeft: "auto" }}>{label(step.status)}</span>
            </div>

            {step.rag_hits && step.rag_hits.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5, paddingLeft: 16 }}>
                {step.rag_hits.slice(0, 3).map((doc, j) => {
                  const short = doc.replace(/^- /, "").slice(0, 40);
                  return (
                    <span
                      key={j}
                      title={doc}
                      style={{
                        fontSize: 10,
                        padding: "2px 6px",
                        borderRadius: 4,
                        background: "rgba(124,58,237,0.12)",
                        color: "var(--primary)",
                        border: "1px solid rgba(124,58,237,0.2)",
                      }}
                    >
                      {short}{doc.length > 40 ? "…" : ""}
                    </span>
                  );
                })}
              </div>
            )}

            {step.error && (
              <pre
                style={{
                  marginTop: 4,
                  paddingLeft: 16,
                  fontSize: 11,
                  color: "var(--red)",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                  fontFamily: "'JetBrains Mono', monospace",
                  maxHeight: 80,
                  overflow: "auto",
                }}
              >
                {step.error.split("\n").slice(0, 4).join("\n")}
              </pre>
            )}
          </div>
        );
      })}

      {pending && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "10px 12px",
            borderRadius: 8,
            background: "rgba(124,58,237,0.06)",
            border: "1px solid rgba(124,58,237,0.2)",
            fontSize: 12,
          }}
        >
          <span
            className="spin"
            style={{
              display: "inline-block",
              width: 8,
              height: 8,
              borderRadius: "50%",
              border: "2px solid rgba(124,58,237,0.3)",
              borderTopColor: "var(--primary)",
              flexShrink: 0,
            }}
          />
          <span style={{ color: "var(--text2)", fontWeight: 600 }}>
            attempt {trace.length + 1}
          </span>
          <span style={{ color: "var(--primary)", marginLeft: "auto" }}>
            writing &amp; executing…
          </span>
        </div>
      )}
    </div>
  );
}
