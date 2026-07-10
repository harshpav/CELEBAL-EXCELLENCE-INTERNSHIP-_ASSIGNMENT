import { useState } from "react";
import type { FormEvent } from "react";

interface Props {
  disabled: boolean;
  loading: boolean;
  columns: string[];
  onAsk: (question: string) => void;
}

function buildSuggestions(columns: string[]): string[] {
  if (columns.length === 0) return [];
  const suggestions = [
    "Show me a summary of missing values",
    `What is the distribution of ${columns[0]}?`,
  ];
  if (columns.length > 1) {
    suggestions.push(`Compare ${columns[0]} vs ${columns[1]}`);
  }
  suggestions.push("Find the top 10 rows by the largest numeric column");
  return suggestions.slice(0, 4);
}

export default function QueryConsole({ disabled, loading, columns, onAsk }: Props) {
  const [value, setValue] = useState("");
  const suggestions = buildSuggestions(columns);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const q = value.trim();
    if (!q || disabled || loading) return;
    onAsk(q);
    setValue("");
  };

  return (
    <div className="glass" style={{ padding: 20 }}>
      <p
        style={{
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.1em",
          color: "var(--primary)",
          textTransform: "uppercase",
          marginBottom: 14,
        }}
      >
        02 · Ask the Co-Pilot
      </p>

      <form onSubmit={submit} style={{ display: "flex", gap: 8 }}>
        <input
          className="input"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={disabled}
          placeholder={
            disabled
              ? "Upload a file to start asking questions…"
              : "e.g. Which region had the highest revenue growth?"
          }
        />
        <button
          className="btn btn-primary"
          type="submit"
          disabled={disabled || loading || !value.trim()}
          style={{ whiteSpace: "nowrap", minWidth: 64 }}
        >
          {loading ? (
            <span
              style={{
                display: "inline-block",
                width: 14,
                height: 14,
                border: "2px solid rgba(255,255,255,0.3)",
                borderTopColor: "#fff",
                borderRadius: "50%",
              }}
              className="spin"
            />
          ) : (
            "Run ▶"
          )}
        </button>
      </form>

      {!disabled && suggestions.length > 0 && (
        <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 6 }}>
          {suggestions.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => !loading && onAsk(s)}
              disabled={loading}
              style={{
                fontSize: 11,
                padding: "4px 10px",
                borderRadius: 20,
                background: "var(--bg3)",
                color: "var(--text2)",
                border: "1px solid var(--border)",
                cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.15s",
                opacity: loading ? 0.5 : 1,
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
