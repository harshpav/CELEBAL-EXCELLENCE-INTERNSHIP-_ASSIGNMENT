import type { DataPreview as DataPreviewType } from "../types";

interface Props {
  preview: DataPreviewType;
}

export default function DataPreview({ preview }: Props) {
  const nullEntries = Object.entries(preview.nulls).filter(([, v]) => v > 0);

  return (
    <div className="glass" style={{ padding: 20 }}>
      <p
        style={{
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.1em",
          color: "var(--primary)",
          textTransform: "uppercase",
          marginBottom: 12,
        }}
      >
        Preview
      </p>

      {/* stats */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
        {[
          `${preview.shape.rows.toLocaleString()} rows`,
          `${preview.shape.cols} cols`,
          `showing ${preview.rows.length}`,
          nullEntries.length
            ? `${nullEntries.length} col${nullEntries.length > 1 ? "s" : ""} w/ nulls`
            : "no nulls",
        ].map((label) => (
          <span
            key={label}
            style={{
              fontSize: 11,
              padding: "3px 8px",
              borderRadius: 4,
              background: "var(--bg3)",
              color: "var(--text2)",
              border: "1px solid var(--border)",
            }}
          >
            {label}
          </span>
        ))}
      </div>

      {/* table */}
      <div
        style={{
          overflowX: "auto",
          overflowY: "auto",
          maxHeight: 200,
          borderRadius: 6,
          border: "1px solid var(--border)",
        }}
      >
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: 11,
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          <thead>
            <tr>
              {preview.columns.map((c) => (
                <th
                  key={c}
                  style={{
                    padding: "6px 10px",
                    textAlign: "left",
                    background: "var(--bg3)",
                    color: "var(--text2)",
                    fontWeight: 600,
                    whiteSpace: "nowrap",
                    position: "sticky",
                    top: 0,
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, i) => (
              <tr
                key={i}
                style={{
                  borderBottom: "1px solid var(--border)",
                  background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)",
                }}
              >
                {preview.columns.map((c) => (
                  <td
                    key={c}
                    title={row[c]}
                    style={{
                      padding: "4px 10px",
                      color: row[c] === "" ? "var(--text3)" : "var(--text)",
                      maxWidth: 120,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {row[c] === "" ? "—" : row[c]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* null tags */}
      {nullEntries.length > 0 && (
        <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6 }}>
          {nullEntries.map(([col, n]) => (
            <span
              key={col}
              style={{
                fontSize: 11,
                padding: "2px 7px",
                borderRadius: 4,
                background: "rgba(245,158,11,0.1)",
                color: "var(--yellow)",
                border: "1px solid rgba(245,158,11,0.3)",
              }}
            >
              {col}: {n} null{n === 1 ? "" : "s"}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
