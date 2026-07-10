interface Props {
  connected: boolean;
  filename: string | null;
}

export default function Header({ connected, filename }: Props) {
  return (
    <header
      className="glass"
      style={{
        borderRadius: 0,
        borderLeft: "none",
        borderRight: "none",
        borderTop: "none",
        padding: "14px 24px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      <div>
        <h1
          className="gradient-text"
          style={{ fontSize: 18, fontWeight: 800, letterSpacing: "0.06em" }}
        >
          ⚡ DATA CO-PILOT
        </h1>
        <p style={{ color: "var(--text3)", fontSize: 12, marginTop: 2 }}>
          Upload · Ask · Analyse · Self-heal
        </p>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            fontSize: 12,
            padding: "4px 10px",
            borderRadius: 20,
            background: connected ? "rgba(16,185,129,0.12)" : "rgba(148,163,184,0.08)",
            color: connected ? "var(--green)" : "var(--text3)",
            border: `1px solid ${connected ? "rgba(16,185,129,0.3)" : "var(--border)"}`,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: connected ? "var(--green)" : "var(--text3)",
              animation: connected ? "pulse 2s ease-in-out infinite" : "none",
              display: "inline-block",
            }}
          />
          {connected ? "session active" : "no file loaded"}
        </span>

        {filename && (
          <span style={{ color: "var(--text2)", fontSize: 12 }}>📄 {filename}</span>
        )}
      </div>
    </header>
  );
}
