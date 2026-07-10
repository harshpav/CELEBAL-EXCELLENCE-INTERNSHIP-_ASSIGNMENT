import { useState } from "react";
import Header from "./components/Header";
import UploadPanel from "./components/UploadPanel";
import DataPreview from "./components/DataPreview";
import QueryConsole from "./components/QueryConsole";
import AgentTrace from "./components/AgentTrace";
import ResultCard from "./components/ResultCard";
import { askQuestion, uploadFile } from "./api";
import type { DataPreview as DataPreviewType, HistoryEntry } from "./types";

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [preview, setPreview] = useState<DataPreviewType | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      const res = await uploadFile(file);
      setSessionId(res.session_id);
      setFilename(res.filename);
      setPreview(res.preview);
      setHistory([]);
      setQueryError(null);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleClear = () => {
    setSessionId(null);
    setFilename(null);
    setPreview(null);
    setUploadError(null);
    setHistory([]);
    setQueryError(null);
  };

  const handleAsk = async (question: string) => {
    if (!sessionId) return;
    setPendingQuestion(question);
    setQueryError(null);
    try {
      const res = await askQuestion(sessionId, question);
      const entry: HistoryEntry = {
        ...res,
        trace: res.trace ?? [],
        id: `${Date.now()}`,
        askedAt: Date.now(),
      };
      setHistory((h) => [entry, ...h]);
    } catch (e) {
      setQueryError(
        e instanceof Error ? e.message : "Something went wrong running that question."
      );
    } finally {
      setPendingQuestion(null);
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Header connected={!!sessionId} filename={filename} />

      <main
        style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "360px 1fr",
          gap: 20,
          maxWidth: 1280,
          width: "100%",
          margin: "0 auto",
          padding: "20px 24px 40px",
        }}
      >
        {/* ── Left column ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <UploadPanel
            filename={filename}
            uploading={uploading}
            error={uploadError}
            onUpload={handleUpload}
            onClear={handleClear}
          />
          {preview && <DataPreview preview={preview} />}
        </div>

        {/* ── Right column ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <QueryConsole
            disabled={!sessionId}
            loading={!!pendingQuestion}
            columns={preview?.columns ?? []}
            onAsk={handleAsk}
          />

          {queryError && (
            <div
              className="glass"
              style={{ padding: 14, borderColor: "rgba(239,68,68,0.5)", color: "var(--red)", fontSize: 13 }}
            >
              ⚠ {queryError}
            </div>
          )}

          {pendingQuestion && (
            <div className="glass fade-in" style={{ padding: 20 }}>
              <p style={{ fontSize: 13, color: "var(--text2)", marginBottom: 12 }}>
                {pendingQuestion}
              </p>
              <AgentTrace trace={[]} pending />
            </div>
          )}

          {!pendingQuestion && history.length === 0 && !queryError && (
            <div
              className="glass"
              style={{
                padding: 48,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 14,
                color: "var(--text3)",
              }}
            >
              <div style={{ fontSize: 40 }}>▚▞</div>
              <p style={{ textAlign: "center", maxWidth: 340, lineHeight: 1.8, fontSize: 13 }}>
                {sessionId
                  ? "Ask a question about your data. The agent writes Python, runs it, and self-corrects on errors."
                  : "Upload a CSV, Excel, or JSON file to begin."}
              </p>
            </div>
          )}

          {history.map((entry) => (
            <ResultCard key={entry.id} entry={entry} />
          ))}
        </div>
      </main>
    </div>
  );
}
