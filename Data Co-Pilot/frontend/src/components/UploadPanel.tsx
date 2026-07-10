import { useRef, useState } from "react";
import type { DragEvent, ChangeEvent } from "react";

interface Props {
  filename: string | null;
  uploading: boolean;
  error: string | null;
  onUpload: (file: File) => void;
  onClear: () => void;
}

const ACCEPTED = [".csv", ".xlsx", ".xls", ".json"];

export default function UploadPanel({ filename, uploading, error, onUpload, onClear }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase();
    if (!ACCEPTED.includes(ext)) return;
    onUpload(file);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    e.target.value = "";
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
        01 · Source Data
      </p>

      {filename ? (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: "rgba(124,58,237,0.1)",
            border: "1px solid rgba(124,58,237,0.3)",
            borderRadius: 8,
            padding: "10px 14px",
          }}
        >
          <span style={{ fontSize: 13 }}>📄 {filename}</span>
          <button
            className="btn btn-ghost"
            onClick={onClear}
            style={{ padding: "4px 10px", fontSize: 12 }}
          >
            replace
          </button>
        </div>
      ) : (
        <div
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
          style={{
            border: `2px dashed ${dragActive ? "var(--primary)" : "var(--border)"}`,
            borderRadius: 10,
            padding: "32px 20px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 8,
            cursor: "pointer",
            transition: "all 0.2s",
            background: dragActive ? "rgba(124,58,237,0.05)" : "transparent",
            outline: "none",
          }}
        >
          <div style={{ fontSize: 30, userSelect: "none" }}>
            {uploading ? "⋯" : "⇪"}
          </div>
          <div style={{ fontSize: 13, color: "var(--text2)", fontWeight: 500 }}>
            {uploading ? "Reading file…" : "Drop a file, or click to browse"}
          </div>
          <div style={{ fontSize: 11, color: "var(--text3)" }}>
            .csv · .xlsx · .xls · .json — up to 25 MB
          </div>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED.join(",")}
            onChange={onChange}
            style={{ display: "none" }}
          />
        </div>
      )}

      {error && (
        <p
          style={{
            marginTop: 10,
            fontSize: 12,
            color: "var(--red)",
            padding: "8px 12px",
            background: "rgba(239,68,68,0.1)",
            borderRadius: 6,
            border: "1px solid rgba(239,68,68,0.3)",
          }}
        >
          ⚠ {error}
        </p>
      )}
    </div>
  );
}
