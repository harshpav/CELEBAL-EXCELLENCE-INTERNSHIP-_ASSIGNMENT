// ─── Data Preview ────────────────────────────────────────────────────────────

export interface DataPreview {
  columns: string[];
  rows: Record<string, string>[];
  totalRows: number;
  dtypes: Record<string, string>;
  nulls: Record<string, number>;
  shape: { rows: number; cols: number };
}

// ─── Upload ───────────────────────────────────────────────────────────────────

export interface UploadResponse {
  session_id: string;
  filename: string;
  preview: DataPreview;
  error?: string;
}

// ─── Query / Agent ───────────────────────────────────────────────────────────

export interface TraceStep {
  attempt: number;
  rag_hits: string[];
  code?: string;
  status: "success" | "error" | "llm_error";
  error?: string;
}

export interface QueryResponse {
  question: string;
  success: boolean;
  answer: string;
  code: string | null;
  chart: string | null;   // base64 PNG
  attempts: number;
  trace: TraceStep[];
  error?: string;
}

// ─── History ─────────────────────────────────────────────────────────────────

export interface HistoryEntry extends QueryResponse {
  id: string;
  askedAt: number;
}
