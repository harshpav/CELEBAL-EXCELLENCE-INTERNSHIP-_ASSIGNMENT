"""
app.py - Flask API for Autonomous Data Science Co-Pilot
"""
import os
import uuid
import logging
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from agent import run_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE, "uploads")
DIST_DIR   = os.path.join(BASE, "..", "frontend", "dist")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=DIST_DIR)
CORS(app)

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

sessions: dict = {}  # session_id -> {df, filename}

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum allowed size is 25 MB."}), 413

def load_df(path: str, ext: str) -> pd.DataFrame:
    if ext == ".csv":   return pd.read_csv(path, encoding_errors="replace")
    if ext in (".xlsx", ".xls"): return pd.read_excel(path)
    if ext == ".json":  return pd.read_json(path)
    raise ValueError(f"Unsupported: {ext}")

def preview(df: pd.DataFrame, n=10) -> dict:
    return {
        "columns": list(df.columns),
        "rows": df.head(n).fillna("").astype(str).to_dict(orient="records"),
        "totalRows": len(df),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nulls": df.isnull().sum().to_dict(),
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
    }

# ── API ──────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".csv", ".xlsx", ".xls", ".json"):
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    sid = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{sid}{ext}")
    f.save(path)

    try:
        df = load_df(path, ext)
        sessions[sid] = {"df": df, "filename": f.filename}
        logger.info(f"Uploaded {f.filename} → {sid} {df.shape}")
        return jsonify({"session_id": sid, "filename": f.filename, "preview": preview(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/query", methods=["POST"])
def query():
    body = request.get_json() or {}
    sid  = body.get("session_id")
    q    = (body.get("question") or "").strip()
    if not sid or not q:
        return jsonify({"error": "Missing session_id or question"}), 400
    if sid not in sessions:
        return jsonify({"error": "Session not found. Please re-upload your file."}), 404

    df     = sessions[sid]["df"]
    result = run_agent(q, df)
    return jsonify({
        "question": q,
        "success":  result["success"],
        "answer":   result["answer"],
        "code":     result["code"],
        "chart":    result["chart"],
        "attempts": result["attempts"],
        "trace":    result.get("trace", []),
    })

# ── Serve React frontend ─────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    full = os.path.join(DIST_DIR, path)
    if path and os.path.exists(full):
        return send_from_directory(DIST_DIR, path)
    idx = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(idx):
        return send_from_directory(DIST_DIR, "index.html")
    return jsonify({"message": "Backend running. Build frontend first."}), 200

if __name__ == "__main__":
    # Verify env is configured before starting
    import os as _os
    if not _os.getenv("GROQ_API_KEY", "").strip():
        print("❌  ERROR: GROQ_API_KEY is not set in backend/.env")
        print("   Copy backend/.env.example → backend/.env and add your key.")
        exit(1)
    port = int(_os.getenv("PORT", 5000))
    print(f"🚀 Data Co-Pilot running at http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
