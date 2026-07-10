# ⚡ Data Co-Pilot

An **Autonomous Data Science Agent** — upload a CSV, Excel, or JSON file, ask questions in plain English, and the AI writes Python/Pandas code, executes it safely, and self-corrects on errors.

---

## 🧠 AI Model

| Component | Detail |
|-----------|--------|
| **LLM Provider** | [Groq](https://console.groq.com) |
| **Model** | `llama-3.3-70b-versatile` |
| **RAG Store** | ChromaDB (in-memory, Pandas/matplotlib docs) |
| **Self-correction** | Up to 3 retry attempts with error feedback |

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, Flask 3.1 |
| AI Agent | Groq SDK + LLaMA 3.3-70B |
| RAG | ChromaDB 0.6 |
| Data | Pandas 2.2, NumPy 1.26 |
| Charts | Matplotlib 3.9, Seaborn 0.13 |
| Frontend | React 18, TypeScript, Vite 5 |
| Deployment | Render (render.yaml included) |

---

## 📁 Project Structure

```
copilot/
├── backend/
│   ├── app.py          # Flask API (upload, query, serve frontend)
│   ├── agent.py        # LLM agent with retry + trace logic
│   ├── rag.py          # ChromaDB RAG pipeline
│   ├── sandbox.py      # Safe code execution (30s timeout)
│   ├── .env            # Your API key (never commit this)
│   ├── .env.example    # Template — copy to .env
│   └── uploads/        # Uploaded files (auto-created)
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── types.ts
│   │   └── components/
│   │       ├── Header.tsx
│   │       ├── UploadPanel.tsx
│   │       ├── DataPreview.tsx
│   │       ├── QueryConsole.tsx
│   │       ├── AgentTrace.tsx
│   │       └── ResultCard.tsx
│   └── package.json
├── requirements.txt
├── render.yaml         # Render deployment config
├── Procfile            # Gunicorn start command
├── dev.bat             # One-click dev startup (Windows)
└── start.bat           # Production-style startup (Windows)
```

---

## ⚙️ Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Groq API Key** → get one free at https://console.groq.com

---

## 🚀 Setup & Run

### Step 1 — Clone / open the project

```bat
cd C:\Users\harsh\Downloads\copilot
```

### Step 2 — Configure API Key

```bat
copy backend\.env.example backend\.env
```

Open `backend\.env` and paste your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 3 — Install backend dependencies

```bat
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r ..\requirements.txt
cd ..
```

### Step 4 — Install frontend dependencies

```bat
cd frontend
npm install
cd ..
```

---

## ▶️ Running the App

### Quickest way (both servers at once)

```bat
dev.bat
```

This opens two terminal windows:
- **Backend (Flask)** → http://localhost:5000
- **Frontend (Vite)** → http://localhost:5173 ← open this in browser

---

### Run separately (if dev.bat doesn't work)

**Terminal 1 — Backend:**

```bat
cd C:\Users\harsh\Downloads\copilot\backend
.venv\Scripts\activate
python app.py
```

**Terminal 2 — Frontend:**

```bat
cd C:\Users\harsh\Downloads\copilot\frontend
npm run dev
```

Then open → **http://localhost:5173**

---

## 🤗 Deploy to Hugging Face Spaces

### Step 1 — Create a new Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **New Space**
3. Fill in:
   - **Space name:** `data-copilot` (or any name)
   - **SDK:** `Docker`
   - **Visibility:** Public or Private
4. Click **Create Space**

### Step 2 — Add your API key as a Secret
1. In your Space → go to **Settings** tab
2. Scroll to **Repository secrets**
3. Add:
   - **Name:** `GROQ_API_KEY`
   - **Value:** your Groq API key
4. Save

### Step 3 — Push code to the Space

```bash
# Clone the HF space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/data-copilot
cd data-copilot

# Copy your project files into it
# (copy everything from copilot/ folder here)

# The SPACES_README.md content must be at the top of README.md
# Copy the HF header from SPACES_README.md to the top of README.md

# Push
git add .
git commit -m "Initial deploy"
git push
```

### Step 4 — Wait for build
Hugging Face will:
1. Run `Dockerfile` — builds React frontend + installs Python deps
2. Start gunicorn on port **7860**
3. Your app is live at: `https://huggingface.co/spaces/YOUR_USERNAME/data-copilot`

> ⚠️ **Important:** When pushing to HuggingFace, make sure `README.md` starts with the YAML block from `SPACES_README.md`:
> ```
> ---
> title: Data Co-Pilot
> emoji: ⚡
> colorFrom: purple
> colorTo: blue
> sdk: docker
> pinned: false
> app_port: 7860
> ---
> ```

---

## 🌐 Deploy to Render

1. Push to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service → connect your repo
3. Render auto-detects `render.yaml`
4. Add environment variable in Render dashboard:
   - Key: `GROQ_API_KEY`
   - Value: your Groq key
5. Deploy ✅

---

## 📦 Supported File Types

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| JSON | `.json` |
| Max size | **25 MB** |

---

## 🔒 Security Notes

- `GROQ_API_KEY` is loaded from `.env` — never hardcoded
- `.env` is blocked by `.gitignore` — won't be committed
- Code execution runs in a sandboxed environment with:
  - Blocked imports: `os`, `sys`, `subprocess`, `socket`, `requests`, etc.
  - **30-second execution timeout**
  - No file system write access

---

## 💡 Example Questions

Once you upload a file, try asking:

- `Show me a summary of missing values`
- `What is the distribution of [column name]?`
- `Plot the top 10 categories by revenue`
- `Find correlation between price and sales`
- `Show monthly trend of orders`

---

## 🛠️ How the Agent Works

```
User question
     ↓
is_data_query() check
     ↓
Build prompt with dataset info + RAG docs
     ↓
Groq LLaMA 3.3-70B generates Python code
     ↓
sandbox.py executes code safely (30s timeout)
     ↓
Success? → return answer + chart
     ↓ (on error)
Feed error back → retry up to 3 times
     ↓
Return trace of all attempts to frontend
```

---

## 📄 License

MIT
