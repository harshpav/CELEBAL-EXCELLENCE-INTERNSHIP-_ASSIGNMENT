# 🧠 Single Agent Pipeline Project

A single-agent smart assistant with conditional routing, tool integration, and local AI using Ollama.

## Project Structure

```
agent_project/
├── backend/
│   ├── app.py       # Flask API server
│   └── agent.py     # Agent logic + tools
├── frontend/
│   └── index.html   # React chat UI
├── requirements.txt
└── README.md
```

## Features

- ⚡ **Calculator Tool** — solves math expressions (`Calculate 25 * 4`)
- 🔑 **Keyword Extractor** — extracts keywords from text
- 🤖 **Ollama AI** — answers general queries using local llama3 model

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama (in a separate terminal)
```bash
ollama serve
```

### 3. Start the backend
```bash
cd backend
python app.py
```

### 4. Open the UI
Visit: [http://localhost:5000](http://localhost:5000)

## API

**POST** `/api/agent`
```json
{ "query": "Calculate 10 + 5" }
```
Response:
```json
{ "type": "calculation", "result": "15" }
```

## Tech Stack
- **Backend:** Python, Flask, Ollama (llama3)
- **Frontend:** React 18, HTML/CSS
