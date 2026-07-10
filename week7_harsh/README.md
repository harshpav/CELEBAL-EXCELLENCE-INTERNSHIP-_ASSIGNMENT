# RAG System — Retrieval-Augmented Generation for Document Q&A

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers
questions grounded in your own documents (PDFs, text files, Markdown).

Instead of relying solely on an LLM's internal knowledge, this system:
1. **Retrieves** relevant passages from your documents using hybrid vector + keyword search.
2. **Augments** the LLM prompt with those passages as context.
3. **Generates** a factually grounded answer with source citations.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Python API](#python-api)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [Experiments & Improvements](#experiments--improvements)
- [References](#references)

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │          INGESTION PIPELINE          │
                        │                                      │
  PDF/TXT/MD ──────────▶│ Load → Preprocess → Chunk → Embed  │
                        │                                  ↓  │
                        │                         ChromaDB    │
                        └─────────────────────────────────────┘

                        ┌─────────────────────────────────────┐
                        │           QUERY PIPELINE             │
                        │                                      │
  User Question ────────▶  Embed Query                        │
                        │       ↓              ↓              │
                        │  Dense Search    BM25 Search        │
                        │  (ChromaDB)      (rank-bm25)        │
                        │       ↓              ↓              │
                        │       └──── RRF Fusion ─────┘       │
                        │                  ↓                  │
                        │             Reranker                │
                        │          (FlashRank)                │
                        │                  ↓                  │
                        │     LLM (gpt-4o-mini / any)        │
                        │                  ↓                  │
  Answer ◀──────────────│   Grounded Answer + Source Citations │
                        └─────────────────────────────────────┘
```

---

## Features

| Feature | Details |
|---------|---------|
| **Document formats** | PDF (page-by-page), TXT, Markdown |
| **Chunking strategies** | Recursive character splitter, Semantic (embedding-based) |
| **Embedding backends** | OpenAI `text-embedding-3-small` or local `BAAI/bge-small-en-v1.5` (offline) |
| **Vector store** | ChromaDB — fully local, no server needed |
| **Hybrid search** | Vector (dense) + BM25 (sparse) fused with Reciprocal Rank Fusion |
| **Reranking** | FlashRank (fast, CPU-only) or CrossEncoder (heavier, more accurate) |
| **LLM generation** | OpenAI GPT-4o-mini (configurable), grounded system prompt, source citations |
| **Evaluation** | RAGAS: faithfulness, answer relevancy, context precision/recall, correctness |
| **CLI** | Full-featured Typer + Rich terminal interface |
| **Idempotent ingestion** | Re-ingesting unchanged files is safe (chunk deduplication via MD5 IDs) |

---

## Quick Start

```bash
# 1. Clone / copy the project
cd rag

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -e .

# 4. Configure your API key
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 5. Ingest a document
rag ingest path\to\your\document.pdf

# 6. Ask a question
rag query "What is the main idea of the document?"
```

**No OpenAI key? Use the local embedding backend (fully offline):**

```bash
# Uses BAAI/bge-small-en-v1.5 (downloads ~130 MB on first run)
rag --embedding-backend local ingest document.pdf
rag --embedding-backend local query "What does the document say about X?"
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# Install with all dependencies
pip install -e .

# Install with development tools (testing, linting)
pip install -e ".[dev]"
```

### Verify installation

```bash
rag --help
```

---

## Configuration

Copy `.env.example` to `.env` and set the values:

```ini
# Required for OpenAI embedding + generation
OPENAI_API_KEY=sk-...

# Embedding backend: "openai" or "local"
EMBEDDING_BACKEND=openai

# Local model (only used when EMBEDDING_BACKEND=local)
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Vector store location
CHROMA_PERSIST_DIR=./data/chroma

# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Retrieval
RETRIEVAL_TOP_K=20
RERANK_TOP_N=5
BM25_WEIGHT=0.4        # 0 = pure vector, 1 = pure BM25

# Generation
OPENAI_CHAT_MODEL=gpt-4o-mini
MAX_ANSWER_TOKENS=1024
LLM_TEMPERATURE=0.0
```

All settings can also be overridden via environment variables.

---

## CLI Usage

### `rag ingest` — Add documents

```bash
# Ingest a single PDF
rag ingest report.pdf

# Ingest all documents in a folder (recursive by default)
rag ingest ./docs

# Use semantic chunking (groups sentences by topic)
rag ingest report.pdf --chunking semantic

# Non-recursive directory scan
rag ingest ./docs --no-recursive
```

Output:
```
┌──────────────────────────────────────────────────────────────────┐
│ Source        │ Docs │ Chunks │ Stored │ Status                  │
│ report.pdf    │  12  │   87   │   87   │  OK                     │
└──────────────────────────────────────────────────────────────────┘
Total: 1 file(s), 87 chunk(s) ingested. Vector store now has 87 chunk(s).
```

### `rag query` — Ask questions

```bash
# Basic question
rag query "What is the main conclusion?"

# Show the retrieved context chunks alongside the answer
rag query "What datasets were used?" --show-context

# Output as JSON (for programmatic use)
rag query "What is RAG?" --json

# Filter to a specific document
rag query "What were the results?" --source report.pdf

# Retrieve more candidates before reranking
rag query "Explain the methodology" --top-k 30 --top-n 7
```

Example output:
```
╭─────────────────── Answer ─────────────────────────────────────╮
│  The main conclusion is that RAG systems significantly improve  │
│  factual accuracy compared to vanilla LLM generation [SOURCE 1]│
│  by grounding answers in retrieved document passages [SOURCE 2].│
╰── model=gpt-4o-mini  tokens=312 ──────────────────────────────╯

┌─────────────────── Sources ───────────────────────────────────┐
│  #  │ File        │ Page │ Score                              │
│  1  │ report.pdf  │  3   │ 0.0148                             │
│  2  │ report.pdf  │  7   │ 0.0141                             │
└───────────────────────────────────────────────────────────────┘
```

### `rag status` — Pipeline info

```bash
rag status
```

### `rag clear` — Remove all documents

```bash
rag clear             # prompts for confirmation
rag clear --yes       # skip confirmation
```

### `rag eval` — Evaluate pipeline quality

```bash
# Prepare an evaluation file (JSONL format)
# Each line: {"question": "...", "ground_truth": "..."}

rag eval eval_set.jsonl

# Save detailed per-sample report
rag eval eval_set.jsonl --output results.json

# Evaluate only 10 samples
rag eval eval_set.jsonl --sample 10
```

---

## Python API

Use the pipeline directly in your own code:

```python
from rag_system.pipeline import RAGPipeline

# Build pipeline from .env config
pipeline = RAGPipeline.from_config()

# Ingest documents
results = pipeline.ingest("path/to/documents/")
for r in results:
    print(r)

# Query
answer = pipeline.query("What is the main idea?")
print(answer.answer)
print(answer.sources)
print(f"Tokens used: {answer.total_tokens}")

# Inspect pipeline state
print(pipeline.status())
```

### Custom configuration

```python
from rag_system.pipeline import RAGPipeline

pipeline = RAGPipeline.from_config(
    embedding_backend="local",   # offline mode
    reranker_backend="none",     # skip reranking (faster)
    chunking="semantic",         # topic-aware chunking
)
```

### Direct module usage

```python
from rag_system.ingestion import load_document
from rag_system.chunking import RecursiveChunker
from rag_system.embeddings import create_embedder
from rag_system.vector_store import VectorStore
from rag_system.retrieval import HybridRetriever
from rag_system.reranker import create_reranker
from rag_system.generation import AnswerGenerator

# Load
docs = load_document("paper.pdf")

# Chunk
chunker = RecursiveChunker(chunk_size=800, chunk_overlap=100)
chunks = chunker.split(docs)

# Embed & store
embedder = create_embedder("local")
store = VectorStore()
store.add_chunks(chunks, embedder=embedder)

# Retrieve
retriever = HybridRetriever(store, embedder)
candidates = retriever.search("What is the main finding?", top_k=20)

# Rerank
reranker = create_reranker("flashrank")
top_chunks = reranker.rerank("What is the main finding?", candidates, top_n=5)

# Generate
gen = AnswerGenerator()
result = gen.generate(query="What is the main finding?", contexts=top_chunks)
print(result)
```

---

## Evaluation

### Prepare an evaluation file

Create `eval_set.jsonl` with one Q&A pair per line:

```jsonl
{"question": "What is the main idea of the document?", "ground_truth": "The document introduces a RAG system that combines retrieval and generation."}
{"question": "What embedding model is used?", "ground_truth": "The system supports OpenAI text-embedding-3-small and BAAI/bge-small-en-v1.5."}
{"question": "What retrieval method is used?", "ground_truth": "Hybrid search combining dense vector search and BM25 with RRF fusion."}
```

### Run evaluation

```bash
rag eval eval_set.jsonl
```

### Understanding the metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Are claims in the answer supported by the context? | ≥ 0.70 |
| **Answer Relevancy** | Does the answer actually address the question? | ≥ 0.70 |
| **Context Precision** | Are retrieved chunks relevant to the question? | ≥ 0.70 |
| **Context Recall** | Does context contain enough info to answer? | ≥ 0.60 |
| **Answer Correctness** | How close is the answer to the ground truth? | ≥ 0.70 |

> **Note:** Full RAGAS metrics require an OpenAI API key (they use an LLM as
> judge). If unavailable, the system falls back to heuristic metrics automatically.

### Using the open RAGBench dataset

```bash
# Download the vectara/open_ragbench dataset from HuggingFace
python -c "
from datasets import load_dataset
ds = load_dataset('vectara/open_ragbench', split='test[:100]')
import json
with open('eval_set.jsonl', 'w') as f:
    for row in ds:
        f.write(json.dumps({'question': row['query'], 'ground_truth': row['answer']}) + '\n')
"
rag eval eval_set.jsonl
```

---

## Project Structure

```
rag/
├── .env.example                 # Environment variable template
├── .gitignore
├── pyproject.toml               # Dependencies and project config
├── README.md
│
├── src/
│   └── rag_system/
│       ├── __init__.py
│       ├── config.py            # Pydantic settings (loads .env)
│       ├── logging_setup.py     # Loguru configuration
│       │
│       ├── ingestion.py         # PDF/TXT/MD loader + preprocessing
│       ├── chunking.py          # Recursive and semantic chunkers
│       ├── embeddings.py        # OpenAI and local embedding backends
│       ├── vector_store.py      # ChromaDB wrapper
│       ├── retrieval.py         # Hybrid search (vector + BM25 + RRF)
│       ├── reranker.py          # FlashRank and CrossEncoder rerankers
│       ├── generation.py        # LLM answer generator + prompt engineering
│       ├── pipeline.py          # End-to-end orchestrator
│       ├── cli.py               # Typer CLI (5 commands)
│       └── evaluation.py        # RAGAS evaluation framework
│
├── data/
│   ├── chroma/                  # ChromaDB storage (auto-created)
│   └── sample_docs/             # Place sample documents here
│
└── logs/                        # Log files (auto-created)
```

---

## Key Design Decisions

### Why ChromaDB?
Local, zero-infrastructure vector database. Data lives on disk — no Docker, no
API key, no cloud costs. Suitable for single-machine deployments and experimentation.

### Why hybrid search (vector + BM25)?
Pure vector search misses exact keyword matches; pure BM25 misses semantic
similarity. RRF fusion consistently outperforms either approach in isolation.

### Why FlashRank for reranking?
FlashRank uses a 4 MB cross-encoder model that runs in ~2 ms on CPU. It
improves precision dramatically over raw RRF scores with negligible overhead.

### Why chunk size 800, overlap 100?
Validated against common benchmarks. Smaller chunks (256–512) often split
important context; larger chunks (>1000) inflate token costs and reduce
retrieval precision. The overlap prevents information loss at chunk boundaries.

### Why idempotent ingestion?
Each chunk gets a deterministic MD5 ID based on source + page + chunk_index.
Re-ingesting the same document only overwrites existing chunks — it never
creates duplicates.

---

## Experiments & Improvements

Ranked by expected impact:

| Experiment | Expected Gain | Effort |
|-----------|--------------|--------|
| Tune chunk size (256 vs 512 vs 800) on your domain | High | Low |
| Try `text-embedding-3-large` (3072-dim) | Medium | Low |
| Use semantic chunking for structured documents | Medium | Low |
| Add query expansion (HyDE or multi-query) | High | Medium |
| Switch reranker to `cross-encoder/ms-marco-MiniLM-L-6-v2` | Medium | Low |
| Add metadata filters (date, author, document type) | Medium | Medium |
| Fine-tune embedding model on domain data | Very High | High |
| Implement parent-document retrieval | High | Medium |

---

## References

- [Vectara Open RAGBench Dataset](https://huggingface.co/datasets/vectara/open_ragbench)
- [RAG Document Q&A Reference Implementation](https://github.com/VivekChauhan05/RAG_Document_Question_Answering)
- [RAGAS: Evaluation Framework for RAG](https://docs.ragas.io)
- [Reciprocal Rank Fusion (RRF) paper](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [FlashRank — Fast reranking](https://github.com/PrithivirajDamodaran/FlashRank)
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
