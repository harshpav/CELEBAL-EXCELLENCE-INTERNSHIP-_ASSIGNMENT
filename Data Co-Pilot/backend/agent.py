"""
agent.py - Autonomous Data Science Agent using Groq + RAG + Sandbox.
"""
import os
import logging
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from sandbox import execute_code
from rag import get_docs

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3

# ── Validate API key on startup ───────────────────────────────────────────────
_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not _GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Create backend/.env with: GROQ_API_KEY=your_key_here"
    )

# Lazy client — created once on first use
_client: Groq | None = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=_GROQ_API_KEY)
    return _client

DATA_KEYWORDS = [
    "column","row","data","dataset","csv","excel","json","plot","chart","graph",
    "visuali","trend","distribution","correlation","outlier","missing","null",
    "mean","average","sum","count","max","min","group","filter","sort","compare",
    "analysis","insight","show","find","top","bottom","percentage","sales",
    "revenue","profit","growth","value","category","customer","price","date",
]

def is_data_query(q: str) -> bool:
    ql = q.lower()
    blocked = ["who is","president of","capital of","joke","poem","history of","weather","what is the"]
    for b in blocked:
        if b in ql: return False
    return any(k in ql for k in DATA_KEYWORDS)

def df_info(df: pd.DataFrame) -> str:
    nulls = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
    return "\n".join([
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"Columns: {list(df.columns)}",
        f"Types: { {k: str(v) for k, v in df.dtypes.items()} }",
        f"Sample:\n{df.head(3).to_string()}",
        *([ f"Nulls: {nulls}" ] if nulls else []),
    ])

def clean(code: str) -> str:
    lines, in_block = [], False
    for line in code.strip().splitlines():
        if line.strip().startswith("```"):
            in_block = not in_block
            continue
        lines.append(line)
    return "\n".join(lines).strip()

def llm(prompt: str) -> str:
    r = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return r.choices[0].message.content

def run_agent(query: str, df: pd.DataFrame) -> dict:
    if not is_data_query(query):
        return {
            "success": False,
            "answer": "I only answer questions about your uploaded dataset. Please ask about your data, columns, trends, or charts.",
            "code": None, "chart": None, "attempts": 0, "trace": [],
        }

    info = df_info(df)
    code, last_err = None, None
    trace = []

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Attempt {attempt}/{MAX_RETRIES}")
        rag_query = query if attempt == 1 else f"{query} {last_err}"
        rag = get_docs(rag_query, n=5)
        rag_hits = [line.strip() for line in rag.splitlines() if line.strip()]

        if attempt == 1:
            prompt = f"""You are an expert Python data analyst. Write Python/Pandas code to answer the question.

Dataset:
{info}

Question: {query}

Helpful docs:
{rag}

RULES:
- Variable 'df' is already loaded. Do NOT redefine it.
- Use pandas, numpy, matplotlib.pyplot as plt, seaborn as sns.
- Add plt.title(), plt.xlabel(), plt.ylabel() to every chart.
- Use plt.tight_layout() at the end.
- Print key results with print().
- Do NOT use plt.show().
- Write ONLY Python code, no markdown, no explanation.

Code:"""
        else:
            prompt = f"""Fix this Python code that produced an error.

Dataset:
{info}

Question: {query}

Broken code:
{code}

Error:
{last_err}

Helpful docs:
{rag}

Write ONLY the fixed Python code, no markdown:"""

        try:
            code = clean(llm(prompt))
            logger.info(f"Code:\n{code}")
        except Exception as e:
            trace.append({"attempt": attempt, "rag_hits": rag_hits, "code": None, "status": "llm_error", "error": str(e)})
            return {"success": False, "answer": f"LLM error: {e}", "code": None, "chart": None, "attempts": attempt, "trace": trace}

        result = execute_code(code, df)
        if result["success"]:
            trace.append({"attempt": attempt, "rag_hits": rag_hits, "code": code, "status": "success", "error": None})
            return {
                "success": True,
                "answer": result["output"] or "Done! See the chart below.",
                "code": code,
                "chart": result["chart"],
                "attempts": attempt,
                "trace": trace,
            }
        last_err = result["error"]
        trace.append({"attempt": attempt, "rag_hits": rag_hits, "code": code, "status": "error", "error": last_err})
        logger.warning(f"Failed: {last_err[:150]}")

    return {
        "success": False,
        "answer": f"Could not complete after {MAX_RETRIES} attempts. Error: {last_err}",
        "code": code, "chart": None, "attempts": MAX_RETRIES, "trace": trace,
    }
