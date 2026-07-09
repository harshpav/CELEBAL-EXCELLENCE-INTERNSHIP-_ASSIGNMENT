import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ── Tool 1: Calculator ──────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Allow only safe characters
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression."
        result = eval(expression)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error in calculation: {str(e)}"


# ── Tool 2: Keyword Extractor ───────────────────────────────────────────────
def extract_keywords(text: str) -> list:
    """Extract meaningful keywords from text."""
    STOPWORDS = {"this", "that", "with", "from", "have", "will", "been",
                 "they", "their", "what", "when", "where", "which", "about",
                 "just", "into", "than", "then", "some", "also", "more"}
    try:
        words = text.split()
        keywords = list(dict.fromkeys(
            w.lower().strip(".,!?\"'") for w in words
            if len(w) > 4 and w.lower().strip(".,!?\"'") not in STOPWORDS
        ))
        return keywords[:6]
    except Exception:
        return []


# ── Main Agent ──────────────────────────────────────────────────────────────
def agent(query: str) -> dict:
    """
    Single-agent smart assistant with conditional routing.

    Routes:
      - 'calculate' keyword  → Calculator Tool
      - 'keywords' keyword   → Keyword Extractor Tool
      - Everything else      → Ollama LLM (concise answer)
    """
    logger.info(f"Query received: {query}")
    query_lower = query.lower().strip()

    try:
        # Route 1: Calculator
        if "calculate" in query_lower:
            expression = query_lower.replace("calculate", "").strip()
            logger.info(f"→ Calculator: {expression}")
            result = calculator(expression)
            return {"type": "calculation", "result": result}

        # Route 2: Keyword Extractor
        elif "keywords" in query_lower:
            if "from" in query_lower:
                text = query[query_lower.index("from") + 4:].strip()
            else:
                text = query.strip()
            logger.info(f"→ Keyword Extractor: {text}")
            result = extract_keywords(text)
            return {"type": "keywords", "result": result}

        # Route 3: General Response
        else:
            logger.info(f"→ General Response: {query}")
            return {
                "type": "general",
                "result": f'You asked: "{query}". This is a general query. Please use a language model for a detailed answer.'
            }

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {"type": "error", "result": f"An error occurred: {str(e)}"}
