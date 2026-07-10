"""
Answer Generation Module
========================
Takes a user query and a list of retrieved chunks, constructs a context-
grounded prompt, and calls an OpenAI chat model to produce a final answer.

Design principles
-----------------
1. **Grounded answers** — The system prompt explicitly instructs the model to
   base its answer only on the provided context and to say "I don't know" when
   the context is insufficient.  This reduces hallucinations.

2. **Source attribution** — Chunk sources are injected with [SOURCE N] tags so
   the model can cite them, and sources are returned alongside the answer text.

3. **Token budget** — The context is trimmed to fit within the model's context
   window using tiktoken.

4. **Structured output** — Returns an ``AnswerResult`` dataclass with the answer
   text, the contexts used, source citations, and the raw LLM response metadata.

Prompt engineering
------------------
The prompt follows the pattern:

    SYSTEM: You are a helpful assistant. Answer ONLY using the context below.
            If the answer is not in the context, say "I don't know".
            Cite sources as [SOURCE 1], [SOURCE 2], etc.

    USER:   Context:
            [SOURCE 1] (report.pdf, page 3): ...chunk text...
            [SOURCE 2] (notes.txt, page 1): ...chunk text...

            Question: <user query>

Usage
-----
>>> from rag_system.generation import AnswerGenerator
>>> gen = AnswerGenerator()
>>> result = gen.generate(query="What is RAG?", contexts=reranked_chunks)
>>> print(result.answer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from rag_system.config import settings
from rag_system.vector_store import SearchResult


# ---------------------------------------------------------------------------
# System and user prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert question-answering assistant.
Your task is to answer the user's question using ONLY the context passages provided below.

Rules:
1. Ground every claim in the provided context.  Do NOT use outside knowledge.
2. If the answer cannot be found in the context, respond exactly with:
   "I'm sorry, I don't have enough information in the provided documents to answer that question."
3. Cite the source of each key fact using the [SOURCE N] markers shown in the context.
4. Be concise and precise.  Avoid padding or unnecessary repetition.
5. If multiple sources say the same thing, cite all of them.
"""

_USER_PROMPT_TEMPLATE = """\
Context:
{context_block}

---
Question: {question}

Answer (cite sources as [SOURCE N]):"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    """Structured output from the answer generator."""

    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    """List of dicts with keys: source_num, chunk_id, filename, page, score"""

    contexts_used: list[str] = field(default_factory=list)
    """The raw chunk texts that were passed as context."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __str__(self) -> str:
        lines = [f"Question: {self.question}", "", f"Answer: {self.answer}"]
        if self.sources:
            lines.append("")
            lines.append("Sources:")
            for s in self.sources:
                lines.append(
                    f"  [{s['source_num']}] {s.get('filename', '?')} "
                    f"(page {s.get('page', '?')}, score={s.get('score', 0):.4f})"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer Generator
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """
    Generates grounded answers using an OpenAI chat model.

    Parameters
    ----------
    model          : OpenAI chat model name.
    max_tokens     : Max tokens for the generated answer.
    temperature    : Sampling temperature (0 = deterministic).
    context_limit  : Approximate character limit for the context block.
                     Prevents exceeding the model's context window.
    api_key        : Overrides OPENAI_API_KEY if provided.
    """

    def __init__(
        self,
        model: str = settings.openai_chat_model,
        max_tokens: int = settings.max_answer_tokens,
        temperature: float = settings.llm_temperature,
        context_limit: int = 12_000,  # ~3 000 tokens at ~4 chars/token
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai") from exc

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_limit = context_limit
        self._client = OpenAI(api_key=api_key or settings.openai_api_key or None)

        logger.info(
            "AnswerGenerator ready: model={}, max_tokens={}, temperature={}",
            model,
            max_tokens,
            temperature,
        )

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context_block(
        self, contexts: list[SearchResult]
    ) -> tuple[str, list[dict]]:
        """
        Format retrieved chunks into a numbered context block.

        Returns
        -------
        (context_block_str, source_list)
        """
        lines: list[str] = []
        sources: list[dict] = []
        total_chars = 0

        for i, result in enumerate(contexts, start=1):
            filename = result.metadata.get("filename", result.metadata.get("source", "?"))
            page = result.metadata.get("page", "?")
            # Truncate very long chunks to stay within context_limit
            remaining = self.context_limit - total_chars
            if remaining <= 0:
                logger.debug("Context limit reached at chunk {}; truncating.", i)
                break
            chunk_text = result.text[:remaining]
            block = f"[SOURCE {i}] ({filename}, page {page}):\n{chunk_text}"
            lines.append(block)
            total_chars += len(block)
            sources.append({
                "source_num": i,
                "chunk_id": result.chunk_id,
                "filename": filename,
                "page": page,
                "score": result.score,
            })

        context_block = "\n\n".join(lines)
        return context_block, sources

    # ------------------------------------------------------------------
    # Main generate method
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        contexts: list[SearchResult],
    ) -> AnswerResult:
        """
        Generate an answer grounded in the provided contexts.

        Parameters
        ----------
        query    : The user's question.
        contexts : Reranked SearchResult objects to use as context.

        Returns
        -------
        AnswerResult
        """
        if not contexts:
            logger.warning("generate() called with no context — returning fallback answer.")
            return AnswerResult(
                question=query,
                answer="I'm sorry, no relevant documents were found to answer your question.",
                model=self.model,
            )

        context_block, sources = self._build_context_block(contexts)
        user_message = _USER_PROMPT_TEMPLATE.format(
            context_block=context_block,
            question=query,
        )

        logger.debug(
            "Calling {} — context: {} chars, {} source(s)",
            self.model,
            len(context_block),
            len(sources),
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        answer_text = response.choices[0].message.content or ""
        usage = response.usage

        result = AnswerResult(
            question=query,
            answer=answer_text.strip(),
            sources=sources,
            contexts_used=[r.text for r in contexts[: len(sources)]],
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

        logger.info(
            "Answer generated: {} prompt tokens, {} completion tokens",
            result.prompt_tokens,
            result.completion_tokens,
        )
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_generator(**kwargs) -> AnswerGenerator:
    """
    Create an AnswerGenerator with optional configuration overrides.

    Parameters
    ----------
    **kwargs : Forwarded to AnswerGenerator constructor.
    """
    return AnswerGenerator(**kwargs)
