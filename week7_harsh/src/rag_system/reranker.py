"""
Reranking Module
================
Re-scores the top-k retrieved chunks to improve precision before the context
is passed to the LLM.

Why reranking?
--------------
The hybrid retriever (vector + BM25) is good at recall: it surfaces most
relevant documents.  But the initial ranking is imprecise because cosine
similarity and BM25 both treat query and document independently.

A cross-encoder re-reads *query + chunk* together and produces a relevance
score, dramatically improving ranking precision.

Two backends are provided:

1. **FlashRankReranker** (default) — Uses the `flashrank` library with a
   lightweight MS-MARCO cross-encoder model (~4 MB, CPU-only, no GPU needed).
   Zero latency overhead, fully offline.

2. **CrossEncoderReranker** — Uses sentence-transformers cross-encoder models
   (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).  More powerful but
   heavier (~80 MB).

Both implement the same ``rerank(query, results, top_n) -> list[SearchResult]``
interface.

Usage
-----
>>> from rag_system.reranker import create_reranker
>>> reranker = create_reranker("flashrank")
>>> top_chunks = reranker.rerank(query, hybrid_results, top_n=5)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

from rag_system.config import settings
from rag_system.vector_store import SearchResult


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Rerank *results* with respect to *query*.

        Parameters
        ----------
        query   : The user query.
        results : Candidate chunks from the retriever.
        top_n   : Maximum results to return after reranking.
                  If None, returns all results in new order.

        Returns
        -------
        list[SearchResult]  Re-scored and sorted results (score field updated).
        """
        ...


# ---------------------------------------------------------------------------
# FlashRank reranker (default — fast, local, small model)
# ---------------------------------------------------------------------------

class FlashRankReranker(BaseReranker):
    """
    Reranker using the ``flashrank`` library.

    Model: ms-marco-MiniLM-L-12-v2 (default) — ~22 MB, runs on CPU.
    Latency: ~2 ms per 10 passages on a modern CPU.

    Parameters
    ----------
    model_name : flashrank model identifier.
    max_length : Max token length for query+passage pair.
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        max_length: int = 512,
    ) -> None:
        try:
            from flashrank import Ranker
        except ImportError as exc:
            raise ImportError("flashrank is required: pip install flashrank") from exc

        self._ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank")
        self._max_length = max_length
        logger.info("FlashRankReranker ready: model={}", model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: Optional[int] = None,
    ) -> list[SearchResult]:
        if not results:
            return []

        from flashrank import RerankRequest

        passages = [
            {"id": r.chunk_id, "text": r.text[:2000]}  # guard against very long chunks
            for r in results
        ]
        request = RerankRequest(query=query, passages=passages)
        ranked = self._ranker.rerank(request)

        # Map chunk_id → original SearchResult
        id_to_result = {r.chunk_id: r for r in results}

        reranked: list[SearchResult] = []
        for item in ranked:
            original = id_to_result.get(item["id"])
            if original is None:
                continue
            reranked.append(
                SearchResult(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    metadata=original.metadata,
                    score=float(item["score"]),
                )
            )

        limit = top_n or settings.rerank_top_n
        logger.info(
            "FlashRank reranked {} → {} results (top_n={})",
            len(results),
            min(len(reranked), limit),
            limit,
        )
        return reranked[:limit]


# ---------------------------------------------------------------------------
# Cross-Encoder reranker (heavier, more accurate)
# ---------------------------------------------------------------------------

class CrossEncoderReranker(BaseReranker):
    """
    Reranker using a sentence-transformers CrossEncoder model.

    Parameters
    ----------
    model_name : HuggingFace model ID.
                 Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (~100 MB).
    device     : "cpu" | "cuda" | "mps".  Auto-detected if None.
    batch_size : Passage pairs per forward pass.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        self._model = CrossEncoder(model_name, device=device)
        self._batch_size = batch_size
        logger.info("CrossEncoderReranker ready: model={}", model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: Optional[int] = None,
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, r.text[:2000]) for r in results]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        scored = sorted(
            zip(results, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        limit = top_n or settings.rerank_top_n
        reranked = []
        for result, score in scored[:limit]:
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    metadata=result.metadata,
                    score=float(score),
                )
            )

        logger.info(
            "CrossEncoder reranked {} → {} results (top_n={})",
            len(results),
            len(reranked),
            limit,
        )
        return reranked


# ---------------------------------------------------------------------------
# Pass-through reranker (no-op, useful in testing/offline mode)
# ---------------------------------------------------------------------------

class IdentityReranker(BaseReranker):
    """No-op reranker — returns the input list, optionally truncated."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: Optional[int] = None,
    ) -> list[SearchResult]:
        limit = top_n or settings.rerank_top_n
        return results[:limit]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_reranker(
    backend: str = "flashrank",
    **kwargs,
) -> BaseReranker:
    """
    Create a reranker instance.

    Parameters
    ----------
    backend : "flashrank" | "cross-encoder" | "none"
    **kwargs : Forwarded to the reranker constructor.

    Returns
    -------
    BaseReranker subclass.

    Examples
    --------
    >>> reranker = create_reranker("flashrank")
    >>> reranker = create_reranker("cross-encoder", model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
    >>> reranker = create_reranker("none")   # pass-through, useful offline
    """
    backend = backend.lower()
    if backend == "flashrank":
        return FlashRankReranker(**kwargs)
    if backend in ("cross-encoder", "crossencoder"):
        return CrossEncoderReranker(**kwargs)
    if backend in ("none", "identity", "passthrough"):
        return IdentityReranker()
    raise ValueError(
        f"Unknown reranker backend '{backend}'. "
        "Choose 'flashrank', 'cross-encoder', or 'none'."
    )
