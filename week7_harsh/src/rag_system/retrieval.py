"""
Hybrid Search Pipeline
======================
Combines dense vector retrieval with sparse BM25 keyword search using
Reciprocal Rank Fusion (RRF) to produce a single ranked list of results.

Why hybrid search?
------------------
- **Vector search** excels at semantic similarity ("what does this mean?").
- **BM25** excels at exact keyword matching ("find the term X").
- Combining both covers the weaknesses of each approach and consistently
  outperforms either in isolation on most benchmarks.

Algorithm
---------
1. **Dense retrieval** — Query ChromaDB with a query embedding, retrieve top_k.
2. **BM25 sparse retrieval** — Rank the entire stored corpus (or the dense
   candidates) by BM25 score against the raw query tokens.
3. **RRF fusion** — Compute a combined score:

       rrf_score = Σ 1 / (rank_i + k)

   where k=60 (standard RRF constant) and the sum is over each retrieval
   method in which the document appeared.

4. Sort by RRF score and return top_n results.

Configuration
-------------
- ``bm25_weight`` (settings.bm25_weight = 0.4): relative weight given to BM25
  rank vs vector rank in weighted RRF.  Set to 0 for pure vector search.
- ``retrieval_top_k`` (settings.retrieval_top_k = 20): candidates retrieved from
  each method before fusion.

Usage
-----
>>> from rag_system.retrieval import HybridRetriever
>>> retriever = HybridRetriever(vector_store, embedder)
>>> results = retriever.search("What is RAG?", top_k=5)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from loguru import logger
from rank_bm25 import BM25Okapi

from rag_system.config import settings
from rag_system.embeddings import BaseEmbedder
from rag_system.vector_store import SearchResult, VectorStore


# ---------------------------------------------------------------------------
# BM25 helper
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokeniser for BM25.
    Strips punctuation at word boundaries.
    """
    import re
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Retriever that fuses dense vector search with BM25 keyword search.

    Parameters
    ----------
    vector_store   : Populated VectorStore instance.
    embedder       : Embedder to convert the query to a dense vector.
    top_k          : Number of candidates to retrieve from each method.
    bm25_weight    : Weight for BM25 ranks in weighted RRF (0..1).
                     Vector weight = 1 - bm25_weight.
    rrf_k          : Smoothing constant for RRF (default 60, per original paper).
    where          : Optional ChromaDB metadata filter.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
        top_k: int = settings.retrieval_top_k,
        bm25_weight: float = settings.bm25_weight,
        rrf_k: int = 60,
        where: Optional[dict] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.bm25_weight = max(0.0, min(1.0, bm25_weight))
        self.vector_weight = 1.0 - self.bm25_weight
        self.rrf_k = rrf_k
        self.where = where

        # Pre-build BM25 index from the corpus in the vector store
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[SearchResult] = []
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build (or rebuild) the BM25 index from stored documents."""
        all_docs = self.vector_store.get_all()
        if not all_docs:
            logger.warning("VectorStore is empty — BM25 index not built yet.")
            return
        self._bm25_docs = all_docs
        corpus = [_tokenise(d.text) for d in all_docs]
        self._bm25 = BM25Okapi(corpus)
        logger.debug("BM25 index built: {} documents", len(all_docs))

    def refresh_bm25(self) -> None:
        """
        Re-build the BM25 index.  Call this after adding new documents to the
        vector store.
        """
        self._build_bm25_index()

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Execute hybrid search and return fused, ranked results.

        Parameters
        ----------
        query  : Natural-language query string.
        top_k  : Override the instance-level top_k for this call.

        Returns
        -------
        list[SearchResult]  Top results ordered by RRF score (score field
                            contains the final RRF value, not raw similarity).
        """
        k = top_k or self.top_k

        if self.vector_store.count() == 0:
            logger.warning("Vector store is empty — returning no results.")
            return []

        # ---- Dense retrieval ------------------------------------------------
        query_vector = self.embedder.embed_one(query)
        dense_results = self.vector_store.query(
            query_vector=query_vector,
            top_k=k,
            where=self.where,
        )
        logger.debug("Dense retrieval: {} results", len(dense_results))

        # ---- Sparse BM25 retrieval ------------------------------------------
        bm25_results: list[SearchResult] = []
        if self._bm25 is not None:
            query_tokens = _tokenise(query)
            bm25_scores = self._bm25.get_scores(query_tokens)
            # Pair each stored doc with its BM25 score
            scored = sorted(
                zip(self._bm25_docs, bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            bm25_results = [doc for doc, score in scored[:k] if score > 0]
            logger.debug("BM25 retrieval: {} results (non-zero score)", len(bm25_results))

        # ---- Reciprocal Rank Fusion -----------------------------------------
        fused = self._rrf_fuse(dense_results, bm25_results)
        logger.info(
            "Hybrid search: query='{}…', dense={}, bm25={}, fused={}, top_k={}",
            query[:50],
            len(dense_results),
            len(bm25_results),
            len(fused),
            k,
        )
        return fused[:k]

    def _rrf_fuse(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Merge two ranked lists using weighted Reciprocal Rank Fusion.

        RRF score for document d:
            score(d) = vector_weight * (1 / (rank_dense(d) + k))
                     + bm25_weight  * (1 / (rank_sparse(d) + k))
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        # Map chunk_id → SearchResult for de-duplication
        id_to_result: dict[str, SearchResult] = {}

        for rank, result in enumerate(dense, start=1):
            rrf_scores[result.chunk_id] += self.vector_weight / (rank + self.rrf_k)
            id_to_result[result.chunk_id] = result

        for rank, result in enumerate(sparse, start=1):
            rrf_scores[result.chunk_id] += self.bm25_weight / (rank + self.rrf_k)
            id_to_result[result.chunk_id] = result

        # Sort by descending RRF score
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        fused: list[SearchResult] = []
        for chunk_id in sorted_ids:
            result = id_to_result[chunk_id]
            # Replace the raw cosine score with the RRF score for transparency
            fused_result = SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                metadata=result.metadata,
                score=rrf_scores[chunk_id],
            )
            fused.append(fused_result)

        return fused


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_retriever(
    vector_store: VectorStore,
    embedder: BaseEmbedder,
    **kwargs,
) -> HybridRetriever:
    """
    Shortcut to create a HybridRetriever with optional overrides.

    Parameters
    ----------
    vector_store : Populated VectorStore.
    embedder     : Embedder matching the one used during ingestion.
    **kwargs     : Forwarded to HybridRetriever constructor.

    Returns
    -------
    HybridRetriever
    """
    return HybridRetriever(vector_store=vector_store, embedder=embedder, **kwargs)
