"""
RAG Pipeline Orchestrator
=========================
The ``RAGPipeline`` class wires together every module in the system into two
high-level operations:

``ingest(path)``
    Load → preprocess → chunk → embed → store.
    Idempotent: re-ingesting the same file only updates changed chunks.

``query(question)``
    Embed query → hybrid search → rerank → generate answer.

Architecture overview
---------------------

    ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
    │  Ingestion  │────▶│  Chunking   │────▶│  Embeddings  │
    └─────────────┘     └─────────────┘     └──────┬───────┘
                                                    │
                                            ┌───────▼───────┐
                                            │  VectorStore  │
                                            └───────┬───────┘
                                                    │
          ┌─────────────────────────────────────────┼──────────────┐
          │                                         │              │
    ┌─────▼─────┐    ┌───────────────┐    ┌────────▼────────┐     │
    │  Query    │    │  BM25 Index   │    │  Dense Search   │     │
    │ Embedding │    │   (sparse)    │    │  (ChromaDB)     │     │
    └─────┬─────┘    └───────┬───────┘    └────────┬────────┘     │
          │                  │                      │              │
          └──────────────────▼──────────────────────┘              │
                       ┌─────────────┐                            │
                       │  RRF Fusion │                            │
                       └──────┬──────┘                            │
                              │                                   │
                       ┌──────▼──────┐                            │
                       │  Reranker   │                            │
                       └──────┬──────┘                            │
                              │                                   │
                       ┌──────▼──────┐                            │
                       │  Generator  │◀───────────────────────────┘
                       └──────┬──────┘
                              │
                       ┌──────▼──────┐
                       │ AnswerResult│
                       └─────────────┘

Usage
-----
>>> from rag_system.pipeline import RAGPipeline
>>> pipeline = RAGPipeline.from_config()
>>> pipeline.ingest("docs/report.pdf")
>>> result = pipeline.query("What is the main idea?")
>>> print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from rag_system.chunking import Chunk, create_chunker
from rag_system.config import settings
from rag_system.embeddings import BaseEmbedder, create_embedder
from rag_system.generation import AnswerResult, create_generator
from rag_system.ingestion import load_directory, load_document
from rag_system.logging_setup import setup_logging
from rag_system.reranker import BaseReranker, create_reranker
from rag_system.retrieval import HybridRetriever, create_retriever
from rag_system.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Ingestion result
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """Summary of an ingest operation."""
    source: str
    num_documents: int = 0
    num_chunks: int = 0
    num_stored: int = 0
    skipped: bool = False
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"IngestionResult(source={self.source!r}, ERROR={self.error})"
        return (
            f"IngestionResult(source={self.source!r}, "
            f"docs={self.num_documents}, chunks={self.num_chunks}, "
            f"stored={self.num_stored})"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline combining ingestion and query operations.

    Parameters
    ----------
    embedder      : Embedder used for both indexing and query.
    vector_store  : Persistent vector store.
    retriever     : Hybrid retriever (built lazily after first ingest).
    reranker      : Reranker applied after retrieval.
    generator     : LLM answer generator.
    chunking      : "recursive" | "semantic".
    chunk_size    : Target chunk size in characters.
    chunk_overlap : Overlap between consecutive chunks.
    retrieval_top_k : Candidates to retrieve before reranking.
    rerank_top_n  : Final contexts passed to the LLM.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        retriever: Optional[HybridRetriever],
        reranker: BaseReranker,
        generator,  # AnswerGenerator
        chunking: str = "recursive",
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        retrieval_top_k: int = settings.retrieval_top_k,
        rerank_top_n: int = settings.rerank_top_n,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self._retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.chunking = chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        embedding_backend: Optional[str] = None,
        reranker_backend: str = "flashrank",
        chunking: str = "recursive",
        **kwargs,
    ) -> "RAGPipeline":
        """
        Build a RAGPipeline from environment settings.

        Parameters
        ----------
        embedding_backend : "openai" | "local".  Defaults to settings value.
        reranker_backend  : "flashrank" | "cross-encoder" | "none".
        chunking          : "recursive" | "semantic".
        **kwargs          : Overrides for any pipeline parameter.

        Returns
        -------
        RAGPipeline (retriever is None until first document is ingested)
        """
        setup_logging()

        embedder = create_embedder(embedding_backend)
        vector_store = VectorStore(embedding_fn=embedder)
        reranker = create_reranker(reranker_backend)
        generator = create_generator()

        # If the store already has documents, build the retriever immediately
        retriever: Optional[HybridRetriever] = None
        if vector_store.count() > 0:
            retriever = create_retriever(vector_store, embedder)

        return cls(
            embedder=embedder,
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            chunking=chunking,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Retriever lazy init
    # ------------------------------------------------------------------

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            if self.vector_store.count() == 0:
                raise RuntimeError(
                    "No documents have been ingested yet.  "
                    "Call pipeline.ingest(path) before querying."
                )
            self._retriever = create_retriever(
                self.vector_store,
                self.embedder,
                top_k=self.retrieval_top_k,
            )
        return self._retriever

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        path: Union[str, Path],
        recursive: bool = True,
    ) -> list[IngestionResult]:
        """
        Ingest a file or directory of documents.

        Parameters
        ----------
        path      : Path to a PDF/TXT/MD file or a directory.
        recursive : (directory only) whether to scan sub-directories.

        Returns
        -------
        list[IngestionResult]  One entry per source file processed.
        """
        path = Path(path)
        results: list[IngestionResult] = []

        if path.is_dir():
            all_docs_grouped = self._load_directory(path, recursive)
        elif path.is_file():
            all_docs_grouped = self._load_file(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

        for source, docs in all_docs_grouped.items():
            result = IngestionResult(source=source, num_documents=len(docs))
            try:
                chunker = create_chunker(
                    strategy=self.chunking,
                    embed_fn=self.embedder if self.chunking == "semantic" else None,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                chunks: list[Chunk] = chunker.split(docs)
                result.num_chunks = len(chunks)

                stored = self.vector_store.add_chunks(chunks, embedder=self.embedder)
                result.num_stored = stored

                # Rebuild BM25 index after each ingest
                if self._retriever is not None:
                    self._retriever.refresh_bm25()
                else:
                    # Build retriever now that we have data
                    self._retriever = create_retriever(
                        self.vector_store,
                        self.embedder,
                        top_k=self.retrieval_top_k,
                    )

                logger.info(
                    "Ingested '{}': {} doc(s), {} chunk(s), {} stored",
                    source,
                    len(docs),
                    len(chunks),
                    stored,
                )
            except Exception as exc:
                result.error = str(exc)
                logger.error("Failed to ingest '{}': {}", source, exc)

            results.append(result)

        return results

    def _load_file(self, path: Path) -> dict:
        from rag_system.ingestion import load_document as _load
        docs = _load(path)
        return {str(path): docs}

    def _load_directory(self, path: Path, recursive: bool) -> dict:
        """Group docs by their source file."""
        from rag_system.ingestion import load_directory as _load_dir
        all_docs = _load_dir(path, recursive=recursive)
        grouped: dict = {}
        for doc in all_docs:
            src = doc.metadata.get("source", str(path))
            grouped.setdefault(src, []).append(doc)
        return grouped

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
    ) -> AnswerResult:
        """
        Answer a question using the RAG pipeline.

        Parameters
        ----------
        question        : The user's natural-language question.
        top_k           : Override retrieval_top_k for this call.
        top_n           : Override rerank_top_n for this call.
        metadata_filter : ChromaDB metadata filter (e.g. {"file_type": "pdf"}).

        Returns
        -------
        AnswerResult  (answer text + sources + token usage)
        """
        logger.info("RAG query: '{}'", question)

        # Temporarily apply metadata filter if provided
        if metadata_filter and self._retriever:
            self._retriever.where = metadata_filter

        # 1. Retrieve candidates
        candidates = self.retriever.search(question, top_k=top_k or self.retrieval_top_k)

        # Reset filter to avoid affecting future calls
        if metadata_filter and self._retriever:
            self._retriever.where = None

        if not candidates:
            return AnswerResult(
                question=question,
                answer="No relevant documents were found. Please ingest documents first.",
                model=self.generator.model,
            )

        # 2. Rerank
        reranked = self.reranker.rerank(
            query=question,
            results=candidates,
            top_n=top_n or self.rerank_top_n,
        )

        # 3. Generate answer
        answer = self.generator.generate(query=question, contexts=reranked)
        return answer

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all stored documents and reset the retriever."""
        self.vector_store.clear()
        self._retriever = None
        logger.warning("Pipeline cleared — all documents removed.")

    def status(self) -> dict:
        """Return a summary of pipeline state."""
        return {
            "documents_stored": self.vector_store.count(),
            "embedding_backend": settings.embedding_backend,
            "embedding_model": (
                settings.openai_embedding_model
                if settings.embedding_backend == "openai"
                else settings.local_embedding_model
            ),
            "vector_store_dir": str(settings.chroma_persist_dir),
            "chunking_strategy": self.chunking,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_top_k": self.retrieval_top_k,
            "rerank_top_n": self.rerank_top_n,
        }
