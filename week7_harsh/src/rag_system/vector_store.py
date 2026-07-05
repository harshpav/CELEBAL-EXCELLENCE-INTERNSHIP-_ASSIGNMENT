"""
Vector Store Module (ChromaDB)
===============================
Persists chunk embeddings in a local ChromaDB collection and exposes
similarity-search operations used by the retrieval pipeline.

Why ChromaDB?
-------------
- Fully local — no docker, no server, no API key.
- Persistent on disk (data/chroma by default).
- Supports metadata filtering, which is essential for multi-document setups.
- Simple Python-native API.

Key design decisions
---------------------
- **Idempotent upsert** — chunks are keyed by ``chunk_id`` (MD5 hash).  Re-ingesting
  the same document is safe; only changed chunks are written.
- **Metadata stored as-is** — all Chunk.metadata keys are stored in ChromaDB's
  ``where`` filter payload so callers can filter by source, page, file_type, etc.
- **Embeddings are external** — the caller (pipeline) provides pre-computed
  embeddings so VectorStore is backend-agnostic with respect to embedding models.

Usage
-----
>>> from rag_system.vector_store import VectorStore
>>> from rag_system.embeddings import create_embedder
>>> embedder = create_embedder("local")
>>> store = VectorStore()
>>> store.add_chunks(chunks, embedder)
>>> results = store.query(query_vector, top_k=10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from rag_system.chunking import Chunk
from rag_system.config import settings
from rag_system.embeddings import BaseEmbedder


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class SearchResult:
    """A single result returned from vector similarity search."""

    __slots__ = ("chunk_id", "text", "metadata", "score")

    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: dict,
        score: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.score = score  # Cosine similarity (higher = better)

    def __repr__(self) -> str:
        return (
            f"SearchResult(score={self.score:.4f}, "
            f"source={self.metadata.get('filename', '?')}, "
            f"text={self.text[:60]!r}…)"
        )


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Wrapper around a ChromaDB persistent collection.

    Parameters
    ----------
    persist_dir  : Directory where ChromaDB stores its data.
    collection   : Name of the ChromaDB collection.
    embedding_fn : Optional embedder used when ``add_chunks`` is called without
                   pre-computed vectors.
    """

    def __init__(
        self,
        persist_dir: Optional[Path | str] = None,
        collection: Optional[str] = None,
        embedding_fn: Optional[BaseEmbedder] = None,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise ImportError("chromadb is required: pip install chromadb") from exc

        self._persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection or settings.chroma_collection
        self.embedding_fn = embedding_fn

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )
        logger.info(
            "VectorStore ready: collection='{}', persist='{}'",
            self._collection_name,
            self._persist_dir,
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        embedder: Optional[BaseEmbedder] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Embed chunks (if vectors not pre-computed) and upsert into ChromaDB.

        Parameters
        ----------
        chunks     : Chunk objects to store.
        embedder   : Embedder to use; falls back to self.embedding_fn.
        batch_size : Number of chunks to upsert per batch (memory control).

        Returns
        -------
        int  Number of chunks actually stored.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — nothing to do.")
            return 0

        embedder = embedder or self.embedding_fn
        if embedder is None:
            raise ValueError(
                "An embedder must be provided via add_chunks(embedder=...) "
                "or VectorStore(embedding_fn=...)."
            )

        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [_sanitise_metadata(c.metadata) for c in batch]

            vectors = embedder.embed(texts)

            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
            total_added += len(batch)
            logger.debug(
                "Upserted batch {}/{} ({} chunks)",
                i // batch_size + 1,
                (len(chunks) - 1) // batch_size + 1,
                len(batch),
            )

        logger.info(
            "add_chunks: {} chunk(s) upserted into '{}'",
            total_added,
            self._collection_name,
        )
        return total_added

    def delete_by_source(self, source: str) -> int:
        """
        Remove all chunks whose metadata.source matches *source*.

        Parameters
        ----------
        source : str  The file path stored in metadata["source"].

        Returns
        -------
        int  Number of chunks deleted.
        """
        results = self._collection.get(where={"source": source}, include=["metadatas"])
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted {} chunk(s) from source '{}'", len(ids), source)
        return len(ids)

    def clear(self) -> None:
        """Delete all documents in the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("VectorStore collection '{}' cleared.", self._collection_name)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_vector: list[float],
        top_k: int = settings.retrieval_top_k,
        where: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Retrieve the top-k most similar chunks by vector similarity.

        Parameters
        ----------
        query_vector : Query embedding (same dimension as stored embeddings).
        top_k        : Number of results to return.
        where        : Optional ChromaDB metadata filter dict.
                       Example: {"file_type": "pdf"}

        Returns
        -------
        list[SearchResult]  Ordered by cosine similarity, descending.
        """
        kwargs: dict = dict(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        if kwargs["n_results"] == 0:
            logger.warning("VectorStore is empty — returning no results.")
            return []

        raw = self._collection.query(**kwargs)

        results: list[SearchResult] = []
        for chunk_id, text, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # ChromaDB cosine distance → similarity: sim = 1 - dist
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=meta,
                    score=float(1.0 - dist),
                )
            )

        return results

    def get_all_texts(self) -> list[str]:
        """Return every stored document text (for BM25 corpus building)."""
        all_data = self._collection.get(include=["documents"])
        return all_data.get("documents") or []

    def get_all(self) -> list[SearchResult]:
        """Return every stored document as SearchResult (no ranking)."""
        all_data = self._collection.get(include=["documents", "metadatas"])
        results = []
        for chunk_id, text, meta in zip(
            all_data.get("ids", []),
            all_data.get("documents", []),
            all_data.get("metadatas", []),
        ):
            results.append(
                SearchResult(chunk_id=chunk_id, text=text, metadata=meta, score=0.0)
            )
        return results

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._collection.count()

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return (
            f"VectorStore(collection='{self._collection_name}', "
            f"count={self.count()}, dir='{self._persist_dir}')"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_metadata(meta: dict) -> dict:
    """
    ChromaDB metadata values must be str, int, float, or bool.
    Convert Path objects and anything else to str.
    """
    sanitised = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            sanitised[k] = v
        elif isinstance(v, Path):
            sanitised[k] = str(v)
        elif v is None:
            continue  # Skip None — ChromaDB does not support null metadata
        else:
            sanitised[k] = str(v)
    return sanitised
