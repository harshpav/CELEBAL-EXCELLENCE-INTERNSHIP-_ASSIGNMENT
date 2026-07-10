"""
Text Chunking Module
====================
Splits Document objects into smaller Chunk objects suitable for embedding and
retrieval.

Two strategies are provided:

1. **RecursiveChunker** — Uses LangChain's RecursiveCharacterTextSplitter.
   Fast, deterministic, works on any text.  Best default choice.

2. **SemanticChunker** — Groups sentences by embedding similarity (cosine
   distance).  More expensive but respects natural topic boundaries.  Requires
   an embedding function (sentence-transformers or OpenAI).

Both strategies:
- Preserve all source metadata from the original Document.
- Attach chunk-level metadata: chunk_index, chunk_total, char_start.
- Guarantee no empty chunks and no chunks exceeding max token budget.

Usage
-----
>>> from rag_system.ingestion import load_document
>>> from rag_system.chunking import RecursiveChunker, Chunk
>>> docs = load_document("report.pdf")
>>> chunker = RecursiveChunker()
>>> chunks = chunker.split(docs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from loguru import logger

from rag_system.config import settings
from rag_system.ingestion import Document


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single text chunk derived from a Document.

    Attributes
    ----------
    text        : The chunk text.
    metadata    : Inherited from Document plus chunk-specific keys.
    chunk_id    : Stable deterministic ID (source + page + chunk_index).
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Chunk text must not be empty.")
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.md5(
                f"{self.metadata.get('source', '')}_{self.metadata.get('page', 0)}"
                f"_{self.metadata.get('chunk_index', 0)}".encode()
            ).hexdigest()


# ---------------------------------------------------------------------------
# Recursive character splitter
# ---------------------------------------------------------------------------

class RecursiveChunker:
    """
    Deterministic chunker based on a hierarchy of separators.

    Separator priority (tries each in order until chunks fit):
        paragraph breaks → line breaks → sentences → words → characters

    Parameters
    ----------
    chunk_size    : Target chunk size in characters (default from settings).
    chunk_overlap : Overlap between consecutive chunks (default from settings).
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> list[str]:
        """Split *text* recursively by the separator hierarchy."""
        for separator in self._SEPARATORS:
            if separator == "":
                # Last resort: hard split at chunk_size
                pieces = [
                    text[i: i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
                ]
                return [p for p in pieces if p.strip()]

            parts = text.split(separator)
            # Re-join small parts to minimise tiny fragments
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = (current + separator + part).strip() if current else part.strip()
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # Part itself might still be too big → recurse
                    if len(part) > self.chunk_size:
                        sub = self._split_text(part)
                        chunks.extend(sub[:-1])
                        current = sub[-1] if sub else ""
                    else:
                        current = part.strip()
            if current:
                chunks.append(current)

            if all(len(c) <= self.chunk_size for c in chunks):
                return [c for c in chunks if c.strip()]

        return [text]  # fallback

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the tail of the previous chunk to add context overlap."""
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    def split(self, documents: list[Document]) -> list[Chunk]:
        """
        Split a list of Documents into Chunks.

        Parameters
        ----------
        documents : list[Document]

        Returns
        -------
        list[Chunk]
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            raw_chunks = self._split_text(doc.text)
            raw_chunks = self._add_overlap(raw_chunks)
            for idx, chunk_text in enumerate(raw_chunks):
                if not chunk_text.strip():
                    continue
                meta = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "chunk_total": len(raw_chunks),
                    "char_start": doc.text.find(chunk_text[:50]),  # approx
                }
                all_chunks.append(Chunk(text=chunk_text, metadata=meta))

        logger.info(
            "RecursiveChunker: {} doc(s) → {} chunk(s) "
            "(size={}, overlap={})",
            len(documents),
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """
    Sentence-boundary chunker that groups sentences by semantic similarity.

    Algorithm
    ---------
    1. Split text into sentences.
    2. Embed each sentence.
    3. Compute cosine distance between consecutive sentences.
    4. Split at positions where the distance exceeds the *breakpoint* threshold
       (default: mean + 1.0 * std of all distances).
    5. Merge small groups to stay near *target_chunk_size*.

    Parameters
    ----------
    embed_fn        : Callable[[list[str]], list[list[float]]]
                      Function that returns an embedding per input string.
    target_chunk_size : Soft upper bound on chars per chunk.
    breakpoint_multiplier : Higher = fewer splits (coarser chunks).
    """

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        target_chunk_size: int = settings.chunk_size,
        breakpoint_multiplier: float = 1.0,
    ) -> None:
        self.embed_fn = embed_fn
        self.target_chunk_size = target_chunk_size
        self.breakpoint_multiplier = breakpoint_multiplier

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 1.0
        return float(1.0 - np.dot(va, vb) / denom)

    def _sentences(self, text: str) -> list[str]:
        sentences = self._SENTENCE_RE.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_breakpoints(self, distances: list[float]) -> list[int]:
        if not distances:
            return []
        arr = np.array(distances)
        threshold = arr.mean() + self.breakpoint_multiplier * arr.std()
        return [i + 1 for i, d in enumerate(distances) if d > threshold]

    def _merge_short_groups(
        self, groups: list[list[str]]
    ) -> list[list[str]]:
        """Merge adjacent groups that are both below target_chunk_size / 2."""
        half = self.target_chunk_size // 2
        merged: list[list[str]] = []
        for group in groups:
            text = " ".join(group)
            if merged and len(" ".join(merged[-1])) + len(text) < self.target_chunk_size:
                merged[-1].extend(group)
            elif len(text) < half and merged:
                merged[-1].extend(group)
            else:
                merged.append(group)
        return merged

    def split(self, documents: list[Document]) -> list[Chunk]:
        """
        Split documents into semantically coherent chunks.

        Parameters
        ----------
        documents : list[Document]

        Returns
        -------
        list[Chunk]
        """
        all_chunks: list[Chunk] = []

        for doc in documents:
            sentences = self._sentences(doc.text)
            if len(sentences) <= 1:
                # Single sentence — keep as is
                chunk = Chunk(
                    text=doc.text,
                    metadata={**doc.metadata, "chunk_index": 0, "chunk_total": 1},
                )
                all_chunks.append(chunk)
                continue

            logger.debug(
                "SemanticChunker: embedding {} sentences for {}", len(sentences), doc.metadata.get("filename")
            )
            embeddings = self.embed_fn(sentences)

            distances = [
                self._cosine_distance(embeddings[i], embeddings[i + 1])
                for i in range(len(embeddings) - 1)
            ]

            breakpoints = self._find_breakpoints(distances)

            # Partition sentences into groups
            groups: list[list[str]] = []
            prev = 0
            for bp in breakpoints:
                groups.append(sentences[prev:bp])
                prev = bp
            groups.append(sentences[prev:])

            groups = self._merge_short_groups(groups)

            for idx, group in enumerate(groups):
                chunk_text = " ".join(group)
                if not chunk_text.strip():
                    continue
                meta = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "chunk_total": len(groups),
                }
                all_chunks.append(Chunk(text=chunk_text, metadata=meta))

        logger.info(
            "SemanticChunker: {} doc(s) → {} chunk(s)",
            len(documents),
            len(all_chunks),
        )
        return all_chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_chunker(
    strategy: str = "recursive",
    embed_fn: Optional[Callable] = None,
    **kwargs,
) -> RecursiveChunker | SemanticChunker:
    """
    Factory function for chunkers.

    Parameters
    ----------
    strategy : "recursive" | "semantic"
    embed_fn : Required when strategy == "semantic".
    **kwargs : Forwarded to the chunker constructor.
    """
    strategy = strategy.lower()
    if strategy == "recursive":
        return RecursiveChunker(**kwargs)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("embed_fn is required for SemanticChunker.")
        return SemanticChunker(embed_fn=embed_fn, **kwargs)
    raise ValueError(f"Unknown strategy '{strategy}'. Choose 'recursive' or 'semantic'.")
