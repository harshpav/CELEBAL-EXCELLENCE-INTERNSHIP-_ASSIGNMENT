"""
Embedding Module
================
Converts text into dense vector representations (embeddings).

Two backends are supported:

1. **OpenAIEmbedder** — Uses the OpenAI Embeddings API.
   - Default model: ``text-embedding-3-small`` (1536-dim)
   - Batches requests automatically (max 2048 texts per call).
   - Requires OPENAI_API_KEY environment variable.

2. **LocalEmbedder** — Uses ``sentence-transformers`` (BAAI/bge-small-en-v1.5
   by default).  Runs entirely offline, no API key needed.

Both implement the same interface:
    embed(texts: list[str]) -> list[list[float]]

Usage
-----
>>> from rag_system.embeddings import create_embedder
>>> embedder = create_embedder("openai")          # or "local"
>>> vectors = embedder.embed(["Hello world"])
>>> len(vectors[0])   # 1536 for OpenAI small
1536
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from loguru import logger

from rag_system.config import settings


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract embedder interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings.

        Parameters
        ----------
        texts : list[str]  Non-empty list of strings.

        Returns
        -------
        list[list[float]]  One embedding vector per input text.
        """
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience method for a single string."""
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Allow using the embedder as a bare callable (for SemanticChunker)."""
        return self.embed(texts)


# ---------------------------------------------------------------------------
# OpenAI Embedder
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder backed by the OpenAI Embeddings API.

    Parameters
    ----------
    model       : OpenAI embedding model name.
    dimensions  : Vector dimension (model-specific; 1536 for 3-small).
    batch_size  : Max texts per API call (OpenAI limit: 2048).
    api_key     : Overrides OPENAI_API_KEY if provided.
    """

    _BATCH_SIZE = 512  # Conservative to stay well within rate limits

    def __init__(
        self,
        model: str = settings.openai_embedding_model,
        dimensions: int = settings.openai_embedding_dimensions,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai") from exc

        self.model = model
        self._dimension = dimensions
        self._client = OpenAI(api_key=api_key or settings.openai_api_key or None)
        logger.info("OpenAIEmbedder initialised: model={}, dim={}", model, dimensions)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Sanitise: replace empty strings to avoid API errors
        sanitised = [t.replace("\n", " ").strip() or "." for t in texts]

        all_embeddings: list[list[float]] = []
        for i in range(0, len(sanitised), self._BATCH_SIZE):
            batch = sanitised[i: i + self._BATCH_SIZE]
            logger.debug(
                "OpenAI embed batch {}/{} ({} texts)",
                i // self._BATCH_SIZE + 1,
                (len(sanitised) - 1) // self._BATCH_SIZE + 1,
                len(batch),
            )
            response = self._client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self._dimension,
            )
            batch_vecs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_vecs)

        logger.debug("OpenAI embedded {} text(s)", len(all_embeddings))
        return all_embeddings


# ---------------------------------------------------------------------------
# Local Embedder (sentence-transformers)
# ---------------------------------------------------------------------------

class LocalEmbedder(BaseEmbedder):
    """
    Embedder backed by sentence-transformers (runs fully offline).

    Parameters
    ----------
    model_name  : Any model on HuggingFace Hub compatible with sentence-transformers.
                  Default: BAAI/bge-small-en-v1.5 (384-dim, fast, high quality).
    device      : "cpu" | "cuda" | "mps".  Auto-detected if None.
    batch_size  : Texts per forward pass.
    normalize   : Whether to L2-normalise embeddings (recommended for cosine sim).
    """

    def __init__(
        self,
        model_name: str = settings.local_embedding_model,
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info("Loading local embedding model: {}", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("LocalEmbedder ready: model={}, dim={}", model_name, self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        sanitised = [t.strip() or "." for t in texts]
        logger.debug("LocalEmbedder: embedding {} text(s)", len(sanitised))
        vectors = self._model.encode(
            sanitised,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vectors.tolist()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedder(
    backend: Optional[str] = None,
    **kwargs,
) -> BaseEmbedder:
    """
    Create an embedder instance.

    Parameters
    ----------
    backend : "openai" | "local"
              If None, uses settings.embedding_backend.
    **kwargs : Forwarded to the embedder constructor.

    Returns
    -------
    BaseEmbedder subclass instance.

    Examples
    --------
    >>> embedder = create_embedder("local")
    >>> embedder = create_embedder("openai", model="text-embedding-3-large")
    """
    backend = (backend or settings.embedding_backend).lower()
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    if backend == "local":
        return LocalEmbedder(**kwargs)
    raise ValueError(f"Unknown embedding backend '{backend}'. Choose 'openai' or 'local'.")
