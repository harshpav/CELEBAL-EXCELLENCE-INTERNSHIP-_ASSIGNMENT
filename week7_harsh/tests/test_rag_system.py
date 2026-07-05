"""
Unit tests for the RAG system.

Run with:
    pytest tests/ -v

Coverage:
    pytest tests/ --cov=rag_system --cov-report=term-missing
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# ingestion tests
# ---------------------------------------------------------------------------

class TestPreprocessText:
    def test_strips_whitespace(self):
        from rag_system.ingestion import preprocess_text
        assert preprocess_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        from rag_system.ingestion import preprocess_text
        assert preprocess_text("hello   world") == "hello world"

    def test_limits_blank_lines(self):
        from rag_system.ingestion import preprocess_text
        result = preprocess_text("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_removes_bom(self):
        from rag_system.ingestion import preprocess_text
        result = preprocess_text("\ufeffhello")
        assert result == "hello"


class TestLoadDocument:
    def test_load_text_file(self, tmp_path):
        from rag_system.ingestion import load_document
        f = tmp_path / "test.txt"
        f.write_text("Hello, this is a test document.", encoding="utf-8")
        docs = load_document(f)
        assert len(docs) == 1
        assert "Hello" in docs[0].text
        assert docs[0].metadata["file_type"] == "txt"

    def test_load_markdown_file(self, tmp_path):
        from rag_system.ingestion import load_document
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nThis is a markdown document.", encoding="utf-8")
        docs = load_document(f)
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "md"

    def test_raises_on_missing_file(self):
        from rag_system.ingestion import load_document
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/file.txt")

    def test_raises_on_unsupported_type(self, tmp_path):
        from rag_system.ingestion import load_document
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(f)

    def test_load_directory(self, tmp_path):
        from rag_system.ingestion import load_directory
        (tmp_path / "a.txt").write_text("Document A content for testing.")
        (tmp_path / "b.md").write_text("Document B content for testing.")
        docs = load_directory(tmp_path, recursive=False)
        assert len(docs) == 2


# ---------------------------------------------------------------------------
# chunking tests
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_splits_long_document(self):
        from rag_system.chunking import RecursiveChunker
        from rag_system.ingestion import Document

        long_text = "Sentence number {}. " * 100
        long_text = long_text.format(*range(100))
        doc = Document(text=long_text, metadata={"source": "test.txt", "page": 1})
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split([doc])
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 220  # allow slight overlap

    def test_chunk_ids_are_unique(self):
        from rag_system.chunking import RecursiveChunker
        from rag_system.ingestion import Document

        text = "Alpha beta gamma delta. " * 50
        doc = Document(text=text, metadata={"source": "test.txt", "page": 1})
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([doc])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_no_empty_chunks(self):
        from rag_system.chunking import RecursiveChunker
        from rag_system.ingestion import Document

        doc = Document(text="Short text.", metadata={"source": "test.txt", "page": 1})
        chunker = RecursiveChunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.split([doc])
        assert all(c.text.strip() for c in chunks)

    def test_metadata_propagates(self):
        from rag_system.chunking import RecursiveChunker
        from rag_system.ingestion import Document

        doc = Document(
            text="Text content for testing metadata propagation. " * 10,
            metadata={"source": "report.pdf", "page": 3, "filename": "report.pdf"},
        )
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([doc])
        for c in chunks:
            assert c.metadata["source"] == "report.pdf"
            assert c.metadata["page"] == 3


# ---------------------------------------------------------------------------
# embeddings tests (mocked to avoid API calls)
# ---------------------------------------------------------------------------

class TestCreateEmbedder:
    def test_factory_returns_openai_embedder(self):
        from rag_system.embeddings import OpenAIEmbedder, create_embedder
        with patch("rag_system.embeddings.OpenAI"):
            emb = create_embedder("openai")
            assert isinstance(emb, OpenAIEmbedder)

    def test_factory_raises_on_unknown_backend(self):
        from rag_system.embeddings import create_embedder
        with pytest.raises(ValueError, match="Unknown"):
            create_embedder("nonexistent")

    def test_local_embedder_embed_returns_correct_shape(self):
        """Smoke-test with a mocked SentenceTransformer."""
        import numpy as np
        from rag_system.embeddings import LocalEmbedder

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.zeros((2, 384))

        with patch("rag_system.embeddings.SentenceTransformer", return_value=mock_model):
            emb = LocalEmbedder(model_name="test-model")
            vectors = emb.embed(["hello", "world"])
            assert len(vectors) == 2
            assert len(vectors[0]) == 384


# ---------------------------------------------------------------------------
# vector store tests (uses real ChromaDB with temp dir)
# ---------------------------------------------------------------------------

class TestVectorStore:
    def _make_store(self, tmp_path):
        from rag_system.vector_store import VectorStore
        return VectorStore(persist_dir=tmp_path / "chroma", collection="test_col")

    def _make_chunks(self, n=3):
        from rag_system.chunking import Chunk
        return [
            Chunk(
                text=f"This is chunk number {i} about machine learning.",
                metadata={"source": "test.txt", "filename": "test.txt", "page": 1, "chunk_index": i},
            )
            for i in range(n)
        ]

    def test_empty_store_count_is_zero(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.count() == 0

    def test_add_and_count(self, tmp_path):
        import numpy as np

        store = self._make_store(tmp_path)
        chunks = self._make_chunks(3)
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [np.zeros(384).tolist() for _ in chunks]
        stored = store.add_chunks(chunks, embedder=mock_embedder)
        assert stored == 3
        assert store.count() == 3

    def test_idempotent_upsert(self, tmp_path):
        """Adding the same chunks twice should not increase count."""
        import numpy as np

        store = self._make_store(tmp_path)
        chunks = self._make_chunks(2)
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [np.zeros(384).tolist() for _ in chunks]
        store.add_chunks(chunks, embedder=mock_embedder)
        store.add_chunks(chunks, embedder=mock_embedder)
        assert store.count() == 2  # not 4

    def test_clear_empties_store(self, tmp_path):
        import numpy as np

        store = self._make_store(tmp_path)
        chunks = self._make_chunks(2)
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [np.zeros(384).tolist() for _ in chunks]
        store.add_chunks(chunks, embedder=mock_embedder)
        store.clear()
        assert store.count() == 0


# ---------------------------------------------------------------------------
# retrieval tests
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _setup(self, tmp_path):
        import numpy as np
        from rag_system.chunking import Chunk
        from rag_system.retrieval import HybridRetriever
        from rag_system.vector_store import VectorStore

        store = VectorStore(persist_dir=tmp_path / "chroma", collection="ret_test")
        chunks = [
            Chunk(
                text=f"RAG combines retrieval and generation. Topic {i}.",
                metadata={"source": "doc.txt", "filename": "doc.txt", "page": 1, "chunk_index": i},
            )
            for i in range(5)
        ]
        mock_emb = MagicMock()
        mock_emb.embed.return_value = [np.random.rand(384).tolist() for _ in chunks]
        mock_emb.embed_one.return_value = np.random.rand(384).tolist()
        store.add_chunks(chunks, embedder=mock_emb)

        retriever = HybridRetriever(store, mock_emb, top_k=3, bm25_weight=0.4)
        return retriever

    def test_search_returns_results(self, tmp_path):
        retriever = self._setup(tmp_path)
        results = retriever.search("What is RAG?")
        assert len(results) > 0
        assert len(results) <= 3

    def test_results_have_scores(self, tmp_path):
        retriever = self._setup(tmp_path)
        results = retriever.search("retrieval and generation")
        for r in results:
            assert isinstance(r.score, float)


# ---------------------------------------------------------------------------
# reranker tests
# ---------------------------------------------------------------------------

class TestIdentityReranker:
    def test_truncates_to_top_n(self):
        from rag_system.reranker import IdentityReranker
        from rag_system.vector_store import SearchResult

        results = [
            SearchResult(chunk_id=str(i), text=f"chunk {i}", metadata={}, score=float(i))
            for i in range(10)
        ]
        reranker = IdentityReranker()
        top = reranker.rerank("query", results, top_n=3)
        assert len(top) == 3

    def test_empty_input(self):
        from rag_system.reranker import IdentityReranker
        reranker = IdentityReranker()
        assert reranker.rerank("query", []) == []


# ---------------------------------------------------------------------------
# generation tests
# ---------------------------------------------------------------------------

class TestAnswerGenerator:
    def test_empty_context_returns_fallback(self):
        """Generator should handle empty context gracefully without calling the LLM."""
        from rag_system.generation import AnswerGenerator

        gen = AnswerGenerator.__new__(AnswerGenerator)
        gen.model = "gpt-4o-mini"
        gen.max_tokens = 512
        gen.temperature = 0.0
        gen.context_limit = 12000
        gen._client = MagicMock()

        result = gen.generate(query="What is RAG?", contexts=[])
        assert "don't have enough information" in result.answer.lower() or "no relevant" in result.answer.lower()
        gen._client.chat.completions.create.assert_not_called()

    def test_build_context_block(self):
        from rag_system.generation import AnswerGenerator
        from rag_system.vector_store import SearchResult

        gen = AnswerGenerator.__new__(AnswerGenerator)
        gen.context_limit = 5000

        results = [
            SearchResult(
                chunk_id="id1",
                text="RAG stands for Retrieval-Augmented Generation.",
                metadata={"filename": "paper.pdf", "page": 2},
                score=0.95,
            )
        ]
        block, sources = gen._build_context_block(results)
        assert "[SOURCE 1]" in block
        assert "paper.pdf" in block
        assert sources[0]["source_num"] == 1
        assert sources[0]["page"] == 2


# ---------------------------------------------------------------------------
# pipeline integration test (mocked LLM)
# ---------------------------------------------------------------------------

class TestRAGPipelineSmoke:
    def test_ingest_and_query(self, tmp_path):
        """Full smoke test: ingest a text file and query it with mocked embedder + LLM."""
        import numpy as np
        from rag_system.chunking import RecursiveChunker
        from rag_system.generation import AnswerGenerator, AnswerResult
        from rag_system.pipeline import RAGPipeline
        from rag_system.reranker import IdentityReranker
        from rag_system.retrieval import create_retriever
        from rag_system.vector_store import VectorStore

        # Create a test document
        doc_path = tmp_path / "test.txt"
        doc_path.write_text(
            "RAG systems improve factual accuracy by grounding LLM answers in retrieved documents. "
            "They combine dense vector search with sparse keyword search. "
            "Reranking further improves the precision of retrieved passages. " * 5
        )

        # Mock embedder
        mock_emb = MagicMock()
        mock_emb.embed.side_effect = lambda texts: [np.random.rand(384).tolist() for _ in texts]
        mock_emb.embed_one.side_effect = lambda t: np.random.rand(384).tolist()

        # Mock generator
        mock_gen = MagicMock(spec=AnswerGenerator)
        mock_gen.model = "gpt-4o-mini"
        mock_gen.generate.return_value = AnswerResult(
            question="What is RAG?",
            answer="RAG improves accuracy by grounding answers in documents. [SOURCE 1]",
            sources=[{"source_num": 1, "filename": "test.txt", "page": 1, "score": 0.9}],
        )

        store = VectorStore(persist_dir=tmp_path / "chroma", collection="smoke_test")
        pipeline = RAGPipeline(
            embedder=mock_emb,
            vector_store=store,
            retriever=None,
            reranker=IdentityReranker(),
            generator=mock_gen,
        )

        # Ingest
        ingest_results = pipeline.ingest(doc_path)
        assert len(ingest_results) == 1
        assert ingest_results[0].num_stored > 0

        # Query
        answer = pipeline.query("What is RAG?")
        assert answer.answer
        assert "[SOURCE 1]" in answer.answer
