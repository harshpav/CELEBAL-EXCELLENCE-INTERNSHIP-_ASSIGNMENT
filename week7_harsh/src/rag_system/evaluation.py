"""
Evaluation Module
=================
Measures RAG pipeline quality using RAGAS metrics.

Metrics measured
----------------
- **Faithfulness** — Is the generated answer grounded in the retrieved context?
  (Detects hallucinations.  Higher = fewer fabrications.)

- **Answer Relevancy** — Does the answer actually address the question?
  (Penalises vague or off-topic answers.  Higher = more on-point.)

- **Context Precision** — Are the retrieved chunks relevant to the question?
  (Penalises irrelevant chunks being surfaced.  Higher = better retrieval.)

- **Context Recall** — Does the retrieved context contain enough information
  to answer the question?
  (Requires ground-truth answers.  Higher = better coverage.)

- **Answer Correctness** — How close is the generated answer to the ground truth?
  (Requires ground-truth answers.  Higher = more factually correct.)

All scores are normalised to [0, 1].  Target thresholds (from the RAG Architect
skill):
    context_precision  >= 0.7
    context_recall     >= 0.6
    faithfulness       >= 0.7
    answer_relevancy   >= 0.7

Input format (JSONL)
--------------------
Each line in the evaluation file must be a JSON object with:
    {
        "question": "What is RAG?",
        "ground_truth": "RAG stands for Retrieval-Augmented Generation …"
    }
Optional:
    {
        "answer": "..."  // if you pre-generated answers to evaluate
    }

Usage
-----
>>> from rag_system.evaluation import Evaluator
>>> evaluator = Evaluator(pipeline)
>>> report = evaluator.evaluate(qa_pairs, top_k=20, top_n=5)
>>> print(report["summary"])
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Per-sample result
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """Evaluation result for a single question."""
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: list[str]
    sources: list[dict]

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None

    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluates the full RAG pipeline using RAGAS metrics.

    Parameters
    ----------
    pipeline : RAGPipeline instance to evaluate.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    # ------------------------------------------------------------------
    # Main evaluate method
    # ------------------------------------------------------------------

    def evaluate(
        self,
        qa_pairs: list[dict],
        top_k: int = 20,
        top_n: int = 5,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Run the pipeline on each Q&A pair and compute RAGAS metrics.

        Parameters
        ----------
        qa_pairs          : List of dicts with "question" and "ground_truth" keys.
        top_k             : Retrieval candidates per question.
        top_n             : Contexts passed to LLM after reranking.
        progress_callback : Called after each sample (for CLI progress bars).

        Returns
        -------
        dict with:
            "samples"  : list of per-sample results (dicts)
            "summary"  : aggregate metric scores
            "passed"   : bool — whether all metrics meet thresholds
            "thresholds": the threshold values used
        """
        samples: list[EvalSample] = []
        logger.info("Starting evaluation: {} samples", len(qa_pairs))

        for i, pair in enumerate(qa_pairs):
            question = pair.get("question", "").strip()
            ground_truth = pair.get("ground_truth", "").strip()

            if not question:
                logger.warning("Skipping pair {} — missing question field", i)
                if progress_callback:
                    progress_callback()
                continue

            sample = self._run_single(
                question=question,
                ground_truth=ground_truth,
                top_k=top_k,
                top_n=top_n,
            )
            samples.append(sample)
            logger.debug("Sample {}/{} done — latency={}ms", i + 1, len(qa_pairs), sample.latency_ms)

            if progress_callback:
                progress_callback()

        report = self._compute_ragas_metrics(samples)
        return report

    def _run_single(
        self,
        question: str,
        ground_truth: str,
        top_k: int,
        top_n: int,
    ) -> EvalSample:
        """Execute one Q&A pipeline run and collect inputs/outputs for RAGAS."""
        sample = EvalSample(
            question=question,
            ground_truth=ground_truth,
            generated_answer="",
            retrieved_contexts=[],
            sources=[],
        )
        start = time.perf_counter()
        try:
            result = self.pipeline.query(question, top_k=top_k, top_n=top_n)
            sample.generated_answer = result.answer
            sample.retrieved_contexts = result.contexts_used
            sample.sources = result.sources
        except Exception as exc:
            sample.error = str(exc)
            logger.error("Error evaluating question '{}': {}", question[:80], exc)
        sample.latency_ms = (time.perf_counter() - start) * 1000
        return sample

    # ------------------------------------------------------------------
    # RAGAS metrics
    # ------------------------------------------------------------------

    def _compute_ragas_metrics(self, samples: list[EvalSample]) -> dict:
        """
        Compute RAGAS metrics over all collected samples.

        Falls back to a simple rule-based scoring if RAGAS or OpenAI is not
        available, so the evaluation command always works.
        """
        valid_samples = [s for s in samples if not s.error]
        if not valid_samples:
            return {"samples": [], "summary": {}, "passed": False, "error": "All samples failed."}

        try:
            return self._run_ragas(valid_samples)
        except Exception as exc:
            logger.warning(
                "RAGAS evaluation failed ({}). Falling back to heuristic metrics.", exc
            )
            return self._run_heuristic(valid_samples)

    def _run_ragas(self, samples: list[EvalSample]) -> dict:
        """Use the ragas library for rigorous LLM-based evaluation."""
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": [s.question for s in samples],
            "answer": [s.generated_answer for s in samples],
            "contexts": [s.retrieved_contexts or [""] for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        })

        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            AnswerCorrectness(),
        ]

        logger.info("Running RAGAS evaluation on {} samples…", len(samples))
        results = evaluate(dataset, metrics=metrics)
        scores_df = results.to_pandas()

        # Build per-sample results
        sample_dicts = []
        for i, s in enumerate(samples):
            row = scores_df.iloc[i].to_dict() if len(scores_df) > i else {}
            sample_dicts.append({
                "question": s.question,
                "ground_truth": s.ground_truth,
                "generated_answer": s.generated_answer,
                "sources": s.sources,
                "latency_ms": round(s.latency_ms, 2),
                "faithfulness": _safe_float(row.get("faithfulness")),
                "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                "context_precision": _safe_float(row.get("context_precision")),
                "context_recall": _safe_float(row.get("context_recall")),
                "answer_correctness": _safe_float(row.get("answer_correctness")),
            })

        summary = {
            metric: _safe_float(scores_df[metric].mean())
            for metric in ["faithfulness", "answer_relevancy", "context_precision",
                           "context_recall", "answer_correctness"]
            if metric in scores_df.columns
        }
        summary["avg_latency_ms"] = round(
            sum(s.latency_ms for s in samples) / len(samples), 2
        )

        thresholds = {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        passed = all(
            summary.get(m, 0.0) >= t
            for m, t in thresholds.items()
            if m in summary
        )

        return {
            "samples": sample_dicts,
            "summary": summary,
            "passed": passed,
            "thresholds": thresholds,
            "backend": "ragas",
        }

    def _run_heuristic(self, samples: list[EvalSample]) -> dict:
        """
        Fallback heuristic evaluation — no LLM calls needed.

        Metrics computed:
        - context_coverage   : fraction of samples that retrieved at least 1 chunk
        - answer_non_empty   : fraction of samples with a non-empty answer
        - has_source_citation: fraction of answers that cite a [SOURCE N]
        - avg_latency_ms
        """
        import re

        source_pattern = re.compile(r"\[SOURCE \d+\]")
        total = len(samples)

        ctx_coverage = sum(1 for s in samples if s.retrieved_contexts) / total
        non_empty = sum(1 for s in samples if s.generated_answer.strip()) / total
        cited = sum(1 for s in samples if source_pattern.search(s.generated_answer)) / total
        avg_lat = sum(s.latency_ms for s in samples) / total

        summary = {
            "context_coverage (heuristic)": round(ctx_coverage, 4),
            "answer_non_empty (heuristic)": round(non_empty, 4),
            "source_citation_rate (heuristic)": round(cited, 4),
            "avg_latency_ms": round(avg_lat, 2),
        }

        sample_dicts = [
            {
                "question": s.question,
                "ground_truth": s.ground_truth,
                "generated_answer": s.generated_answer,
                "sources": s.sources,
                "latency_ms": round(s.latency_ms, 2),
                "has_context": bool(s.retrieved_contexts),
                "cited_source": bool(source_pattern.search(s.generated_answer)),
            }
            for s in samples
        ]

        return {
            "samples": sample_dicts,
            "summary": summary,
            "passed": ctx_coverage >= 0.8 and non_empty >= 0.9,
            "thresholds": {"context_coverage": 0.8, "answer_non_empty": 0.9},
            "backend": "heuristic (ragas unavailable)",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> float:
    """Convert numpy/pandas floats to Python float, handle NaN."""
    try:
        f = float(value)
        return round(f, 4) if f == f else 0.0  # NaN check
    except (TypeError, ValueError):
        return 0.0


def load_qa_file(path: Path) -> list[dict]:
    """
    Load a JSONL evaluation file.

    Each line should be a JSON object with at least a "question" key.
    A "ground_truth" key is required for full RAGAS metrics.

    Parameters
    ----------
    path : Path to the .jsonl file.

    Returns
    -------
    list[dict]
    """
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line {} in {} — JSON parse error: {}", line_num, path.name, exc)
    return pairs
