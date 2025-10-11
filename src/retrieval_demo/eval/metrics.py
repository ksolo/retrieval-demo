"""Metrics for evaluating retrieval strategies."""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import defaultdict


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval evaluation."""

    eval_id: str
    retrieval_relevance: float  # 0-1, LLM judge score
    groundedness: float  # 0-1, LLM judge score
    recall_at_1: bool  # First chunk from correct document?
    recall_at_3: bool  # Any of top 3 from correct document?
    recall_at_5: bool  # Any of top 5 from correct document?
    precision_at_k: float  # Proportion of relevant chunks in top-k
    latency_ms: float  # Retrieval latency in milliseconds


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple evaluations."""

    avg_retrieval_relevance: float
    avg_groundedness: float
    avg_recall_at_1: float  # Proportion of samples with recall@1
    avg_recall_at_3: float  # Proportion of samples with recall@3
    avg_recall_at_5: float  # Proportion of samples with recall@5
    avg_precision_at_k: float
    avg_latency_ms: float
    total_samples: int


class MetricsCalculator:
    """Calculator for retrieval metrics."""

    def calculate_aggregate(self, metrics: List[RetrievalMetrics]) -> AggregateMetrics:
        """Calculate aggregate metrics from list of sample metrics.

        Args:
            metrics: List of per-sample metrics

        Returns:
            Aggregate metrics across all samples
        """
        if not metrics:
            return AggregateMetrics(
                avg_retrieval_relevance=0.0,
                avg_groundedness=0.0,
                avg_recall_at_1=0.0,
                avg_recall_at_3=0.0,
                avg_recall_at_5=0.0,
                avg_precision_at_k=0.0,
                avg_latency_ms=0.0,
                total_samples=0,
            )

        n = len(metrics)
        return AggregateMetrics(
            avg_retrieval_relevance=sum(m.retrieval_relevance for m in metrics) / n,
            avg_groundedness=sum(m.groundedness for m in metrics) / n,
            avg_recall_at_1=sum(1 for m in metrics if m.recall_at_1) / n,
            avg_recall_at_3=sum(1 for m in metrics if m.recall_at_3) / n,
            avg_recall_at_5=sum(1 for m in metrics if m.recall_at_5) / n,
            avg_precision_at_k=sum(m.precision_at_k for m in metrics) / n,
            avg_latency_ms=sum(m.latency_ms for m in metrics) / n,
            total_samples=n,
        )

    def calculate_by_category(
        self, metrics: List[RetrievalMetrics], categories: List[str]
    ) -> Dict[str, AggregateMetrics]:
        """Calculate aggregate metrics grouped by category.

        Args:
            metrics: List of per-sample metrics
            categories: List of category labels (must match length of metrics)

        Returns:
            Dictionary mapping category name to aggregate metrics
        """
        if len(metrics) != len(categories):
            raise ValueError("metrics and categories must have same length")

        # Group metrics by category
        by_category = defaultdict(list)
        for metric, category in zip(metrics, categories):
            by_category[category].append(metric)

        # Calculate aggregate for each category
        return {
            category: self.calculate_aggregate(category_metrics)
            for category, category_metrics in by_category.items()
        }

    def calculate_recall_at_k(
        self, retrieved_chunks: List[Dict[str, Any]], correct_eval_id: str
    ) -> Tuple[bool, bool, bool]:
        """Calculate recall@1, recall@3, recall@5.

        Args:
            retrieved_chunks: List of retrieved chunks with metadata.document_id
            correct_eval_id: The correct eval_id to match

        Returns:
            Tuple of (recall@1, recall@3, recall@5)
        """
        # Extract document IDs from retrieved chunks
        retrieved_ids = [chunk["metadata"]["document_id"] for chunk in retrieved_chunks]

        # Check if correct document appears in top-k
        recall_at_1 = len(retrieved_ids) >= 1 and retrieved_ids[0] == correct_eval_id
        recall_at_3 = correct_eval_id in retrieved_ids[:3]
        recall_at_5 = correct_eval_id in retrieved_ids[:5]

        return recall_at_1, recall_at_3, recall_at_5

    def calculate_precision_at_k(self, relevant_flags: List[bool], k: int) -> float:
        """Calculate precision@k.

        Args:
            relevant_flags: List of boolean flags indicating relevance for each chunk
            k: Number of top chunks to consider

        Returns:
            Precision@k (proportion of relevant chunks in top-k)
        """
        if k == 0 or not relevant_flags:
            return 0.0

        top_k_flags = relevant_flags[:k]
        return sum(top_k_flags) / len(top_k_flags)
