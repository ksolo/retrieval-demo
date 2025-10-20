"""Metrics for evaluating retrieval strategies."""

from typing import List, Dict, Any, Tuple


class MetricsCalculator:
    """Calculator for retrieval metrics."""

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
