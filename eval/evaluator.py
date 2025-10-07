"""Evaluators for chunking and retrieval strategies."""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from langgraph.graph import StateGraph

from .judges import RetrievalRelevanceJudge, GroundednessJudge
from .metrics import RetrievalMetrics, AggregateMetrics, MetricsCalculator

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """Evaluator for chunking and retrieval strategies."""

    def __init__(
        self,
        graph: StateGraph,
        relevance_judge: RetrievalRelevanceJudge,
        groundedness_judge: GroundednessJudge,
    ):
        """Initialize evaluator with agent graph and judges.

        Args:
            graph: Compiled LangGraph agent
            relevance_judge: Judge for retrieval relevance
            groundedness_judge: Judge for answer groundedness
        """
        self.graph = graph
        self.relevance_judge = relevance_judge
        self.groundedness_judge = groundedness_judge
        self.metrics_calculator = MetricsCalculator()

    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        collection_name: str,
        retrieval_strategy: str,
        topk: int = 5,
    ) -> RetrievalMetrics:
        """Evaluate a single sample.

        Args:
            sample: Sample with eval_id, question, answer, category
            collection_name: Collection to retrieve from
            retrieval_strategy: Retrieval strategy to use
            topk: Number of chunks to retrieve

        Returns:
            RetrievalMetrics for this sample
        """
        eval_id = sample["eval_id"]
        question = sample["question"]

        # Invoke agent graph with timing
        start_time = time.time()

        try:
            result = self.graph.invoke(
                {
                    "query": question,
                    "collection": collection_name,
                    "retrieval_strategy": retrieval_strategy,
                    "topk": topk,
                    "messages": [],
                    "documents": [],
                }
            )
        except Exception as e:
            logger.error(f"Error invoking graph for {eval_id}: {e}")
            # Return zero metrics on error
            return RetrievalMetrics(
                eval_id=eval_id,
                retrieval_relevance=0.0,
                groundedness=0.0,
                recall_at_1=False,
                recall_at_3=False,
                recall_at_5=False,
                precision_at_k=0.0,
                latency_ms=0.0,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Extract documents and generated answer
        documents = result.get("documents", [])
        messages = result.get("messages", [])

        # Get generated answer (last message)
        generated_answer = messages[-1].content if messages else ""

        # Convert documents to chunk format for judges
        retrieved_chunks = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]

        # Judge retrieval relevance
        try:
            retrieval_relevance = self.relevance_judge.judge(
                question=question, retrieved_chunks=retrieved_chunks
            )
        except Exception as e:
            logger.error(f"Error judging relevance for {eval_id}: {e}")
            retrieval_relevance = 0.0

        # Judge groundedness
        try:
            groundedness = self.groundedness_judge.judge(
                answer=generated_answer, retrieved_chunks=retrieved_chunks
            )
        except Exception as e:
            logger.error(f"Error judging groundedness for {eval_id}: {e}")
            groundedness = 0.0

        # Calculate recall@k
        recall_at_1, recall_at_3, recall_at_5 = (
            self.metrics_calculator.calculate_recall_at_k(
                retrieved_chunks=retrieved_chunks, correct_eval_id=eval_id
            )
        )

        # For precision@k, we'd need individual chunk relevance judgments
        # For now, use a simplified version based on document match
        relevant_flags = [
            chunk["metadata"].get("document_id") == eval_id
            for chunk in retrieved_chunks
        ]
        precision_at_k = self.metrics_calculator.calculate_precision_at_k(
            relevant_flags=relevant_flags, k=topk
        )

        return RetrievalMetrics(
            eval_id=eval_id,
            retrieval_relevance=retrieval_relevance,
            groundedness=groundedness,
            recall_at_1=recall_at_1,
            recall_at_3=recall_at_3,
            recall_at_5=recall_at_5,
            precision_at_k=precision_at_k,
            latency_ms=latency_ms,
        )

    def evaluate_strategy(
        self,
        samples: List[Dict[str, Any]],
        collection_name: str,
        retrieval_strategy: str,
        topk: int = 5,
        include_category_breakdown: bool = False,
    ) -> Tuple[
        List[RetrievalMetrics], AggregateMetrics, Optional[Dict[str, AggregateMetrics]]
    ]:
        """Evaluate strategy across all samples.

        Args:
            samples: List of samples to evaluate
            collection_name: Collection to retrieve from
            retrieval_strategy: Retrieval strategy to use
            topk: Number of chunks to retrieve
            include_category_breakdown: Whether to include per-category metrics

        Returns:
            Tuple of (per_sample_metrics, aggregate_metrics, category_metrics)
        """
        logger.info(
            f"Evaluating strategy: {collection_name} with {retrieval_strategy}, {len(samples)} samples"
        )

        # Evaluate each sample
        per_sample_metrics = []
        for i, sample in enumerate(samples):
            logger.info(
                f"Evaluating sample {i + 1}/{len(samples)}: {sample['eval_id']}"
            )

            metrics = self.evaluate_sample(
                sample=sample,
                collection_name=collection_name,
                retrieval_strategy=retrieval_strategy,
                topk=topk,
            )
            per_sample_metrics.append(metrics)

        # Calculate aggregate metrics
        aggregate = self.metrics_calculator.calculate_aggregate(per_sample_metrics)

        # Calculate category breakdown if requested
        category_metrics = None
        if include_category_breakdown:
            categories = [sample["category"] for sample in samples]
            category_metrics = self.metrics_calculator.calculate_by_category(
                metrics=per_sample_metrics, categories=categories
            )

        logger.info(
            f"Strategy evaluation complete: avg_retrieval_relevance={aggregate.avg_retrieval_relevance:.3f}, "
            f"avg_groundedness={aggregate.avg_groundedness:.3f}, "
            f"avg_recall@1={aggregate.avg_recall_at_1:.3f}"
        )

        return per_sample_metrics, aggregate, category_metrics
