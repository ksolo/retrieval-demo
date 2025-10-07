"""LangSmith evaluator functions for retrieval evaluation."""

from eval.judges import RetrievalRelevanceJudge, GroundednessJudge
from eval.metrics import MetricsCalculator


def create_langsmith_evaluators(judge_model: str = "gpt-5-mini"):
    """Create evaluator functions for LangSmith evaluation.

    Args:
        judge_model: Model to use for LLM judges

    Returns:
        List of evaluator functions
    """
    relevance_judge = RetrievalRelevanceJudge(model=judge_model)
    groundedness_judge = GroundednessJudge(model=judge_model)
    metrics_calculator = MetricsCalculator()

    def retrieval_relevance_evaluator(run, example):
        """Evaluate retrieval relevance."""
        documents = run.outputs.get("documents", [])
        question = example.inputs["question"]
        chunks = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]
        score = relevance_judge.judge(question=question, retrieved_chunks=chunks)
        return {"key": "retrieval_relevance", "score": score}

    def groundedness_evaluator(run, example):
        """Evaluate groundedness."""
        documents = run.outputs.get("documents", [])
        messages = run.outputs.get("messages", [])
        answer = messages[-1].content if messages else ""
        chunks = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]
        score = groundedness_judge.judge(answer=answer, retrieved_chunks=chunks)
        return {"key": "groundedness", "score": score}

    def recall_at_1_evaluator(run, example):
        """Evaluate recall@1."""
        documents = run.outputs.get("documents", [])
        eval_id = example.metadata["eval_id"]
        chunks = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]
        recall_1, _, _ = metrics_calculator.calculate_recall_at_k(
            retrieved_chunks=chunks, correct_eval_id=eval_id
        )
        return {"key": "recall_at_1", "score": 1.0 if recall_1 else 0.0}

    def recall_at_3_evaluator(run, example):
        """Evaluate recall@3."""
        documents = run.outputs.get("documents", [])
        eval_id = example.metadata["eval_id"]
        chunks = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]
        _, recall_3, _ = metrics_calculator.calculate_recall_at_k(
            retrieved_chunks=chunks, correct_eval_id=eval_id
        )
        return {"key": "recall_at_3", "score": 1.0 if recall_3 else 0.0}

    def recall_at_5_evaluator(run, example):
        """Evaluate recall@5."""
        documents = run.outputs.get("documents", [])
        eval_id = example.metadata["eval_id"]
        chunks = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]
        _, _, recall_5 = metrics_calculator.calculate_recall_at_k(
            retrieved_chunks=chunks, correct_eval_id=eval_id
        )
        return {"key": "recall_at_5", "score": 1.0 if recall_5 else 0.0}

    def precision_at_k_evaluator(run, example):
        """Evaluate precision@k."""
        documents = run.outputs.get("documents", [])
        eval_id = example.metadata["eval_id"]
        chunks = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]
        relevant_flags = [
            chunk["metadata"].get("document_id") == eval_id for chunk in chunks
        ]
        precision = metrics_calculator.calculate_precision_at_k(
            relevant_flags=relevant_flags, k=len(chunks)
        )
        return {"key": "precision_at_k", "score": precision}

    return [
        retrieval_relevance_evaluator,
        groundedness_evaluator,
        recall_at_1_evaluator,
        recall_at_3_evaluator,
        recall_at_5_evaluator,
        precision_at_k_evaluator,
    ]
