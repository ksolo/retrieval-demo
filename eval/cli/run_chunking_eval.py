"""CLI script to run chunking strategy evaluation."""

import argparse
import logging
from dotenv import load_dotenv
from langsmith import Client

from retrieval_demo.dataloader.chunking.recursive import RecursiveTextSplitterStrategy
from retrieval_demo.dataloader.chunking.semantic import SemanticChunkerStrategy
from retrieval_demo.vectorstore.client import get_weaviate_client
from retrieval_demo.agent.graph import get_graph
from eval.judges import RetrievalRelevanceJudge, GroundednessJudge
from eval.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_evaluators(judge_model: str):
    """Create evaluator functions for LangSmith."""
    relevance_judge = RetrievalRelevanceJudge(model=judge_model)
    groundedness_judge = GroundednessJudge(model=judge_model)
    metrics_calculator = MetricsCalculator()

    def retrieval_relevance_evaluator(run, example):
        """Evaluate retrieval relevance."""
        # Extract from run outputs
        documents = run.outputs.get("documents", [])
        question = example.inputs["question"]

        # Convert to chunks format
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


def main():
    """Run chunking strategy evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate chunking strategies")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rag-eval-dataset",
        help="LangSmith dataset name (default: rag-eval-dataset)",
    )
    parser.add_argument(
        "--retrieval-strategy",
        type=str,
        default="semantic",
        choices=["semantic", "rerank", "multiquery", "hybrid"],
        help="Retrieval strategy to use (default: semantic)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5-mini",
        help="Model to use for LLM judges (default: gpt-5-mini)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    logger.info("Starting chunking strategy evaluation...")

    # Step 1: Verify collections exist
    logger.info("Step 1: Verifying vector store collections")
    chunking_strategies = [
        RecursiveTextSplitterStrategy(chunk_size=200, chunk_overlap=20),
        RecursiveTextSplitterStrategy(chunk_size=500, chunk_overlap=50),
        RecursiveTextSplitterStrategy(chunk_size=1000, chunk_overlap=100),
        RecursiveTextSplitterStrategy(chunk_size=2000, chunk_overlap=200),
        SemanticChunkerStrategy(openai_api_key=None),
    ]

    weaviate_client = get_weaviate_client()
    collection_names = [s.get_collection_name() for s in chunking_strategies]

    for collection_name in collection_names:
        count = weaviate_client.get_collection_count(collection_name)
        if count == 0:
            logger.error(
                f"Collection '{collection_name}' not found or empty! "
                "Run 'prepare-eval-dataset' first to ingest data."
            )
            return 1
        logger.info(f"  ✓ {collection_name}: {count} chunks")

    # Step 2: Initialize graph and evaluators
    logger.info("Step 2: Initializing graph and evaluators")
    graph = get_graph()
    evaluators = create_evaluators(args.judge_model)

    # Step 3: Run evaluation for each chunking strategy
    logger.info("Step 3: Running evaluations")
    langsmith_client = Client()

    def create_target(collection_name):
        """Create a target function for this collection."""

        def target(inputs):
            return graph.invoke(
                {
                    "query": inputs["question"],
                    "collection": collection_name,
                    "retrieval_strategy": args.retrieval_strategy,
                    "topk": args.topk,
                    "messages": [],
                    "documents": [],
                }
            )

        return target

    for strategy in chunking_strategies:
        collection_name = strategy.get_collection_name()
        experiment_name = f"chunking_eval_{collection_name}_{args.retrieval_strategy}"

        logger.info(f"\nEvaluating {collection_name}...")
        logger.info(f"  Experiment: {experiment_name}")

        # Run evaluation using LangSmith
        results = langsmith_client.evaluate(
            create_target(collection_name),
            data=args.dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_name,
        )

        logger.info(f"  ✓ Evaluation complete for {collection_name}")
        logger.info(f"    Results: {results}")

    logger.info("\n✓ All chunking evaluations complete!")
    logger.info("\nView detailed results at: https://smith.langchain.com")

    return 0


if __name__ == "__main__":
    exit(main())
