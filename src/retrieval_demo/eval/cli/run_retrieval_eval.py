"""CLI script to run retrieval strategy evaluation."""

import argparse
import logging
from dotenv import load_dotenv
from langsmith import Client

from retrieval_demo.vectorstore.client import get_weaviate_client
from retrieval_demo.agent.graph import get_graph
from retrieval_demo.eval.langsmith_evaluators import create_langsmith_evaluators

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run retrieval strategy evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rag-eval-dataset",
        help="LangSmith dataset name (default: rag-eval-dataset)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        required=True,
        help="Collection name to use (e.g., chunks_recursive_500_50)",
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

    logger.info("Starting retrieval strategy evaluation...")

    # Step 1: Verify collection exists
    logger.info("Step 1: Verifying vector store collection")
    weaviate_client = get_weaviate_client()
    count = weaviate_client.get_collection_count(args.collection_name)

    if count == 0:
        logger.error(
            f"Collection '{args.collection_name}' not found or empty! "
            "Run 'prepare-eval-dataset' first to ingest data."
        )
        return 1

    logger.info(f"  ✓ {args.collection_name}: {count} chunks")

    # Step 2: Define retrieval strategies to evaluate
    logger.info("Step 2: Defining retrieval strategies")
    retrieval_strategies = [
        "semantic", # if you ran the chunking eval this one is redundant and not required
        "rerank",
        "multiquery",
        "hybrid"
    ]

    logger.info(f"Evaluating {len(retrieval_strategies)} retrieval strategies")

    # Step 3: Initialize graph and evaluators
    logger.info("Step 3: Initializing graph and evaluators")
    graph = get_graph()
    evaluators = create_langsmith_evaluators(args.judge_model)

    # Step 4: Run evaluation for each retrieval strategy
    logger.info("Step 4: Running evaluations")
    langsmith_client = Client()

    def create_target(retrieval_strategy):
        """Create a target function for this retrieval strategy."""

        def target(inputs):
            return graph.invoke(
                {
                    "query": inputs["question"],
                    "collection": args.collection_name,
                    "retrieval_strategy": retrieval_strategy,
                    "topk": args.topk,
                    "messages": [],
                    "documents": [],
                }
            )

        return target

    for retrieval_strategy in retrieval_strategies:
        experiment_name = f"retrieval_eval_{args.collection_name}_{retrieval_strategy}"

        logger.info(f"\nEvaluating {retrieval_strategy} strategy...")
        logger.info(f"  Collection: {args.collection_name}")
        logger.info(f"  Experiment: {experiment_name}")

        # Run evaluation using LangSmith
        results = langsmith_client.evaluate(
            create_target(retrieval_strategy),
            data=args.dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_name,
        )

        logger.info(f"  ✓ Evaluation complete for {retrieval_strategy}")
        logger.info(f"    Results: {results}")

    logger.info("\n✓ All retrieval evaluations complete!")
    logger.info("\nView detailed results at: https://smith.langchain.com")

    return 0


if __name__ == "__main__":
    exit(main())
