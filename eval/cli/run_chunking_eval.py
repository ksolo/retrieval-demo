"""CLI script to run chunking strategy evaluation."""

import argparse
import logging
from dotenv import load_dotenv
from langsmith import Client

from retrieval_demo.dataloader.chunking.recursive import RecursiveTextSplitterStrategy
from retrieval_demo.dataloader.chunking.semantic import SemanticChunkerStrategy
from retrieval_demo.vectorstore.client import get_weaviate_client
from retrieval_demo.agent.graph import get_graph
from eval.langsmith_evaluators import create_langsmith_evaluators

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    evaluators = create_langsmith_evaluators(args.judge_model)

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
