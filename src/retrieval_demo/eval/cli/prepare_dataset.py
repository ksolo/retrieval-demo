"""CLI script to prepare evaluation dataset with verification."""

import argparse
import logging
from dotenv import load_dotenv

from retrieval_demo.dataloader.data.loader import RAGDatasetLoader
from retrieval_demo.dataloader.categorization import Categorizer
from retrieval_demo.pipeline.ingestion import DataIngestionPipeline
from retrieval_demo.vectorstore.client import get_weaviate_client
from retrieval_demo.eval.dataset import EvalDatasetManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Prepare evaluation dataset and verify vector store collections."""
    parser = argparse.ArgumentParser(
        description="Prepare evaluation dataset and verify collections"
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=5,
        help="Maximum number of categories (default: 5)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rag-eval-dataset",
        help="LangSmith dataset name (default: rag-eval-dataset)",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="eval/data/categorization_cache.json",
        help="Path to categorization cache (default: eval/data/categorization_cache.json)",
    )
    parser.add_argument(
        "--clear-vectorstore",
        action="store_true",
        help="Clear all collections in the vector store before ingestion",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    logger.info("Starting dataset preparation...")

    # Step 0: Clear vector store if requested
    if args.clear_vectorstore:
        logger.info("Step 0: Clearing vector store")
        weaviate_client = get_weaviate_client()
        weaviate_client.delete_all_collections()
        logger.info("✓ Vector store cleared")

    # Step 1: Load and categorize dataset
    logger.info("Step 1: Loading and categorizing dataset")
    loader = RAGDatasetLoader()
    categorizer = Categorizer(model="gpt-5-mini")

    # Hardcoded to 100 samples for demo purposes
    TARGET_SAMPLES = 100

    eval_samples = loader.get_categorized_stratified_sample(
        samples_per_category=TARGET_SAMPLES,
        cache_path=args.cache_path,
        max_categories=args.max_categories,
        categorizer=categorizer,
    )

    logger.info(f"Prepared {len(eval_samples)} evaluation samples")

    # Step 2: Upload to LangSmith
    logger.info("Step 2: Uploading to LangSmith")
    dataset_manager = EvalDatasetManager()

    dataset_id = dataset_manager.create_or_update_dataset(
        dataset_name=args.dataset_name,
        samples=eval_samples,
        description=f"RAG evaluation dataset with {len(eval_samples)} stratified samples",
    )

    logger.info(f"Dataset uploaded to LangSmith: {dataset_id}")

    # Step 3: Verify LangSmith dataset count
    logger.info("Step 3: Verifying LangSmith dataset")
    langsmith_count = dataset_manager.get_dataset_count(args.dataset_name)

    if langsmith_count != len(eval_samples):
        logger.error(
            f"LangSmith count mismatch: expected {len(eval_samples)}, got {langsmith_count}"
        )
        return 1

    logger.info(f"✓ LangSmith dataset verified: {langsmith_count} samples")

    # Step 4: Ingest into vector store collections
    logger.info("Step 4: Ingesting documents into vector store")
    weaviate_client = get_weaviate_client()
    pipeline = DataIngestionPipeline(vectorstore_client=weaviate_client)

    # Define chunking strategies to evaluate
    from retrieval_demo.dataloader.chunking.recursive import (
        RecursiveTextSplitterStrategy,
    )
    from retrieval_demo.dataloader.chunking.semantic import SemanticChunkerStrategy

    chunking_strategies = [
        RecursiveTextSplitterStrategy(chunk_size=200, chunk_overlap=20),
        RecursiveTextSplitterStrategy(chunk_size=500, chunk_overlap=50),
        RecursiveTextSplitterStrategy(chunk_size=1000, chunk_overlap=100),
        RecursiveTextSplitterStrategy(chunk_size=2000, chunk_overlap=200),
        SemanticChunkerStrategy(openai_api_key=None),  # Uses env var
    ]

    for strategy in chunking_strategies:
        collection_name = strategy.get_collection_name()
        logger.info(f"  Ingesting documents for {collection_name}")

        pipeline.process_dataset(
            chunking_strategies=[strategy],
            eval_samples=eval_samples,
        )

        # Verify collection count
        stats = pipeline.get_collection_statistics([collection_name])
        logger.info(f"    ✓ {stats[collection_name]} chunks")

    logger.info("✓ Dataset preparation complete!")
    logger.info(
        f"  - {len(eval_samples)} samples in LangSmith dataset '{args.dataset_name}'"
    )
    logger.info(f"  - Categorization cache saved to {args.cache_path}")

    return 0


if __name__ == "__main__":
    exit(main())
