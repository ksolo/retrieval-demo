"""CLI script to run the data ingestion pipeline."""

import logging
import os
import sys
from typing import List

from dotenv import load_dotenv

from ..dataloader.chunking.base import ChunkingStrategy
from ..dataloader.chunking.recursive import RecursiveTextSplitterStrategy
from ..dataloader.chunking.semantic import SemanticChunkerStrategy
from ..vectorstore.client import WeaviateClient
from .ingestion import DataIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_chunking_strategies(openai_api_key: str) -> List[ChunkingStrategy]:
    """Create all chunking strategies for the pipeline."""
    strategies = [
        # Recursive strategies with different chunk sizes and overlaps
        RecursiveTextSplitterStrategy(chunk_size=100, chunk_overlap=25),
        RecursiveTextSplitterStrategy(chunk_size=250, chunk_overlap=50),
        RecursiveTextSplitterStrategy(chunk_size=500, chunk_overlap=100),
        RecursiveTextSplitterStrategy(chunk_size=1000, chunk_overlap=200),
        
        # Semantic strategy
        SemanticChunkerStrategy(openai_api_key=openai_api_key),
    ]
    
    return strategies


def main():
    """Main CLI function."""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    # Initialize components
    logger.info("Initializing Weaviate client...")
    vectorstore_client = WeaviateClient(api_key=openai_api_key)
    
    logger.info("Creating chunking strategies...")
    chunking_strategies = create_chunking_strategies(openai_api_key)
    
    logger.info("Initializing ingestion pipeline...")
    pipeline = DataIngestionPipeline(vectorstore_client)
    
    try:
        # Run the pipeline with a small subset for testing
        logger.info("Starting data ingestion pipeline...")
        stats = pipeline.process_dataset(
            chunking_strategies=chunking_strategies,
            limit=10  # Start with just 10 documents for testing
        )
        
        logger.info("Ingestion completed successfully!")
        logger.info("Final statistics:")
        for collection_name, chunk_count in stats.items():
            logger.info(f"  {collection_name}: {chunk_count} chunks")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    
    finally:
        # Close the client
        vectorstore_client.close()


if __name__ == "__main__":
    main()