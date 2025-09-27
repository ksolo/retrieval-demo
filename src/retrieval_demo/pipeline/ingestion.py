"""Data ingestion pipeline for processing documents through chunking strategies and storing in Weaviate."""

import logging
from typing import List, Dict, Any

from ..dataloader.chunking.base import ChunkingStrategy
from ..dataloader.data.loader import RAGDatasetLoader
from ..vectorstore.client import WeaviateClient

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Pipeline for processing datasets through chunking strategies and storing in vectorstore."""
    
    def __init__(self, vectorstore_client: WeaviateClient):
        """Initialize with vectorstore client."""
        self.vectorstore_client = vectorstore_client
        self.dataset_loader = RAGDatasetLoader()
    
    def process_dataset(
        self, 
        chunking_strategies: List[ChunkingStrategy],
        limit: int = 1000
    ) -> Dict[str, int]:
        """Process dataset through all chunking strategies and insert into vectorstore.
        
        Returns:
            Dictionary mapping collection names to total chunks inserted
        """
        logger.info(f"Starting data ingestion pipeline with {len(chunking_strategies)} strategies")
        
        # Load dataset
        dataset = self.dataset_loader.get_train_subset(limit=limit)
        logger.info(f"Loaded {len(dataset)} documents for processing")
        
        # Create collections for all strategies
        for strategy in chunking_strategies:
            collection_name = strategy.get_collection_name()
            self.vectorstore_client.create_collection(collection_name)
        
        # Process each document through all strategies
        collection_stats = {strategy.get_collection_name(): 0 for strategy in chunking_strategies}
        
        for doc_id, document in enumerate(dataset):
            logger.info(f"Processing document {doc_id + 1}/{len(dataset)}")
            self._process_document(document, doc_id, chunking_strategies, collection_stats)
        
        # Log final statistics
        for collection_name, chunk_count in collection_stats.items():
            logger.info(f"Collection '{collection_name}': {chunk_count} total chunks")
        
        return collection_stats
    
    def _process_document(
        self,
        document: Dict[str, Any],
        document_id: int,
        chunking_strategies: List[ChunkingStrategy],
        collection_stats: Dict[str, int]
    ) -> None:
        """Process single document through all strategies and insert chunks."""
        context = document['context']
        
        for strategy in chunking_strategies:
            collection_name = strategy.get_collection_name()
            
            # Chunk the document
            chunks = strategy.chunk_document(context, document_id)
            
            if chunks:
                # Batch insert chunks for this document
                self.vectorstore_client.batch_insert_chunks(collection_name, chunks)
                collection_stats[collection_name] += len(chunks)
                
                logger.debug(
                    f"Document {document_id}: {len(chunks)} chunks -> {collection_name}"
                )
            else:
                logger.warning(f"No chunks generated for document {document_id} with {collection_name}")
    
    def get_collection_statistics(self, collection_names: List[str]) -> Dict[str, int]:
        """Get current statistics for collections."""
        stats = {}
        for collection_name in collection_names:
            count = self.vectorstore_client.get_collection_count(collection_name)
            stats[collection_name] = count
        return stats
    
    def cleanup_collections(self, collection_names: List[str]) -> None:
        """Delete specified collections (useful for testing/cleanup)."""
        for collection_name in collection_names:
            self.vectorstore_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")