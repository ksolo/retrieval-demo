"""Weaviate client wrapper for batch operations and collection management."""

import logging
import os
from functools import cache
from typing import List, Optional, Any, Dict
import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery
from dotenv import load_dotenv

from ..dataloader.chunking.base import Chunk

logger = logging.getLogger(__name__)


class WeaviateClient:
    """Wrapper around Weaviate client with our domain-specific operations."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Weaviate client."""
        if not api_key:
            raise ValueError(
                "OpenAI API key is required for text2vec-openai vectorizer"
            )

        self.api_key = api_key

        # Configure connection with OpenAI API key
        self.client = weaviate.connect_to_local(headers={"X-OpenAI-Api-Key": api_key})

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the Weaviate client connection."""
        self.client.close()

    def create_collection(self, collection_name: str) -> None:
        """Create a new collection with our standard schema."""
        if self.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return

        # Create collection with text2vec-openai vectorizer
        self.client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            properties=[
                weaviate.classes.config.Property(
                    name="text", data_type=weaviate.classes.config.DataType.TEXT
                ),
                weaviate.classes.config.Property(
                    name="document_id", data_type=weaviate.classes.config.DataType.INT
                ),
                weaviate.classes.config.Property(
                    name="chunk_index", data_type=weaviate.classes.config.DataType.INT
                ),
                weaviate.classes.config.Property(
                    name="chunk_size", data_type=weaviate.classes.config.DataType.INT
                ),
            ],
        )
        logger.info(f"Created collection: {collection_name}")

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        return self.client.collections.exists(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        if self.collection_exists(collection_name):
            self.client.collections.delete(collection_name)
            logger.info(f"Deleted collection: {collection_name}")

    def batch_insert_chunks(self, collection_name: str, chunks: List[Chunk]) -> None:
        """Batch insert chunks into specified collection."""
        if not chunks:
            logger.warning("No chunks to insert")
            return

        collection = self.client.collections.get(collection_name)

        # Prepare objects for batch insert
        objects = []
        for chunk in chunks:
            obj = {
                "text": chunk.text,
                "document_id": chunk.metadata.document_id,
                "chunk_index": chunk.metadata.chunk_index,
                "chunk_size": chunk.metadata.chunk_size,
            }
            objects.append(obj)

        # Batch insert
        with collection.batch.dynamic() as batch:
            for obj in objects:
                batch.add_object(properties=obj)

        logger.info(f"Inserted {len(chunks)} chunks into {collection_name}")

    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of objects in a collection."""
        if not self.collection_exists(collection_name):
            return 0

        collection = self.client.collections.get(collection_name)
        return collection.aggregate.over_all(total_count=True).total_count

    def semantic_search(
        self, collection_name: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on a collection using vector similarity.

        Args:
            collection_name: Name of the collection to search
            query: Query text to search for
            limit: Maximum number of results to return

        Returns:
            List of dictionaries containing properties and metadata from Weaviate
        """
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return []

        collection = self.client.collections.get(collection_name)

        # Perform near_text query (semantic search)
        response = collection.query.near_text(
            query=query, limit=limit, return_metadata=MetadataQuery(distance=True)
        )

        # Convert Weaviate objects to dicts
        results = []
        for obj in response.objects:
            result = {
                "properties": obj.properties,
                "metadata": {
                    "uuid": str(obj.uuid),
                    "distance": obj.metadata.distance if obj.metadata else None,
                },
            }
            results.append(result)

        logger.info(f"Found {len(results)} results for query in {collection_name}")
        return results


@cache
def get_weaviate_client() -> WeaviateClient:
    """
    Get or create a singleton WeaviateClient instance.

    Uses functools.cache to ensure only one client is created per process.
    Loads environment variables and retrieves the OpenAI API key.

    Returns:
        Singleton WeaviateClient instance

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    logger.info("Creating singleton WeaviateClient instance")
    return WeaviateClient(api_key=api_key)
