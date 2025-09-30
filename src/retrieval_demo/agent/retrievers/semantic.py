"""Semantic retriever using vector similarity search."""

import logging
from typing import List

from ...vectorstore.client import WeaviateClient
from .base import Document

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Retriever that uses semantic vector similarity search.

    This retriever delegates to WeaviateClient's semantic_search method
    to find documents based on vector similarity.
    """

    def __init__(self, client: WeaviateClient, collection_name: str):
        """
        Initialize semantic retriever with injected dependencies.

        Args:
            client: WeaviateClient instance for performing searches
            collection_name: Name of the collection to search in
        """
        self.client = client
        self.collection_name = collection_name

    def retrieve(self, query: str, limit: int) -> List[Document]:
        """
        Retrieve documents using semantic vector similarity search.

        Args:
            query: The search query text
            limit: Maximum number of documents to return

        Returns:
            List of Document objects ranked by semantic similarity
        """
        # Delegate search to WeaviateClient
        results = self.client.semantic_search(
            collection_name=self.collection_name, query=query, limit=limit
        )

        # Convert Weaviate results to Document objects
        documents = []
        for result in results:
            properties = result["properties"]
            metadata = result["metadata"].copy()

            # Add properties as metadata
            metadata.update(
                {
                    "document_id": properties.get("document_id"),
                    "chunk_index": properties.get("chunk_index"),
                    "chunk_size": properties.get("chunk_size"),
                }
            )

            doc = Document(page_content=properties.get("text", ""), metadata=metadata)
            documents.append(doc)

        logger.info(
            f"Retrieved {len(documents)} documents from {self.collection_name} "
            f"for query: {query[:50]}..."
        )

        return documents
