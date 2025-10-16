"""Hybrid retriever using Weaviate's combined vector and BM25 search."""

import logging
from typing import List

from ...vectorstore.client import WeaviateClient
from .base import Document

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Retriever that uses Weaviate's hybrid search.

    Hybrid search combines vector similarity search with BM25 keyword search
    using a balanced weighting (alpha=0.5) between the two approaches.
    """

    def __init__(self, client: WeaviateClient, collection_name: str):
        """
        Initialize hybrid retriever with injected dependencies.

        Args:
            client: WeaviateClient instance for performing searches
            collection_name: Name of the collection to search in
        """
        self.client = client
        self.collection_name = collection_name

    def retrieve(self, query: str, limit: int) -> List[Document]:
        """
        Retrieve documents using hybrid search (vector + BM25).

        Args:
            query: The search query text
            limit: Maximum number of documents to return

        Returns:
            List of Document objects ranked by hybrid search score

        Raises:
            ValueError: If collection does not exist
        """
        # Delegate search to WeaviateClient
        results = self.client.hybrid_search(
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
            f"using hybrid search for query: {query[:50]}..."
        )

        return documents
