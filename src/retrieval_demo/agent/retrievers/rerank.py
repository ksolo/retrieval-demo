"""Rerank retriever using Cohere reranking API."""

import logging
from typing import List, Dict, Any

import cohere

from ...vectorstore.client import WeaviateClient
from .base import Document

logger = logging.getLogger(__name__)


class RerankRetriever:
    """
    Retriever that uses semantic search followed by Cohere reranking.

    This retriever first retrieves 2x the requested limit using semantic search,
    then uses Cohere's rerank API to reorder and select the top results.
    """

    def __init__(
        self, client: WeaviateClient, collection_name: str, cohere_api_key: str
    ):
        """
        Initialize rerank retriever with injected dependencies.

        Args:
            client: WeaviateClient instance for performing searches
            collection_name: Name of the collection to search in
            cohere_api_key: API key for Cohere reranking service
        """
        self.client = client
        self.collection_name = collection_name
        self.cohere_client = cohere.Client(api_key=cohere_api_key)

    def retrieve(self, query: str, limit: int) -> List[Document]:
        """
        Retrieve documents using semantic search followed by Cohere reranking.

        This method coordinates the retrieval pipeline:
        1. Get initial candidates via semantic search (2x limit)
        2. Rerank candidates using Cohere
        3. Convert reranked results to Document objects

        Args:
            query: The search query text
            limit: Maximum number of documents to return

        Returns:
            List of Document objects ranked by Cohere reranking scores

        Raises:
            Exception: If Cohere API call fails
        """
        # Step 1: Get semantic search candidates
        semantic_results = self._get_semantic_candidates(query, limit)

        if not semantic_results:
            logger.warning("No documents found for semantic search, returning empty list")
            return []

        # Step 2: Rerank candidates with Cohere
        rerank_response = self._rerank_with_cohere(query, semantic_results, limit)

        # Step 3: Build Document objects from reranked results
        documents = self._build_documents(rerank_response, semantic_results)

        logger.info(
            f"Returning {len(documents)} reranked documents for query: {query[:50]}..."
        )

        return documents

    def _get_semantic_candidates(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Retrieve initial candidate documents using semantic search.

        Args:
            query: The search query text
            limit: Number of final results needed (will retrieve 2x this amount)

        Returns:
            List of Weaviate search results (dicts with properties and metadata)
        """
        initial_limit = 2 * limit
        semantic_results = self.client.semantic_search(
            collection_name=self.collection_name, query=query, limit=initial_limit
        )

        logger.info(
            f"Retrieved {len(semantic_results)} candidates from {self.collection_name} "
            f"for reranking (requested {initial_limit})"
        )

        return semantic_results

    def _rerank_with_cohere(
        self, query: str, semantic_results: List[Dict[str, Any]], limit: int
    ) -> cohere.RerankResponse:
        """
        Rerank semantic search results using Cohere API.

        Args:
            query: The search query text
            semantic_results: Results from semantic search
            limit: Number of top results to return from reranking

        Returns:
            Cohere rerank response object

        Raises:
            Exception: If Cohere API call fails
        """
        # Extract text content from semantic results
        doc_texts = [result["properties"].get("text", "") for result in semantic_results]

        logger.info(
            f"Reranking {len(doc_texts)} documents using Cohere rerank-english-v3.0"
        )

        rerank_response = self.cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=doc_texts,
            top_n=limit,
        )

        logger.info(f"Reranking complete, received {len(rerank_response.results)} results")

        return rerank_response

    def _build_documents(
        self, rerank_response: cohere.RerankResponse, semantic_results: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Convert reranked results to Document objects.

        Args:
            rerank_response: Response from Cohere rerank API
            semantic_results: Original semantic search results

        Returns:
            List of Document objects with rerank scores in metadata
        """
        documents = []

        for result in rerank_response.results:
            # Get the original Weaviate result using the index
            original_result = semantic_results[result.index]
            properties = original_result["properties"]
            metadata = original_result["metadata"].copy()

            # Add properties and rerank score to metadata
            metadata.update(
                {
                    "document_id": properties.get("document_id"),
                    "chunk_index": properties.get("chunk_index"),
                    "chunk_size": properties.get("chunk_size"),
                    "rerank_score": result.relevance_score,
                }
            )

            doc = Document(page_content=properties.get("text", ""), metadata=metadata)
            documents.append(doc)

        return documents
