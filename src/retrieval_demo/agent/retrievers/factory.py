"""Factory for creating retriever instances based on strategy."""

import logging
from typing import Literal

from ...vectorstore.client import WeaviateClient
from .base import Retriever
from .semantic import SemanticRetriever

logger = logging.getLogger(__name__)

RetrieverStrategy = Literal["semantic", "rerank", "multiquery", "hybrid"]


def make_retriever(
    client: WeaviateClient, collection_name: str, strategy: RetrieverStrategy
) -> Retriever:
    """
    Factory function to create retriever instances based on strategy.

    This function follows the Factory pattern to encapsulate retriever creation
    logic and dependency injection. Uses a shared WeaviateClient instance to
    avoid creating multiple connections.

    Args:
        client: Shared WeaviateClient instance for database connection
        collection_name: Name of the Weaviate collection to search
        strategy: The retrieval strategy to use

    Returns:
        A retriever instance implementing the Retriever protocol

    Raises:
        ValueError: If strategy is not supported
    """

    if strategy == "semantic":
        logger.info(f"Creating SemanticRetriever for collection: {collection_name}")
        return SemanticRetriever(client=client, collection_name=collection_name)

    elif strategy == "rerank":
        # TODO: Implement RerankRetriever
        raise NotImplementedError("Rerank retriever not yet implemented")

    elif strategy == "multiquery":
        # TODO: Implement MultiQueryRetriever
        raise NotImplementedError("MultiQuery retriever not yet implemented")

    elif strategy == "hybrid":
        # TODO: Implement HybridRetriever
        raise NotImplementedError("Hybrid retriever not yet implemented")

    else:
        raise ValueError(
            f"Unknown retrieval strategy: {strategy}. "
            f"Supported strategies: semantic, rerank, multiquery, hybrid"
        )
