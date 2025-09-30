"""Unit tests for retriever factory."""

from unittest.mock import Mock
import pytest

from src.retrieval_demo.agent.retrievers.factory import make_retriever
from src.retrieval_demo.agent.retrievers.semantic import SemanticRetriever


class TestRetrieverFactory:
    """Unit tests for make_retriever factory function."""

    @pytest.fixture
    def mock_client(self):
        """Create mock WeaviateClient instance."""
        return Mock()

    def test_make_retriever_creates_semantic_retriever(self, mock_client):
        """Test that factory creates SemanticRetriever for semantic strategy."""
        retriever = make_retriever(
            client=mock_client,
            collection_name="test_collection",
            strategy="semantic",
        )

        # Verify correct retriever type
        assert isinstance(retriever, SemanticRetriever)
        assert retriever.client == mock_client
        assert retriever.collection_name == "test_collection"

    def test_make_retriever_rerank_not_implemented(self, mock_client):
        """Test that rerank strategy raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Rerank retriever not yet implemented"
        ):
            make_retriever(
                client=mock_client,
                collection_name="test_collection",
                strategy="rerank",
            )

    def test_make_retriever_multiquery_not_implemented(self, mock_client):
        """Test that multiquery strategy raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="MultiQuery retriever not yet implemented"
        ):
            make_retriever(
                client=mock_client,
                collection_name="test_collection",
                strategy="multiquery",
            )

    def test_make_retriever_hybrid_not_implemented(self, mock_client):
        """Test that hybrid strategy raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Hybrid retriever not yet implemented"
        ):
            make_retriever(
                client=mock_client,
                collection_name="test_collection",
                strategy="hybrid",
            )

    def test_make_retriever_invalid_strategy_raises_error(self, mock_client):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            make_retriever(
                client=mock_client,
                collection_name="test_collection",
                strategy="invalid_strategy",  # type: ignore
            )

    def test_make_retriever_uses_shared_client(self, mock_client):
        """Test that factory uses the same shared client for multiple retrievers."""
        retriever1 = make_retriever(
            client=mock_client, collection_name="collection1", strategy="semantic"
        )

        retriever2 = make_retriever(
            client=mock_client, collection_name="collection2", strategy="semantic"
        )

        # Both retrievers should use the same client instance
        assert retriever1.client is mock_client
        assert retriever2.client is mock_client
        assert retriever1.client is retriever2.client

    def test_make_retriever_passes_collection_name_correctly(self, mock_client):
        """Test that factory passes collection name to retriever correctly."""
        retriever = make_retriever(
            client=mock_client,
            collection_name="my_custom_collection",
            strategy="semantic",
        )

        assert retriever.collection_name == "my_custom_collection"
