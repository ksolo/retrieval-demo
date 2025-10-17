"""Unit tests for retriever factory."""

from unittest.mock import Mock, patch
import pytest

from src.retrieval_demo.agent.retrievers.factory import make_retriever
from src.retrieval_demo.agent.retrievers.semantic import SemanticRetriever
from src.retrieval_demo.agent.retrievers.rerank import RerankRetriever
from src.retrieval_demo.agent.retrievers.hybrid import HybridRetriever
from src.retrieval_demo.agent.retrievers.multiquery import MultiQueryRetriever


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

    @patch("src.retrieval_demo.agent.retrievers.factory.os.getenv")
    def test_make_retriever_creates_rerank_retriever(self, mock_getenv, mock_client):
        """Test that factory creates RerankRetriever for rerank strategy."""
        mock_getenv.return_value = "test-cohere-api-key"

        retriever = make_retriever(
            client=mock_client,
            collection_name="test_collection",
            strategy="rerank",
        )

        # Verify correct retriever type
        assert isinstance(retriever, RerankRetriever)
        assert retriever.client == mock_client
        assert retriever.collection_name == "test_collection"

        # Verify environment variable was checked
        mock_getenv.assert_called_once_with("COHERE_API_KEY")

    @patch("src.retrieval_demo.agent.retrievers.factory.os.getenv")
    def test_make_retriever_rerank_raises_error_without_api_key(
        self, mock_getenv, mock_client
    ):
        """Test that rerank strategy raises ValueError when COHERE_API_KEY is missing."""
        mock_getenv.return_value = None

        with pytest.raises(
            ValueError,
            match="COHERE_API_KEY environment variable is required for rerank strategy",
        ):
            make_retriever(
                client=mock_client,
                collection_name="test_collection",
                strategy="rerank",
            )

    @patch("src.retrieval_demo.agent.retrievers.factory.os.getenv")
    def test_make_retriever_creates_multiquery_retriever(self, mock_getenv, mock_client):
        """Test that factory creates MultiQueryRetriever for multiquery strategy."""
        mock_getenv.return_value = "test-openai-api-key"

        retriever = make_retriever(
            client=mock_client,
            collection_name="test_collection",
            strategy="multiquery",
        )

        # Verify correct retriever type
        assert isinstance(retriever, MultiQueryRetriever)
        assert retriever.client == mock_client
        assert retriever.collection_name == "test_collection"

        # Verify environment variable was checked (may be called multiple times by ChatOpenAI init)
        assert any(call[0][0] == "OPENAI_API_KEY" for call in mock_getenv.call_args_list)

    @patch("src.retrieval_demo.agent.retrievers.factory.os.getenv")
    def test_make_retriever_multiquery_raises_error_without_api_key(
        self, mock_getenv, mock_client
    ):
        """Test that multiquery strategy raises ValueError when OPENAI_API_KEY is missing."""
        mock_getenv.return_value = None

        with pytest.raises(
            ValueError,
            match="OPENAI_API_KEY environment variable is required for multiquery strategy",
        ):
            make_retriever(
                client=mock_client,
                collection_name="test_collection",
                strategy="multiquery",
            )

    def test_make_retriever_creates_hybrid_retriever(self, mock_client):
        """Test that factory creates HybridRetriever for hybrid strategy."""
        retriever = make_retriever(
            client=mock_client,
            collection_name="test_collection",
            strategy="hybrid",
        )

        # Verify correct retriever type
        assert isinstance(retriever, HybridRetriever)
        assert retriever.client == mock_client
        assert retriever.collection_name == "test_collection"

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
