"""Unit tests for WeaviateClient singleton pattern and methods."""

from unittest.mock import patch, Mock, MagicMock
import pytest

from src.retrieval_demo.vectorstore.client import get_weaviate_client, WeaviateClient


class TestWeaviateClientSingleton:
    """Tests for get_weaviate_client singleton function."""

    def test_get_weaviate_client_creates_instance(self):
        """Test that get_weaviate_client creates a WeaviateClient instance."""
        # Clear the cache before testing
        get_weaviate_client.cache_clear()

        with patch(
            "src.retrieval_demo.vectorstore.client.WeaviateClient"
        ) as mock_client_class:
            mock_instance = Mock()
            mock_client_class.return_value = mock_instance

            with patch(
                "src.retrieval_demo.vectorstore.client.os.getenv"
            ) as mock_getenv:
                mock_getenv.return_value = "test-api-key"

                client = get_weaviate_client()

                # Verify client was created with API key
                mock_client_class.assert_called_once_with(api_key="test-api-key")
                assert client == mock_instance

    def test_get_weaviate_client_returns_same_instance(self):
        """Test that get_weaviate_client returns the same instance (singleton)."""
        # Clear the cache before testing
        get_weaviate_client.cache_clear()

        with patch(
            "src.retrieval_demo.vectorstore.client.WeaviateClient"
        ) as mock_client_class:
            mock_instance = Mock()
            mock_client_class.return_value = mock_instance

            with patch(
                "src.retrieval_demo.vectorstore.client.os.getenv"
            ) as mock_getenv:
                mock_getenv.return_value = "test-api-key"

                # Call multiple times
                client1 = get_weaviate_client()
                client2 = get_weaviate_client()
                client3 = get_weaviate_client()

                # Should only create once
                assert mock_client_class.call_count == 1

                # All should be the same instance
                assert client1 is client2
                assert client2 is client3

    def test_get_weaviate_client_raises_error_without_api_key(self):
        """Test that get_weaviate_client raises error when API key is missing."""
        # Clear the cache before testing
        get_weaviate_client.cache_clear()

        with patch("src.retrieval_demo.vectorstore.client.os.getenv") as mock_getenv:
            mock_getenv.return_value = None

            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable"):
                get_weaviate_client()

    def test_get_weaviate_client_calls_load_dotenv(self):
        """Test that get_weaviate_client loads environment variables."""
        # Clear the cache before testing
        get_weaviate_client.cache_clear()

        with patch("src.retrieval_demo.vectorstore.client.load_dotenv") as mock_load:
            with patch(
                "src.retrieval_demo.vectorstore.client.WeaviateClient"
            ) as mock_client_class:
                mock_client_class.return_value = Mock()

                with patch(
                    "src.retrieval_demo.vectorstore.client.os.getenv"
                ) as mock_getenv:
                    mock_getenv.return_value = "test-api-key"

                    get_weaviate_client()

                    # Verify load_dotenv was called
                    mock_load.assert_called_once()


class TestWeaviateClientHybridSearch:
    """Tests for WeaviateClient hybrid_search method."""

    @pytest.fixture
    def mock_weaviate_client(self):
        """Create a WeaviateClient with mocked internal client."""
        with patch("src.retrieval_demo.vectorstore.client.weaviate.connect_to_local"):
            client = WeaviateClient(api_key="test-key")
            client.client = Mock()
            return client

    @pytest.fixture
    def mock_collection(self):
        """Create mock collection with hybrid query response."""
        collection = Mock()

        # Mock response objects
        obj1 = MagicMock()
        obj1.properties = {
            "text": "First hybrid result",
            "document_id": 1,
            "chunk_index": 0,
            "chunk_size": 100,
        }
        obj1.uuid = "uuid-1"
        obj1.metadata.score = 0.95

        obj2 = MagicMock()
        obj2.properties = {
            "text": "Second hybrid result",
            "document_id": 2,
            "chunk_index": 0,
            "chunk_size": 120,
        }
        obj2.uuid = "uuid-2"
        obj2.metadata.score = 0.87

        # Mock response
        response = Mock()
        response.objects = [obj1, obj2]
        collection.query.hybrid.return_value = response

        return collection

    def test_hybrid_search_calls_collection_query_hybrid(
        self, mock_weaviate_client, mock_collection
    ):
        """Test that hybrid_search calls collection.query.hybrid with correct parameters."""
        mock_weaviate_client.client.collections.exists.return_value = True
        mock_weaviate_client.client.collections.get.return_value = mock_collection

        mock_weaviate_client.hybrid_search(
            collection_name="test_collection", query="test query", limit=5
        )

        # Verify collection.query.hybrid was called with correct parameters
        from weaviate.classes.query import MetadataQuery

        mock_collection.query.hybrid.assert_called_once_with(
            query="test query",
            limit=5,
            alpha=0.5,
            return_metadata=MetadataQuery(score=True),
        )

    def test_hybrid_search_returns_correct_format(
        self, mock_weaviate_client, mock_collection
    ):
        """Test that hybrid_search returns results in the expected format."""
        mock_weaviate_client.client.collections.exists.return_value = True
        mock_weaviate_client.client.collections.get.return_value = mock_collection

        results = mock_weaviate_client.hybrid_search(
            collection_name="test_collection", query="test query", limit=5
        )

        assert len(results) == 2

        # Check first result structure
        assert results[0]["properties"]["text"] == "First hybrid result"
        assert results[0]["properties"]["document_id"] == 1
        assert results[0]["metadata"]["uuid"] == "uuid-1"
        assert results[0]["metadata"]["hybrid_score"] == 0.95

        # Check second result
        assert results[1]["properties"]["text"] == "Second hybrid result"
        assert results[1]["metadata"]["hybrid_score"] == 0.87

    def test_hybrid_search_raises_error_for_nonexistent_collection(
        self, mock_weaviate_client
    ):
        """Test that hybrid_search raises ValueError for nonexistent collection."""
        mock_weaviate_client.client.collections.exists.return_value = False

        with pytest.raises(
            ValueError, match="Collection test_collection does not exist"
        ):
            mock_weaviate_client.hybrid_search(
                collection_name="test_collection", query="test query", limit=5
            )

    def test_hybrid_search_handles_empty_results(self, mock_weaviate_client):
        """Test that hybrid_search handles empty results gracefully."""
        mock_weaviate_client.client.collections.exists.return_value = True

        collection = Mock()
        response = Mock()
        response.objects = []
        collection.query.hybrid.return_value = response
        mock_weaviate_client.client.collections.get.return_value = collection

        results = mock_weaviate_client.hybrid_search(
            collection_name="test_collection", query="test query", limit=5
        )

        assert results == []

    def test_hybrid_search_handles_none_metadata(self, mock_weaviate_client):
        """Test that hybrid_search handles None metadata gracefully."""
        mock_weaviate_client.client.collections.exists.return_value = True

        collection = Mock()
        obj = MagicMock()
        obj.properties = {"text": "Test", "document_id": 1}
        obj.uuid = "uuid-1"
        obj.metadata = None

        response = Mock()
        response.objects = [obj]
        collection.query.hybrid.return_value = response
        mock_weaviate_client.client.collections.get.return_value = collection

        results = mock_weaviate_client.hybrid_search(
            collection_name="test_collection", query="test query", limit=5
        )

        assert len(results) == 1
        assert results[0]["metadata"]["hybrid_score"] is None
