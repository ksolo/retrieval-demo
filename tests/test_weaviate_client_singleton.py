"""Unit tests for WeaviateClient singleton pattern."""

from unittest.mock import patch, Mock
import pytest

from src.retrieval_demo.vectorstore.client import get_weaviate_client


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
