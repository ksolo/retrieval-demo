"""Unit tests for HybridRetriever."""

from unittest.mock import Mock
import pytest

from src.retrieval_demo.agent.retrievers.hybrid import HybridRetriever
from src.retrieval_demo.agent.retrievers.base import Document


class TestHybridRetriever:
    """Unit tests for HybridRetriever using dependency injection."""

    @pytest.fixture
    def mock_client(self):
        """Create mock WeaviateClient."""
        return Mock()

    @pytest.fixture
    def retriever(self, mock_client):
        """Create HybridRetriever with mocked client."""
        return HybridRetriever(client=mock_client, collection_name="test_collection")

    @pytest.fixture
    def sample_weaviate_results(self):
        """Sample results from Weaviate hybrid search."""
        return [
            {
                "properties": {
                    "text": "First document content",
                    "document_id": 1,
                    "chunk_index": 0,
                    "chunk_size": 100,
                },
                "metadata": {"uuid": "uuid-1", "hybrid_score": 0.95},
            },
            {
                "properties": {
                    "text": "Second document content",
                    "document_id": 1,
                    "chunk_index": 1,
                    "chunk_size": 120,
                },
                "metadata": {"uuid": "uuid-2", "hybrid_score": 0.87},
            },
        ]

    def test_retrieve_calls_client_with_correct_parameters(
        self, retriever, mock_client
    ):
        """Test that retrieve delegates to client with correct parameters."""
        mock_client.hybrid_search.return_value = []

        retriever.retrieve(query="test query", limit=5)

        mock_client.hybrid_search.assert_called_once_with(
            collection_name="test_collection", query="test query", limit=5
        )

    def test_retrieve_converts_weaviate_results_to_documents(
        self, retriever, mock_client, sample_weaviate_results
    ):
        """Test that retrieve properly converts Weaviate results to Documents."""
        mock_client.hybrid_search.return_value = sample_weaviate_results

        documents = retriever.retrieve(query="test query", limit=2)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

        # Check first document
        assert documents[0].page_content == "First document content"
        assert documents[0].metadata["document_id"] == 1
        assert documents[0].metadata["chunk_index"] == 0
        assert documents[0].metadata["chunk_size"] == 100
        assert documents[0].metadata["uuid"] == "uuid-1"
        assert documents[0].metadata["hybrid_score"] == 0.95

        # Check second document
        assert documents[1].page_content == "Second document content"
        assert documents[1].metadata["document_id"] == 1
        assert documents[1].metadata["chunk_index"] == 1
        assert documents[1].metadata["hybrid_score"] == 0.87

    def test_retrieve_handles_empty_results(self, retriever, mock_client):
        """Test that retrieve handles empty results gracefully."""
        mock_client.hybrid_search.return_value = []

        documents = retriever.retrieve(query="test query", limit=5)

        assert documents == []

    def test_retrieve_handles_missing_text_field(self, retriever, mock_client):
        """Test that retrieve handles missing text field gracefully."""
        mock_client.hybrid_search.return_value = [
            {
                "properties": {"document_id": 1, "chunk_index": 0, "chunk_size": 100},
                "metadata": {"uuid": "uuid-1", "hybrid_score": 0.85},
            }
        ]

        documents = retriever.retrieve(query="test query", limit=1)

        assert len(documents) == 1
        assert documents[0].page_content == ""

    def test_retriever_uses_injected_collection_name(self, mock_client):
        """Test that retriever uses the collection name provided during initialization."""
        retriever = HybridRetriever(
            client=mock_client, collection_name="custom_collection"
        )
        mock_client.hybrid_search.return_value = []

        retriever.retrieve(query="test", limit=3)

        mock_client.hybrid_search.assert_called_once_with(
            collection_name="custom_collection", query="test", limit=3
        )

    def test_retriever_preserves_metadata_from_weaviate(self, retriever, mock_client):
        """Test that retriever preserves all metadata from Weaviate results."""
        mock_client.hybrid_search.return_value = [
            {
                "properties": {
                    "text": "Test content",
                    "document_id": 42,
                    "chunk_index": 5,
                    "chunk_size": 200,
                },
                "metadata": {
                    "uuid": "test-uuid",
                    "hybrid_score": 0.92,
                    "custom_field": "custom_value",
                },
            }
        ]

        documents = retriever.retrieve(query="test", limit=1)

        # Verify all metadata is preserved
        assert documents[0].metadata["uuid"] == "test-uuid"
        assert documents[0].metadata["hybrid_score"] == 0.92
        assert documents[0].metadata["custom_field"] == "custom_value"
        assert documents[0].metadata["document_id"] == 42
        assert documents[0].metadata["chunk_index"] == 5
        assert documents[0].metadata["chunk_size"] == 200

    def test_retrieve_propagates_client_errors(self, retriever, mock_client):
        """Test that errors from client are propagated."""
        mock_client.hybrid_search.side_effect = ValueError("Collection does not exist")

        with pytest.raises(ValueError, match="Collection does not exist"):
            retriever.retrieve(query="test query", limit=5)
