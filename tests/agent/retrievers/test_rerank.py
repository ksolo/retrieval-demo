"""Unit tests for RerankRetriever."""

from unittest.mock import Mock, MagicMock
import pytest

from src.retrieval_demo.agent.retrievers.rerank import RerankRetriever
from src.retrieval_demo.agent.retrievers.base import Document


class TestRerankRetriever:
    """Unit tests for RerankRetriever using dependency injection."""

    @pytest.fixture
    def mock_weaviate_client(self):
        """Create mock WeaviateClient."""
        return Mock()

    @pytest.fixture
    def mock_cohere_client(self):
        """Create mock Cohere client."""
        return Mock()

    @pytest.fixture
    def retriever(self, mock_weaviate_client, mock_cohere_client):
        """Create RerankRetriever with mocked clients."""
        retriever = RerankRetriever(
            client=mock_weaviate_client,
            collection_name="test_collection",
            cohere_api_key="test-api-key",
        )
        # Replace the real Cohere client with our mock
        retriever.cohere_client = mock_cohere_client
        return retriever

    @pytest.fixture
    def sample_weaviate_results(self):
        """Sample results from Weaviate semantic search."""
        return [
            {
                "properties": {
                    "text": "First document content",
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "chunk_size": 100,
                },
                "metadata": {"uuid": "uuid-1", "distance": 0.15},
            },
            {
                "properties": {
                    "text": "Second document content",
                    "document_id": "doc-1",
                    "chunk_index": 1,
                    "chunk_size": 120,
                },
                "metadata": {"uuid": "uuid-2", "distance": 0.25},
            },
            {
                "properties": {
                    "text": "Third document content",
                    "document_id": "doc-2",
                    "chunk_index": 0,
                    "chunk_size": 110,
                },
                "metadata": {"uuid": "uuid-3", "distance": 0.30},
            },
            {
                "properties": {
                    "text": "Fourth document content",
                    "document_id": "doc-2",
                    "chunk_index": 1,
                    "chunk_size": 105,
                },
                "metadata": {"uuid": "uuid-4", "distance": 0.35},
            },
        ]

    @pytest.fixture
    def mock_rerank_response(self):
        """Mock response from Cohere rerank API."""
        response = MagicMock()
        # Simulate reranking: indices 2, 0 are most relevant
        result1 = MagicMock()
        result1.index = 2
        result1.relevance_score = 0.95

        result2 = MagicMock()
        result2.index = 0
        result2.relevance_score = 0.87

        response.results = [result1, result2]
        return response

    def test_retrieve_gets_2x_candidates_from_semantic_search(
        self, retriever, mock_weaviate_client, mock_cohere_client, sample_weaviate_results
    ):
        """Test that retrieve requests 2x limit from semantic search."""
        mock_weaviate_client.semantic_search.return_value = sample_weaviate_results
        mock_cohere_client.rerank.return_value = MagicMock(results=[])

        retriever.retrieve(query="test query", limit=2)

        # Should request 2 * 2 = 4 documents
        mock_weaviate_client.semantic_search.assert_called_once_with(
            collection_name="test_collection", query="test query", limit=4
        )

    def test_retrieve_calls_cohere_rerank_with_correct_parameters(
        self, retriever, mock_weaviate_client, mock_cohere_client, sample_weaviate_results
    ):
        """Test that retrieve calls Cohere rerank with correct parameters."""
        mock_weaviate_client.semantic_search.return_value = sample_weaviate_results
        mock_cohere_client.rerank.return_value = MagicMock(results=[])

        retriever.retrieve(query="test query", limit=2)

        # Verify Cohere rerank was called with correct parameters
        mock_cohere_client.rerank.assert_called_once_with(
            model="rerank-english-v3.0",
            query="test query",
            documents=[
                "First document content",
                "Second document content",
                "Third document content",
                "Fourth document content",
            ],
            top_n=2,
        )

    def test_retrieve_returns_reranked_documents_in_correct_order(
        self,
        retriever,
        mock_weaviate_client,
        mock_cohere_client,
        sample_weaviate_results,
        mock_rerank_response,
    ):
        """Test that retrieve returns documents in reranked order."""
        mock_weaviate_client.semantic_search.return_value = sample_weaviate_results
        mock_cohere_client.rerank.return_value = mock_rerank_response

        documents = retriever.retrieve(query="test query", limit=2)

        # Should return 2 documents in reranked order
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

        # First document should be index 2 from semantic results (Third document)
        assert documents[0].page_content == "Third document content"
        assert documents[0].metadata["document_id"] == "doc-2"
        assert documents[0].metadata["chunk_index"] == 0
        assert documents[0].metadata["rerank_score"] == 0.95

        # Second document should be index 0 from semantic results (First document)
        assert documents[1].page_content == "First document content"
        assert documents[1].metadata["document_id"] == "doc-1"
        assert documents[1].metadata["chunk_index"] == 0
        assert documents[1].metadata["rerank_score"] == 0.87

    def test_retrieve_preserves_metadata_from_semantic_search(
        self,
        retriever,
        mock_weaviate_client,
        mock_cohere_client,
        sample_weaviate_results,
        mock_rerank_response,
    ):
        """Test that retrieve preserves all metadata from semantic search."""
        mock_weaviate_client.semantic_search.return_value = sample_weaviate_results
        mock_cohere_client.rerank.return_value = mock_rerank_response

        documents = retriever.retrieve(query="test query", limit=2)

        # Verify metadata from semantic search is preserved
        assert documents[0].metadata["uuid"] == "uuid-3"
        assert documents[0].metadata["distance"] == 0.30
        assert documents[0].metadata["chunk_size"] == 110

    def test_retrieve_handles_empty_semantic_results(
        self, retriever, mock_weaviate_client, mock_cohere_client
    ):
        """Test that retrieve handles empty semantic search results gracefully."""
        mock_weaviate_client.semantic_search.return_value = []

        documents = retriever.retrieve(query="test query", limit=5)

        assert documents == []
        # Cohere should not be called if no semantic results
        mock_cohere_client.rerank.assert_not_called()

    def test_retrieve_handles_missing_text_field(
        self, retriever, mock_weaviate_client, mock_cohere_client
    ):
        """Test that retrieve handles missing text field gracefully."""
        results_with_missing_text = [
            {
                "properties": {"document_id": "doc-1", "chunk_index": 0, "chunk_size": 100},
                "metadata": {"uuid": "uuid-1", "distance": 0.15},
            },
            {
                "properties": {
                    "text": "Second document",
                    "document_id": "doc-2",
                    "chunk_index": 0,
                    "chunk_size": 120,
                },
                "metadata": {"uuid": "uuid-2", "distance": 0.25},
            },
        ]

        mock_weaviate_client.semantic_search.return_value = results_with_missing_text

        # Mock rerank to return the first document
        rerank_response = MagicMock()
        result = MagicMock()
        result.index = 0
        result.relevance_score = 0.9
        rerank_response.results = [result]
        mock_cohere_client.rerank.return_value = rerank_response

        documents = retriever.retrieve(query="test", limit=1)

        assert len(documents) == 1
        assert documents[0].page_content == ""  # Empty string for missing text

    def test_cohere_api_error_propagates(
        self, retriever, mock_weaviate_client, mock_cohere_client, sample_weaviate_results
    ):
        """Test that Cohere API errors are propagated (fail fast)."""
        mock_weaviate_client.semantic_search.return_value = sample_weaviate_results
        mock_cohere_client.rerank.side_effect = Exception("Cohere API error")

        with pytest.raises(Exception, match="Cohere API error"):
            retriever.retrieve(query="test query", limit=2)

    def test_retriever_uses_injected_collection_name(
        self, mock_weaviate_client, mock_cohere_client
    ):
        """Test that retriever uses the collection name provided during initialization."""
        retriever = RerankRetriever(
            client=mock_weaviate_client,
            collection_name="custom_collection",
            cohere_api_key="test-key",
        )
        retriever.cohere_client = mock_cohere_client

        mock_weaviate_client.semantic_search.return_value = []

        retriever.retrieve(query="test", limit=3)

        mock_weaviate_client.semantic_search.assert_called_once_with(
            collection_name="custom_collection", query="test", limit=6
        )

    def test_get_semantic_candidates_returns_correct_results(self, retriever, mock_weaviate_client, sample_weaviate_results):
        """Test _get_semantic_candidates private method."""
        mock_weaviate_client.semantic_search.return_value = sample_weaviate_results

        results = retriever._get_semantic_candidates("test query", limit=2)

        assert results == sample_weaviate_results
        mock_weaviate_client.semantic_search.assert_called_once_with(
            collection_name="test_collection", query="test query", limit=4
        )

    def test_rerank_with_cohere_extracts_text_correctly(
        self, retriever, mock_cohere_client, sample_weaviate_results
    ):
        """Test _rerank_with_cohere extracts text from semantic results."""
        mock_cohere_client.rerank.return_value = MagicMock(results=[])

        retriever._rerank_with_cohere("test query", sample_weaviate_results, limit=2)

        call_args = mock_cohere_client.rerank.call_args
        assert call_args[1]["documents"] == [
            "First document content",
            "Second document content",
            "Third document content",
            "Fourth document content",
        ]

    def test_build_documents_creates_correct_document_objects(
        self, retriever, sample_weaviate_results, mock_rerank_response
    ):
        """Test _build_documents creates Document objects correctly."""
        documents = retriever._build_documents(mock_rerank_response, sample_weaviate_results)

        assert len(documents) == 2
        assert documents[0].page_content == "Third document content"
        assert documents[0].metadata["rerank_score"] == 0.95
        assert documents[1].page_content == "First document content"
        assert documents[1].metadata["rerank_score"] == 0.87
