"""Unit tests for MultiQueryRetriever."""

from unittest.mock import Mock, MagicMock, patch
import pytest

from src.retrieval_demo.agent.retrievers.multiquery import MultiQueryRetriever
from src.retrieval_demo.agent.retrievers.base import Document


class TestMultiQueryRetriever:
    """Unit tests for MultiQueryRetriever using dependency injection."""

    @pytest.fixture
    def mock_weaviate_client(self):
        """Create mock WeaviateClient."""
        return Mock()

    @pytest.fixture
    def mock_llm(self):
        """Create mock ChatOpenAI LLM."""
        return Mock()

    @pytest.fixture
    def retriever(self, mock_weaviate_client, mock_llm):
        """Create MultiQueryRetriever with mocked clients."""
        retriever = MultiQueryRetriever(
            client=mock_weaviate_client,
            collection_name="test_collection",
            openai_api_key="test-api-key",
        )
        # Replace the real LLM with our mock
        retriever.llm = mock_llm
        return retriever

    @pytest.fixture
    def sample_weaviate_results_query1(self):
        """Sample results from Weaviate semantic search for query 1."""
        return [
            {
                "properties": {
                    "text": "Document A content",
                    "document_id": "doc-a",
                    "chunk_index": 0,
                    "chunk_size": 100,
                },
                "metadata": {"uuid": "uuid-a", "distance": 0.10},
            },
            {
                "properties": {
                    "text": "Document B content",
                    "document_id": "doc-b",
                    "chunk_index": 0,
                    "chunk_size": 120,
                },
                "metadata": {"uuid": "uuid-b", "distance": 0.20},
            },
            {
                "properties": {
                    "text": "Document C content",
                    "document_id": "doc-c",
                    "chunk_index": 0,
                    "chunk_size": 110,
                },
                "metadata": {"uuid": "uuid-c", "distance": 0.30},
            },
        ]

    @pytest.fixture
    def sample_weaviate_results_query2(self):
        """Sample results from Weaviate semantic search for query 2."""
        return [
            {
                "properties": {
                    "text": "Document B content",
                    "document_id": "doc-b",
                    "chunk_index": 0,
                    "chunk_size": 120,
                },
                "metadata": {"uuid": "uuid-b", "distance": 0.15},
            },
            {
                "properties": {
                    "text": "Document A content",
                    "document_id": "doc-a",
                    "chunk_index": 0,
                    "chunk_size": 100,
                },
                "metadata": {"uuid": "uuid-a", "distance": 0.25},
            },
            {
                "properties": {
                    "text": "Document D content",
                    "document_id": "doc-d",
                    "chunk_index": 0,
                    "chunk_size": 105,
                },
                "metadata": {"uuid": "uuid-d", "distance": 0.35},
            },
        ]

    @pytest.fixture
    def mock_llm_response(self):
        """Mock response from LLM query generation."""
        response = MagicMock()
        response.content = (
            "Alternative query 1\nAlternative query 2\nAlternative query 3"
        )
        return response

    def test_generate_query_variants_calls_llm_correctly(
        self, retriever, mock_llm, mock_llm_response
    ):
        """Test that _generate_query_variants calls LLM with correct prompt."""
        mock_llm.invoke.return_value = mock_llm_response

        variants = retriever._generate_query_variants("original query")

        # Verify LLM was called
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 1
        assert "original query" in call_args[0].content
        assert "3 alternative" in call_args[0].content

        # Verify variants were parsed correctly
        assert len(variants) == 3
        assert variants == [
            "Alternative query 1",
            "Alternative query 2",
            "Alternative query 3",
        ]

    def test_generate_query_variants_handles_numbered_responses(
        self, retriever, mock_llm
    ):
        """Test that variant generation strips numbering from LLM responses."""
        response = MagicMock()
        response.content = "1. First variant\n2. Second variant\n3. Third variant"
        mock_llm.invoke.return_value = response

        variants = retriever._generate_query_variants("test query")

        # Should strip numbering
        assert len(variants) == 3
        assert variants[0] == "First variant"
        assert variants[1] == "Second variant"
        assert variants[2] == "Third variant"

    def test_generate_query_variants_handles_fewer_than_expected(
        self, retriever, mock_llm
    ):
        """Test that variant generation handles fewer variants than requested."""
        response = MagicMock()
        response.content = "Only one variant"
        mock_llm.invoke.return_value = response

        variants = retriever._generate_query_variants("test query")

        # Should return whatever was generated
        assert len(variants) == 1
        assert variants[0] == "Only one variant"

    def test_retrieve_for_queries_calls_semantic_search_with_2x_limit(
        self, retriever, mock_weaviate_client
    ):
        """Test that _retrieve_for_queries requests 2x limit per query."""
        mock_weaviate_client.semantic_search.return_value = []

        queries = ["query 1", "query 2", "query 3"]
        retriever._retrieve_for_queries(queries, limit=5)

        # Should call semantic_search 3 times with limit=10 (2 * 5)
        assert mock_weaviate_client.semantic_search.call_count == 3
        for call in mock_weaviate_client.semantic_search.call_args_list:
            assert call[1]["limit"] == 10
            assert call[1]["collection_name"] == "test_collection"

    def test_retrieve_for_queries_returns_results_by_query(
        self, retriever, mock_weaviate_client, sample_weaviate_results_query1
    ):
        """Test that _retrieve_for_queries returns dict mapping queries to results."""
        mock_weaviate_client.semantic_search.return_value = (
            sample_weaviate_results_query1
        )

        queries = ["query 1", "query 2"]
        results = retriever._retrieve_for_queries(queries, limit=5)

        # Should return dict with one entry per query
        assert len(results) == 2
        assert "query 1" in results
        assert "query 2" in results
        assert results["query 1"] == sample_weaviate_results_query1

    def test_fuse_results_rrf_combines_scores_correctly(self, retriever):
        """Test that RRF fusion combines scores correctly across queries."""
        query_results = {
            "query1": [
                {
                    "properties": {"text": "Doc A", "document_id": "doc-a"},
                    "metadata": {"uuid": "uuid-a"},
                },
                {
                    "properties": {"text": "Doc B", "document_id": "doc-b"},
                    "metadata": {"uuid": "uuid-b"},
                },
            ],
            "query2": [
                {
                    "properties": {"text": "Doc B", "document_id": "doc-b"},
                    "metadata": {"uuid": "uuid-b"},
                },
                {
                    "properties": {"text": "Doc A", "document_id": "doc-a"},
                    "metadata": {"uuid": "uuid-a"},
                },
            ],
        }

        fused = retriever._fuse_results_rrf(query_results, limit=2)

        # Both documents appear in both queries, so they should have equal RRF scores
        # Doc A: rank 0 in query1 (1/60) + rank 1 in query2 (1/61)
        # Doc B: rank 1 in query1 (1/61) + rank 0 in query2 (1/60)
        assert len(fused) == 2
        assert "rrf_score" in fused[0]["metadata"]
        assert "rrf_score" in fused[1]["metadata"]

        # Verify scores are close (they should be nearly equal)
        score_a = next(
            r["metadata"]["rrf_score"]
            for r in fused
            if r["metadata"]["uuid"] == "uuid-a"
        )
        score_b = next(
            r["metadata"]["rrf_score"]
            for r in fused
            if r["metadata"]["uuid"] == "uuid-b"
        )
        assert abs(score_a - score_b) < 0.001  # Should be very close

    def test_fuse_results_rrf_ranks_by_score(self, retriever):
        """Test that RRF fusion ranks documents by combined score."""
        query_results = {
            "query1": [
                {
                    "properties": {"text": "Doc A", "document_id": "doc-a"},
                    "metadata": {"uuid": "uuid-a"},
                },
                {
                    "properties": {"text": "Doc B", "document_id": "doc-b"},
                    "metadata": {"uuid": "uuid-b"},
                },
                {
                    "properties": {"text": "Doc C", "document_id": "doc-c"},
                    "metadata": {"uuid": "uuid-c"},
                },
            ],
            "query2": [
                {
                    "properties": {"text": "Doc A", "document_id": "doc-a"},
                    "metadata": {"uuid": "uuid-a"},
                },
                {
                    "properties": {"text": "Doc C", "document_id": "doc-c"},
                    "metadata": {"uuid": "uuid-c"},
                },
            ],
        }

        fused = retriever._fuse_results_rrf(query_results, limit=3)

        # Doc A appears at rank 0 in both queries -> highest score
        # Doc C appears at rank 2 in query1, rank 1 in query2
        # Doc B appears at rank 1 in query1 only
        assert len(fused) == 3
        assert fused[0]["metadata"]["uuid"] == "uuid-a"  # Highest score

    def test_fuse_results_rrf_respects_limit(self, retriever):
        """Test that RRF fusion respects the limit parameter."""
        query_results = {
            "query1": [
                {"properties": {"text": f"Doc {i}"}, "metadata": {"uuid": f"uuid-{i}"}}
                for i in range(10)
            ],
        }

        fused = retriever._fuse_results_rrf(query_results, limit=3)

        assert len(fused) == 3

    def test_build_documents_converts_to_document_objects(self, retriever):
        """Test that _build_documents creates proper Document objects."""
        results = [
            {
                "properties": {
                    "text": "Test content",
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "chunk_size": 100,
                },
                "metadata": {"uuid": "uuid-1", "rrf_score": 0.5},
            }
        ]

        documents = retriever._build_documents(results)

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert documents[0].page_content == "Test content"
        assert documents[0].metadata["document_id"] == "doc-1"
        assert documents[0].metadata["chunk_index"] == 0
        assert documents[0].metadata["rrf_score"] == 0.5

    def test_build_documents_preserves_all_metadata(self, retriever):
        """Test that _build_documents preserves all metadata fields."""
        results = [
            {
                "properties": {
                    "text": "Test content",
                    "document_id": "doc-1",
                    "chunk_index": 2,
                    "chunk_size": 150,
                },
                "metadata": {
                    "uuid": "uuid-1",
                    "rrf_score": 0.75,
                    "custom_field": "custom_value",
                },
            }
        ]

        documents = retriever._build_documents(results)

        metadata = documents[0].metadata
        assert metadata["uuid"] == "uuid-1"
        assert metadata["rrf_score"] == 0.75
        assert metadata["custom_field"] == "custom_value"
        assert metadata["document_id"] == "doc-1"
        assert metadata["chunk_index"] == 2
        assert metadata["chunk_size"] == 150

    def test_retrieve_orchestrates_full_pipeline(
        self,
        retriever,
        mock_weaviate_client,
        mock_llm,
        mock_llm_response,
        sample_weaviate_results_query1,
    ):
        """Test that retrieve method orchestrates the full multi-query pipeline."""
        # Setup mocks
        mock_llm.invoke.return_value = mock_llm_response
        mock_weaviate_client.semantic_search.return_value = (
            sample_weaviate_results_query1
        )

        # Execute retrieve
        documents = retriever.retrieve(query="test query", limit=2)

        # Verify LLM was called to generate variants
        assert mock_llm.invoke.called

        # Verify semantic search was called 4 times (original + 3 variants)
        assert mock_weaviate_client.semantic_search.call_count == 4

        # Verify results are Document objects
        assert len(documents) <= 2  # Should respect limit
        assert all(isinstance(doc, Document) for doc in documents)

        # Verify documents have RRF scores
        if documents:
            assert "rrf_score" in documents[0].metadata

    def test_retrieve_handles_empty_results(
        self, retriever, mock_weaviate_client, mock_llm, mock_llm_response
    ):
        """Test that retrieve handles empty results from all queries."""
        mock_llm.invoke.return_value = mock_llm_response
        mock_weaviate_client.semantic_search.return_value = []

        documents = retriever.retrieve(query="test query", limit=5)

        # Should return empty list
        assert documents == []

    def test_retrieve_handles_missing_text_field(
        self, retriever, mock_weaviate_client, mock_llm, mock_llm_response
    ):
        """Test that retrieve handles missing text field gracefully."""
        mock_llm.invoke.return_value = mock_llm_response
        results_with_missing_text = [
            {
                "properties": {
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "chunk_size": 100,
                },
                "metadata": {"uuid": "uuid-1", "distance": 0.15},
            }
        ]
        mock_weaviate_client.semantic_search.return_value = results_with_missing_text

        documents = retriever.retrieve(query="test", limit=1)

        # Should handle gracefully with empty string
        assert len(documents) >= 0
        if documents:
            assert documents[0].page_content == ""

    def test_retriever_uses_injected_collection_name(
        self, mock_weaviate_client, mock_llm, mock_llm_response
    ):
        """Test that retriever uses the collection name provided during initialization."""
        retriever = MultiQueryRetriever(
            client=mock_weaviate_client,
            collection_name="custom_collection",
            openai_api_key="test-key",
        )
        retriever.llm = mock_llm

        mock_llm.invoke.return_value = mock_llm_response
        mock_weaviate_client.semantic_search.return_value = []

        retriever.retrieve(query="test", limit=3)

        # Verify all semantic_search calls use custom collection
        for call in mock_weaviate_client.semantic_search.call_args_list:
            assert call[1]["collection_name"] == "custom_collection"

    def test_retriever_uses_correct_rrf_constant(self, retriever):
        """Test that retriever uses k=60 for RRF calculation."""
        assert retriever.rrf_k == 60

    def test_retriever_generates_correct_number_of_variants(self, retriever):
        """Test that retriever is configured to generate 3 variants."""
        assert retriever.num_variants == 3

    @patch("src.retrieval_demo.agent.retrievers.multiquery.ChatOpenAI")
    def test_retriever_initializes_with_correct_llm_model(
        self, mock_chat_openai, mock_weaviate_client
    ):
        """Test that retriever initializes ChatOpenAI with gpt-5-mini."""
        MultiQueryRetriever(
            client=mock_weaviate_client,
            collection_name="test_collection",
            openai_api_key="test-key",
        )

        # Verify ChatOpenAI was initialized with correct parameters
        mock_chat_openai.assert_called_once_with(model="gpt-5-mini", api_key="test-key")
