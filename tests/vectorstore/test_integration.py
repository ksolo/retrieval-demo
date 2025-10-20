"""Integration tests for Weaviate vectorstore client."""

import pytest
import os
from dotenv import load_dotenv
from src.retrieval_demo.vectorstore.client import WeaviateClient
from src.retrieval_demo.dataloader.chunking.base import Chunk, ChunkMetadata

# Load environment variables
load_dotenv()


@pytest.mark.integration
class TestWeaviateClientIntegration:
    """Integration tests for WeaviateClient requiring running Weaviate instance."""

    @pytest.fixture
    def api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
        return api_key

    @pytest.fixture
    def client(self, api_key):
        """Create WeaviateClient for testing."""
        return WeaviateClient(api_key=api_key)

    @pytest.fixture
    def test_collection_name(self):
        """Test collection name."""
        return "test_collection_chunks"

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                text="This is the first test chunk.",
                metadata=ChunkMetadata(document_id=1, chunk_index=0, chunk_size=30),
            ),
            Chunk(
                text="This is the second test chunk with more content.",
                metadata=ChunkMetadata(document_id=1, chunk_index=1, chunk_size=48),
            ),
        ]

    def test_client_initialization_requires_api_key(self):
        """Test that client requires API key."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            WeaviateClient(api_key=None)

    def test_create_and_delete_collection(self, client, test_collection_name):
        """Test creating and deleting a collection."""
        try:
            # Initially should not exist
            assert not client.collection_exists(test_collection_name)

            # Create collection
            client.create_collection(test_collection_name)
            assert client.collection_exists(test_collection_name)

            # Creating again should not error
            client.create_collection(test_collection_name)
            assert client.collection_exists(test_collection_name)

        finally:
            # Cleanup
            client.delete_collection(test_collection_name)
            assert not client.collection_exists(test_collection_name)
            client.close()

    def test_batch_insert_and_count(self, client, test_collection_name, sample_chunks):
        """Test batch inserting chunks and counting."""
        try:
            # Create collection
            client.create_collection(test_collection_name)

            # Initially empty
            assert client.get_collection_count(test_collection_name) == 0

            # Insert chunks
            client.batch_insert_chunks(test_collection_name, sample_chunks)

            # Should have 2 chunks
            assert client.get_collection_count(test_collection_name) == 2

        finally:
            # Cleanup
            client.delete_collection(test_collection_name)
            client.close()

    def test_batch_insert_empty_list(self, client, test_collection_name):
        """Test that inserting empty list doesn't error."""
        try:
            client.create_collection(test_collection_name)
            client.batch_insert_chunks(test_collection_name, [])
            assert client.get_collection_count(test_collection_name) == 0

        finally:
            client.delete_collection(test_collection_name)
            client.close()

    def test_get_count_nonexistent_collection(self, client):
        """Test getting count of non-existent collection returns 0."""
        try:
            count = client.get_collection_count("nonexistent_collection")
            assert count == 0
        finally:
            client.close()

    def test_semantic_search_returns_relevant_results(
        self, client, test_collection_name, sample_chunks
    ):
        """Test semantic search returns relevant results."""
        try:
            # Setup: create collection and insert chunks
            client.create_collection(test_collection_name)
            client.batch_insert_chunks(test_collection_name, sample_chunks)

            # Perform semantic search
            results = client.semantic_search(
                collection_name=test_collection_name, query="first test chunk", limit=2
            )

            # Assertions
            assert len(results) == 2
            assert all("properties" in r for r in results)
            assert all("metadata" in r for r in results)

            # First result should have text field
            assert "text" in results[0]["properties"]
            assert "document_id" in results[0]["properties"]

            # Metadata should have distance and uuid
            assert "distance" in results[0]["metadata"]
            assert "uuid" in results[0]["metadata"]

        finally:
            client.delete_collection(test_collection_name)
            client.close()

    def test_semantic_search_respects_limit(self, client, test_collection_name):
        """Test semantic search respects limit parameter."""
        try:
            # Setup: create collection with multiple chunks
            client.create_collection(test_collection_name)
            chunks = [
                Chunk(
                    text=f"Test chunk number {i} with some content.",
                    metadata=ChunkMetadata(document_id=1, chunk_index=i, chunk_size=30),
                )
                for i in range(5)
            ]
            client.batch_insert_chunks(test_collection_name, chunks)

            # Search with limit
            results = client.semantic_search(
                collection_name=test_collection_name, query="test chunk", limit=3
            )

            assert len(results) == 3

        finally:
            client.delete_collection(test_collection_name)
            client.close()

    def test_semantic_search_nonexistent_collection(self, client):
        """Test semantic search on non-existent collection returns empty list."""
        try:
            results = client.semantic_search(
                collection_name="nonexistent_collection", query="test query", limit=5
            )
            assert results == []
        finally:
            client.close()

    def test_semantic_search_empty_collection(self, client, test_collection_name):
        """Test semantic search on empty collection returns empty list."""
        try:
            client.create_collection(test_collection_name)

            results = client.semantic_search(
                collection_name=test_collection_name, query="test query", limit=5
            )

            assert results == []

        finally:
            client.delete_collection(test_collection_name)
            client.close()

    def test_hybrid_search_returns_relevant_results(
        self, client, test_collection_name, sample_chunks
    ):
        """Test hybrid search returns relevant results."""
        try:
            # Setup: create collection and insert chunks
            client.create_collection(test_collection_name)
            client.batch_insert_chunks(test_collection_name, sample_chunks)

            # Perform hybrid search
            results = client.hybrid_search(
                collection_name=test_collection_name, query="first test chunk", limit=2
            )

            # Assertions
            assert len(results) == 2
            assert all("properties" in r for r in results)
            assert all("metadata" in r for r in results)

            # First result should have text field
            assert "text" in results[0]["properties"]
            assert "document_id" in results[0]["properties"]

            # Metadata should have hybrid_score and uuid
            assert "hybrid_score" in results[0]["metadata"]
            assert "uuid" in results[0]["metadata"]

        finally:
            client.delete_collection(test_collection_name)
            client.close()

    def test_hybrid_search_respects_limit(self, client, test_collection_name):
        """Test hybrid search respects limit parameter."""
        try:
            # Setup: create collection with multiple chunks
            client.create_collection(test_collection_name)
            chunks = [
                Chunk(
                    text=f"Test chunk number {i} with some content.",
                    metadata=ChunkMetadata(document_id=1, chunk_index=i, chunk_size=30),
                )
                for i in range(5)
            ]
            client.batch_insert_chunks(test_collection_name, chunks)

            # Search with limit
            results = client.hybrid_search(
                collection_name=test_collection_name, query="test chunk", limit=3
            )

            assert len(results) == 3

        finally:
            client.delete_collection(test_collection_name)
            client.close()

    def test_hybrid_search_nonexistent_collection(self, client):
        """Test hybrid search on non-existent collection raises ValueError."""
        try:
            with pytest.raises(
                ValueError, match="Collection nonexistent_collection does not exist"
            ):
                client.hybrid_search(
                    collection_name="nonexistent_collection",
                    query="test query",
                    limit=5,
                )
        finally:
            client.close()

    def test_hybrid_search_empty_collection(self, client, test_collection_name):
        """Test hybrid search on empty collection returns empty list."""
        try:
            client.create_collection(test_collection_name)

            results = client.hybrid_search(
                collection_name=test_collection_name, query="test query", limit=5
            )

            assert results == []

        finally:
            client.delete_collection(test_collection_name)
            client.close()
