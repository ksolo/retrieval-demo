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
                metadata=ChunkMetadata(document_id=1, chunk_index=0, chunk_size=30)
            ),
            Chunk(
                text="This is the second test chunk with more content.",
                metadata=ChunkMetadata(document_id=1, chunk_index=1, chunk_size=48)
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