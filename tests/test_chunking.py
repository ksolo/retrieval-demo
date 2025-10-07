"""Tests for chunking strategies."""

from src.retrieval_demo.dataloader.chunking.recursive import (
    RecursiveTextSplitterStrategy,
)
from src.retrieval_demo.dataloader.chunking.base import Chunk, ChunkMetadata


class TestRecursiveTextSplitterStrategy:
    """Tests for RecursiveTextSplitterStrategy."""

    def test_collection_name_format(self):
        """Test that collection name follows expected format."""
        strategy = RecursiveTextSplitterStrategy(chunk_size=100, chunk_overlap=25)
        assert strategy.get_collection_name() == "chunks_recursive_100_25"

    def test_chunk_document_returns_chunks(self):
        """Test that chunking returns Chunk objects with proper metadata."""
        strategy = RecursiveTextSplitterStrategy(chunk_size=50, chunk_overlap=10)
        document = "This is a test document. " * 10  # Make it long enough to chunk
        document_id = "test_doc_42"

        chunks = strategy.chunk_document(document, document_id)

        # Should return at least one chunk
        assert len(chunks) > 0

        # All chunks should be Chunk objects
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert isinstance(chunk.metadata, ChunkMetadata)
            assert chunk.metadata.document_id == document_id
            assert chunk.metadata.chunk_size == len(chunk.text)
            assert chunk.metadata.chunk_index >= 0
            assert chunk.metadata.category is None  # No category by default

    def test_chunk_indices_are_sequential(self):
        """Test that chunk indices are sequential starting from 0."""
        strategy = RecursiveTextSplitterStrategy(chunk_size=20, chunk_overlap=5)
        document = (
            "This is a test document that should be split into multiple chunks. " * 5
        )

        chunks = strategy.chunk_document(document, "doc_0")

        # Check that indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_empty_document_returns_empty_list(self):
        """Test that empty document returns empty list."""
        strategy = RecursiveTextSplitterStrategy(chunk_size=100, chunk_overlap=25)

        chunks = strategy.chunk_document("", "empty_doc")

        assert chunks == []

    def test_short_document_returns_single_chunk(self):
        """Test that document shorter than chunk size returns single chunk."""
        strategy = RecursiveTextSplitterStrategy(chunk_size=100, chunk_overlap=25)
        document = "Short text"

        chunks = strategy.chunk_document(document, "short_doc_5")

        assert len(chunks) == 1
        assert chunks[0].text == document
        assert chunks[0].metadata.document_id == "short_doc_5"
        assert chunks[0].metadata.chunk_index == 0
        assert chunks[0].metadata.chunk_size == len(document)

    def test_chunk_document_with_category(self):
        """Test that category is properly stored in metadata."""
        strategy = RecursiveTextSplitterStrategy(chunk_size=50, chunk_overlap=10)
        document = "This is a test document. " * 10
        document_id = "test_doc_category"
        category = "Science"

        chunks = strategy.chunk_document(document, document_id, category=category)

        # All chunks should have the category
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.category == category
            assert chunk.metadata.document_id == document_id
