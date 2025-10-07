from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .base import Chunk, ChunkMetadata, ChunkingStrategy


class RecursiveTextSplitterStrategy(ChunkingStrategy):
    """Chunking strategy using LangChain's RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize with chunk size and overlap parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def get_collection_name(self) -> str:
        """Return collection name based on chunk size and overlap."""
        return f"chunks_recursive_{self.chunk_size}_{self.chunk_overlap}"

    def chunk_document(
        self, document: str, document_id: str, category: Optional[str] = None
    ) -> List[Chunk]:
        """Chunk document using RecursiveCharacterTextSplitter."""
        text_chunks = self.splitter.split_text(document)

        chunks = []
        for chunk_index, text in enumerate(text_chunks):
            metadata = ChunkMetadata(
                document_id=document_id,
                chunk_index=chunk_index,
                chunk_size=len(text),
                category=category,
            )
            chunks.append(Chunk(text=text, metadata=metadata))

        return chunks
