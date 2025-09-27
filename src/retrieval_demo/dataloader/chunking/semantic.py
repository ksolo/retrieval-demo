from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from .base import Chunk, ChunkMetadata, ChunkingStrategy


class SemanticChunkerStrategy(ChunkingStrategy):
    """Chunking strategy using LangChain's SemanticChunker with OpenAI embeddings."""
    
    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key."""
        self.embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile"
        )
    
    def get_collection_name(self) -> str:
        """Return collection name for semantic chunking strategy."""
        return "chunks_semantic_percentile"
    
    def chunk_document(self, document: str, document_id: int) -> List[Chunk]:
        """Chunk document using SemanticChunker."""
        text_chunks = self.splitter.split_text(document)
        
        chunks = []
        for chunk_index, text in enumerate(text_chunks):
            metadata = ChunkMetadata(
                document_id=document_id,
                chunk_index=chunk_index,
                chunk_size=len(text)
            )
            chunks.append(Chunk(text=text, metadata=metadata))
        
        return chunks