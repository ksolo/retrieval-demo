from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    document_id: str  # Changed from int to support eval_id format
    chunk_index: int
    chunk_size: int
    category: Optional[str] = None  # Added for categorized datasets


@dataclass
class Chunk:
    """A text chunk with associated metadata."""
    text: str
    metadata: ChunkMetadata


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies."""

    @abstractmethod
    def get_collection_name(self) -> str:
        """Return the collection name for this chunking strategy."""

    @abstractmethod
    def chunk_document(
        self,
        document: str,
        document_id: str,  # Changed from int
        category: Optional[str] = None  # Added for categorized datasets
    ) -> List[Chunk]:
        """Chunk a document and return list of chunks with metadata."""