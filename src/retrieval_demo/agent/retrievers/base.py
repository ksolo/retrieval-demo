"""Base types and protocols for retrieval strategies."""

from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any


@dataclass
class Document:
    """
    Document structure mimicking LangChain's Document.

    Attributes:
        page_content: The text content of the document
        metadata: Additional metadata about the document
    """

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Protocol defining the interface for all retriever implementations."""

    def retrieve(self, query: str, limit: int) -> List[Document]:
        """
        Retrieve documents relevant to the query.

        Args:
            query: The search query text
            limit: Maximum number of documents to return

        Returns:
            List of Document objects ranked by relevance
        """
        ...
