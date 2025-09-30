"""Retrieval strategies for document search."""

from .base import Document, Retriever
from .factory import make_retriever
from .semantic import SemanticRetriever

__all__ = [
    "Document",
    "Retriever",
    "make_retriever",
    "SemanticRetriever",
]
