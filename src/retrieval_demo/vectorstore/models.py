"""Data models and schema definitions for Weaviate integration."""

from typing import Dict, Any

# Standard schema for all chunk collections
CHUNK_SCHEMA_TEMPLATE = {
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "type": "text"
        }
    },
    "properties": [
        {
            "name": "text",
            "dataType": ["text"],
            "description": "The chunk content"
        },
        {
            "name": "document_id", 
            "dataType": ["int"],
            "description": "Original dataset document ID"
        },
        {
            "name": "chunk_index",
            "dataType": ["int"], 
            "description": "Position of chunk within document"
        },
        {
            "name": "chunk_size",
            "dataType": ["int"],
            "description": "Character count of this chunk"
        }
    ]
}


def create_collection_schema(collection_name: str) -> Dict[str, Any]:
    """Create a collection schema with the given name."""
    schema = CHUNK_SCHEMA_TEMPLATE.copy()
    schema["class"] = collection_name
    return schema