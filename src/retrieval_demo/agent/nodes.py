from ...vectorstore.client import get_weaviate_client
from .agent_state import AgentState
from .retrievers import make_retriever


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieval node that creates a retriever and fetches documents.

    Uses a singleton WeaviateClient instance and the factory to create
    the appropriate retriever based on the strategy specified in the state.

    Args:
        state: Agent state containing collection, strategy, query, and topk

    Returns:
        Dictionary with documents key containing retrieved documents
    """
    collection = state["collection"]
    strategy = state["retrieval_strategy"]
    query = state["query"]
    topk = state["topk"]

    # Get singleton client instance
    client = get_weaviate_client()

    # Create retriever using factory with shared client
    retriever = make_retriever(
        client=client, collection_name=collection, strategy=strategy
    )

    # Retrieve documents
    documents = retriever.retrieve(query=query, limit=topk)

    return {"documents": documents}


def model_node(state: AgentState) -> dict:
    return {}
