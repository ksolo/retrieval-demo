from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from retrieval_demo.vectorstore.client import get_weaviate_client
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
    """
    Model node that generates an answer based on retrieved documents.

    Args:
        state: Agent state containing query and retrieved documents

    Returns:
        Dictionary with messages key containing the full conversation
    """
    query = state["query"]
    documents = state["documents"]

    # Format context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in documents])

    # Create system and user messages
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use only the information from the context to answer the question. "
            "If the context doesn't contain enough information, say so."
        )
    )

    user_message = HumanMessage(
        content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    # Initialize LLM (uses OPENAI_API_KEY from env)
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0)

    # Generate response
    response = llm.invoke([system_message, user_message])

    # Return all messages for LangSmith tracing
    return {"messages": [system_message, user_message, response]}
