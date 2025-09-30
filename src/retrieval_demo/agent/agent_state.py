from typing import List, Any
from typing_extensions import TypedDict, Annotated, Literal

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    query: str
    messages: Annotated[List[AnyMessage], add_messages]
    collection: str
    retrieval_strategy: Literal["semantic", "rerank", "multiquery", "hybrid"]
    topk: int
    documents: List[Any]
