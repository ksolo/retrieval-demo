```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Retriever
    participant VectorStore as Vector Store
    participant Reranker
    participant LLM

    User->>Agent: Query
    Agent->>Retriever: Query
    Note over Retriever: Embed query
    Retriever->>VectorStore: Search
    VectorStore->>Retriever: Documents
    Retriever->>Reranker: Documents + Query
    Reranker->>Retriever: Reranked Documents
    Note over Retriever: Select topk
    Retriever->>Agent: Documents
    Agent->>LLM: Documents + Query
    LLM->>Agent: Answer
    Agent->>User: Answer
```
