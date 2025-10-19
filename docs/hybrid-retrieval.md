```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Retriever
    participant VectorStore as Vector Store
    participant LLM

    User->>Agent: Query
    Agent->>Retriever: Query
    Note over Retriever: Embed query
    par Dense and Sparse Search
        Retriever->>VectorStore: Dense vector search
        VectorStore->>Retriever: Dense results
        Retriever->>VectorStore: Sparse vector search
        VectorStore->>Retriever: Sparse results
    end
    Note over Retriever: Fusion algorithm - merge and rank
    Note over Retriever: Select topk
    Retriever->>Agent: Documents
    Agent->>LLM: Documents + Query
    LLM->>Agent: Answer
    Agent->>User: Answer
```
