```mermaid
flowchart LR
    A[(Document Source)] --> B(Parsing)
    B --> C(Chunking)
    C --> D(Dense Embedding)
    C --> E(Sparse Embedding)
    D --> F[(Vector Store)]
    E --> F
```
