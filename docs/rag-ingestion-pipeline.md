```mermaid
flowchart LR
    A[(Document Source)] --> B(Parsing)
    B --> C(Chunking)
    C --> D(Embedding)
    D --> E[(Vector Store)]
```
