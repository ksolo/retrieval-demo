# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for a retrieval demo, built using uv as the package manager. The project structure follows standard Python conventions with source code in `src/retrieval_demo/`.

## Key Dependencies

- **langgraph**: Graph-based language processing framework
- **openevals**: Evaluation framework for language models  
- **python-dotenv**: Environment variable management

## Development Commands

### Package Management
- `uv sync` - Install dependencies and sync virtual environment
- `uv add <package>` - Add new dependencies
- `uv remove <package>` - Remove dependencies
- `uv run <command>` - Run commands in the project environment

### Running the Application
- `uv run retrieval-demo` - Run the main application (defined in pyproject.toml scripts)
- `uv run python -m retrieval_demo` - Alternative way to run the module

### Code Quality
- `uv run ruff check` - Run linting
- `uv run ruff format` - Format code
- `uv run pytest` - Run tests
- `pre-commit run --all-files` - Run pre-commit hooks manually

### Development Setup
- `uv sync --dev` - Install development dependencies
- `pre-commit install` - Install pre-commit hooks

## Code Standards

- Python 3.13.7 is the target version (see .tool-versions)
- Code formatting and linting handled by Ruff
- Pre-commit hooks enforce code quality automatically
- EditorConfig enforces consistent indentation (4 spaces for Python, 2 for YAML)

## Architecture Notes

### Project Structure
- Entry point is `retrieval_demo:main` function in `src/retrieval_demo/__init__.py`
- Project uses uv's build system with `uv_build` backend
- Development dependencies are organized in dependency groups in pyproject.toml

### LangGraph Agent (`src/retrieval_demo/agent/`)
- **Graph**: Built using LangGraph StateGraph pattern with nodes connected START → retrieval_node → model_node → END
- **State**: `AgentState` (TypedDict) contains: query, messages, collection, retrieval_strategy, topk, documents
- **Nodes**: Located in `nodes.py` - each node takes AgentState and returns dict with state updates

### Retrieval System (`src/retrieval_demo/agent/retrievers/`)
- **Factory Pattern**: `make_retriever(client, collection_name, strategy)` creates retrievers based on strategy
- **Strategies**: semantic (implemented), rerank, multiquery, hybrid (placeholders for future implementation)
- **Protocol**: All retrievers implement `retrieve(query: str, limit: int) -> List[Document]`
- **Document**: Mimics LangChain's Document with `page_content` (str) and `metadata` (dict)

### Vector Store (`src/retrieval_demo/vectorstore/`)
- **Singleton Pattern**: Use `get_weaviate_client()` to get cached WeaviateClient instance (uses @cache decorator)
- **Collections**: Named by chunking strategy (e.g., "chunks_recursive_500_50", "chunks_semantic_percentile")
- **Schema**: Collections have properties: text, document_id, chunk_index, chunk_size
- **Search**: `client.semantic_search(collection_name, query, limit)` returns list of dicts with properties and metadata

### Environment & Configuration
- **API Keys**: OPENAI_API_KEY required (loaded via python-dotenv in get_weaviate_client())
- **Singleton Pattern**: Both `get_graph()` and `get_weaviate_client()` use @cache for singleton behavior
- **Connection Management**: Single WeaviateClient instance shared across all retrievers to prevent connection leaks

## Design Patterns & Principles

- **Dependency Injection**: Components receive dependencies rather than creating them (e.g., retrievers receive WeaviateClient)
- **Factory Pattern**: Centralized retriever creation in `make_retriever()` function
- **Singleton Pattern**: Cached instances for graph and Weaviate client using @cache decorator
- **Protocol/Interface**: `Retriever` Protocol defines contract without tight coupling
- **Single Responsibility**: Each class has one clear purpose (e.g., SemanticRetriever only does semantic search)

## Testing Strategy

- **Unit Tests**: Mock external dependencies (WeaviateClient) to test logic in isolation
- **Integration Tests**: Marked with `@pytest.mark.integration`, require running Weaviate instance
- **Test Organization**: Each module has corresponding test file (e.g., `semantic.py` → `test_semantic_retriever.py`)
- **Mocking**: Use unittest.mock for dependency injection testing

## Demo Purpose & Future Work

### Talk Demo Focus
This project demonstrates the impact of different retrieval strategies on RAG (Retrieval Augmented Generation) performance. The agent allows switching between strategies to compare results with the same queries and collections.

### Planned Retrieval Strategies
1. **Semantic** (✅ Implemented) - Vector similarity search using text-embedding-3-small
2. **Rerank** (🚧 TODO) - Semantic search with larger topk, then Cohere rerank to final topk
3. **MultiQuery** (🚧 TODO) - Generate alternative query phrasings, retrieve for each, apply fusion algorithm, select topk
4. **Hybrid** (🚧 TODO) - Weaviate hybrid search combining vector similarity + BM25 sparse vectors

### Implementation Notes for Future Strategies
- **Rerank**: Will need Cohere API integration, retrieves with `limit > topk` then reranks
- **MultiQuery**: Needs LLM call to generate queries, reciprocal rank fusion for combining results
- **Hybrid**: Weaviate already provides BM25 vectors, use `collection.query.hybrid()` method
- All strategies should implement the `Retriever` protocol: `retrieve(query: str, limit: int) -> List[Document]`