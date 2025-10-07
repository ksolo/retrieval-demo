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

### Evaluation Commands
- `uv run prepare-eval-dataset` - Prepare evaluation dataset (categorize, upload to LangSmith, ingest to vector store)
- `uv run run-chunking-eval` - Evaluate chunking strategies and submit results to LangSmith
- `uv run run-retrieval-eval --collection-name <name>` - Evaluate retrieval strategies for a specific collection

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
- **LangSmith**: LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_TRACING_V2 for evaluation tracking
- **Singleton Pattern**: Both `get_graph()` and `get_weaviate_client()` use @cache for singleton behavior
- **Connection Management**: Single WeaviateClient instance shared across all retrievers to prevent connection leaks
- **Model**: Using `gpt-5-mini` for LLM judges and categorization

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
- **TDD Approach**: Tests written first, then implementation (37 passing tests for evaluation infrastructure)

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

## Evaluation Infrastructure (`eval/`)

### Overview
Complete evaluation system for comparing chunking and retrieval strategies using LLM-as-judge with openevals and LangSmith tracking.

### Dataset Preparation
- **Source**: `neural-bridge/rag-dataset-12000` from HuggingFace (12k examples)
- **Categorization**: LLM-based categorization into 5 categories (cached in `eval/data/categorization_cache.json`)
- **Sampling**: Stratified sampling (100 samples per category = 500 total)
- **Storage**: Uploaded to LangSmith as dataset, context stored in Weaviate collections
- **Versioning**: Tracked via HuggingFace git commit hash from download_checksums

### Core Components

#### `eval/dataset.py` - EvalDatasetManager
- Creates/updates LangSmith datasets with proper input/output/metadata structure
- **Inputs**: `{"question": str}` - only the query
- **Outputs**: `{"answer": str}` - expected answer
- **Metadata**: `{"eval_id": str, "category": str, "original_index": int}` - tracking info
- Context NOT stored in LangSmith (it's in vector store)

#### `eval/metrics.py` - Metrics Calculation
- **RetrievalMetrics**: Per-sample metrics (eval_id, retrieval_relevance, groundedness, recall@1/3/5, precision@k, latency_ms)
- **AggregateMetrics**: Averages across all samples
- **MetricsCalculator**: Computes all metrics including recall@k and precision@k

#### `eval/judges.py` - LLM Judges
- **RetrievalRelevanceJudge**: Uses openevals `RAG_RETRIEVAL_RELEVANCE_PROMPT`
- **GroundednessJudge**: Uses openevals `RAG_GROUNDEDNESS_PROMPT`
- Both return scores 0.0-1.0, handle errors gracefully
- Default model: `gpt-5-mini`

#### `eval/langsmith_evaluators.py` - LangSmith Integration
- `create_langsmith_evaluators()`: Creates evaluator functions for LangSmith API
- Returns list of evaluators: retrieval_relevance, groundedness, recall@1/3/5, precision@k
- Shared between chunking and retrieval evaluation scripts
- Each evaluator returns `{"key": str, "score": float}` for LangSmith

#### `eval/evaluator.py` - StrategyEvaluator (Not currently used)
- Orchestrates evaluation flow (kept for potential future use)
- Note: CLI scripts now use LangSmith's `client.evaluate()` directly

### Chunking Strategies Evaluated
1. RecursiveTextSplitterStrategy(chunk_size=200, chunk_overlap=20)
2. RecursiveTextSplitterStrategy(chunk_size=500, chunk_overlap=50)
3. RecursiveTextSplitterStrategy(chunk_size=1000, chunk_overlap=100)
4. RecursiveTextSplitterStrategy(chunk_size=2000, chunk_overlap=200)
5. SemanticChunkerStrategy (percentile threshold)

### Data Flow
1. **Categorization**: `src/retrieval_demo/dataloader/categorization.py`
   - `Categorizer`: LLM categorizes samples (with caching)
   - `CategorizedDataset`: Manages cache with version tracking

2. **Data Loading**: `src/retrieval_demo/dataloader/data/loader.py`
   - `get_categorized_stratified_sample()`: Full pipeline (load → categorize → sample)
   - `get_samples_by_ids()`: Retrieve specific samples by eval_id
   - Uses `eval_id` format: `rag12000_{original_index}`

3. **Chunking**: `src/retrieval_demo/dataloader/chunking/`
   - Updated to support `document_id: str` (was int) for eval_id
   - Added `category: Optional[str]` to ChunkMetadata
   - Both recursive and semantic strategies updated

4. **Ingestion**: `src/retrieval_demo/pipeline/ingestion.py`
   - `process_dataset(eval_samples=...)`: Accepts eval samples directly
   - Extracts eval_id and category from samples
   - Passes to chunking strategies for metadata

### Evaluation Metrics

**Per-Sample Metrics:**
- **Retrieval Relevance** (0-1): LLM judges if retrieved chunks are relevant to question
- **Groundedness** (0-1): LLM judges if generated answer is grounded in retrieved chunks
- **Recall@1** (0/1): First retrieved chunk from correct document?
- **Recall@3** (0/1): Any of top 3 from correct document?
- **Recall@5** (0/1): Any of top 5 from correct document?
- **Precision@k** (0-1): Proportion of relevant chunks in top-k
- **Latency** (ms): Retrieval + generation time

**Aggregate Metrics:**
- Averages of all above metrics
- Category-level breakdown (5 categories)

### CLI Workflow

1. **Prepare Dataset** (`prepare-eval-dataset`)
   - Load dataset from HuggingFace
   - Categorize using LLM (cached)
   - Stratified sample (100 per category)
   - Upload to LangSmith
   - Ingest into Weaviate collections (all chunking strategies)

2. **Evaluate Chunking** (`run-chunking-eval`)
   - Verify collections exist
   - For each chunking strategy:
     - Run graph on all eval samples
     - Apply evaluators (relevance, groundedness, recall, precision)
     - Submit scores to LangSmith via `client.evaluate()`
   - Results tracked in LangSmith experiments

3. **Evaluate Retrieval** (`run-retrieval-eval --collection-name <name>`)
   - Verify collection exists
   - For each retrieval strategy (currently only semantic):
     - Run graph on all eval samples
     - Apply same evaluators
     - Submit scores to LangSmith
   - Compare retrieval strategies on same chunking

### LangSmith Integration
- All graph executions automatically traced (messages, documents, latency)
- Evaluators submit scores as feedback
- Experiments named: `{chunking|retrieval}_eval_{collection}_{strategy}`
- View side-by-side comparisons at https://smith.langchain.com

### Key Design Decisions
- **Groundedness over Correctness**: Evaluates if answer is supported by retrieved chunks (tests retrieval quality)
- **Document-level Recall**: Uses eval_id to check if correct source document retrieved (works across chunk sizes)
- **Recall@1/3/5**: Shows ranking quality - is correct document at top vs just "in the mix"?
- **No MRR**: Dropped in favor of simpler recall@k metrics
- **LangSmith as Source of Truth**: Dataset in LangSmith, verify collections match
- **Separation of Concerns**: Prepare (ingest) and evaluate are separate commands