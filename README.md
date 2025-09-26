# Retrieval Demo

A Python demonstration project showcasing retrieval-augmented generation (RAG) using LangGraph, OpenAI embeddings, and Qdrant vector store.

## Prerequisites

- Python 3.13.7 (see `.tool-versions`)
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

## Setup

1. **Clone and install dependencies:**
   ```bash
   uv sync --dev
   ```

2. **Set up environment variables:**
   Copy the example environment file and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your OpenAI API key and other required variables.

3. **Start the Qdrant vector store:**
   ```bash
   docker compose up -d
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Services

### Qdrant Vector Store
- **URL:** http://localhost:6333
- **Web UI:** http://localhost:6333/dashboard
- **API Documentation:** http://localhost:6333/docs
- **Configuration:** Optimized for OpenAI text-embedding-3-small (1536 dimensions)

## Usage

Run the main application:
```bash
uv run retrieval-demo
```

## Development

### Code Quality
```bash
# Format code
uv run ruff format

# Check linting
uv run ruff check

# Run tests
uv run pytest

# Run all pre-commit hooks
pre-commit run --all-files
```

### Docker Services
```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Reset Qdrant data
docker compose down -v
```