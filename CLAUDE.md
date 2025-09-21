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

- Entry point is `retrieval_demo:main` function in `src/retrieval_demo/__init__.py`
- Project uses uv's build system with `uv_build` backend
- Development dependencies are organized in dependency groups in pyproject.toml