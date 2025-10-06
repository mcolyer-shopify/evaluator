# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Evaluator Framework - A Python framework for running LLM judges on generic outputs with structured storage and analysis. This is an async-first framework using DuckDB for storage and OpenAI APIs for LLM evaluation.

## Development Commands

### Package Management
- `uv sync` - Install dependencies
- `uv sync --dev` - Install with development dependencies

### Testing
- `uv run pytest tests/ -v` - Run all tests with verbose output
- `uv run pytest --cov=src/evaluator --cov-report=term-missing tests/` - Run tests with coverage
- `uv run pytest -n auto tests/` - Run tests in parallel
- `uv run pytest -m "not slow" tests/` - Skip slow tests
- `uv run pytest tests/test_judges.py::TestChatCompletionLLM::test_judge_success -v` - Run single test

### Linting
- `ruff check src/ tests/` - Check code style
- `ruff format src/ tests/` - Format code

### Pre-Commit Workflow
Before committing changes, always run:
1. `ruff check src/ tests/` - Check code style and fix issues
2. `ruff format src/ tests/` - Format code consistently

## Architecture

### Core Components Pipeline
```
Source → Formatter → Judge(s) → Storage → Analyzer(s)
```

### Key Classes Structure
- **Abstract Bases**: `Source`, `Formatter`, `Judge`, `LLMJudge`, `HeuristicJudge`, `Analyzer` in `src/evaluator/base.py`
- **Implementations**: Concrete classes in respective files (sources.py, formatters.py, judges.py, etc.)
- **Orchestrator**: `Evaluator` class manages the full pipeline with async concurrency control
- **Storage**: `ResultsStorage` handles DuckDB operations with dynamic schemas

### Judge Hierarchy
```python
Judge (ABC)
├── LLMJudge (ABC)
│   ├── ChatCompletionLLM - OpenAI chat completions with structured output
│   └── ResponsesLLM - Stub for future OpenAI responses API
└── HeuristicJudge - Deterministic Python evaluation
```

### Critical Design Patterns
- All judges are async and return `dict[str, Any]`
- Resumable execution via DuckDB deduplication on `id` field
- Structured output extraction using Pydantic models for LLM judges
- Results stored as `judge_<name>.<metric>` columns
- Concurrency controlled via `asyncio.Semaphore`

### File Organization
- `src/evaluator/` - Main framework code
- `tests/` - Comprehensive test suite with mocked LLM calls
- `judge.py` - Example usage showing judge configuration
- All tests use `AsyncMock` for OpenAI API calls for fast, reliable testing

### Key Implementation Notes
- Framework uses `asyncio.run()` wrapper for sync usage (`run_sync()`)
- Storage schema adapts dynamically to judge output metrics
- All OpenAI interactions are mocked in tests for speed and reliability
- Uses `uv` for dependency management, configured in `pyproject.toml`
- always use "uv run" for any python commands in this project