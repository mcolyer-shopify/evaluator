# Evaluator Framework

A Python framework for running LLM judges on generic outputs with structured storage and analysis capabilities.

## Overview

The framework enables running LLM judges on arbitrary data, storing results in DuckDB, and analyzing outcomes with custom analyzers. It supports parallel execution, resumable runs, and structured output extraction.

## Architecture

### Core Components

#### Sources
- **Source** (ABC): Base class with `load() -> Iterator[dict]`
  - **Required**: Each dict must include an `id` field for evaluation deduplication
  - **Example**: `{"id": "conv_123", "text": "data", "other": "fields"}`
- **CSVSource**: Load data from CSV files using pandas
- **YAMLSource**: Load data from YAML files using pyyaml
- **Custom**: User-defined source implementations

#### Formatters
- **Formatter** (ABC): Base class with `format(data: dict) -> str`
- **XMLFormatter**: Format data as XML strings
- **Custom**: User-defined formatter implementations

#### Judges
- **Judge** (ABC): Base class with async `judge(formatted_input: str) -> dict`
- **LLMJudge** (ABC): Shared base for LLM-based judges
  - **ChatCompletionLLM**: OpenAI chat completions API with structured output
  - **ResponsesLLM**: OpenAI response format API (stub for future implementation)
- **HeuristicJudge**: Deterministic Python code-based evaluation

#### Analyzers
- **Analyzer** (ABC): Base class with `analyze(results: list[dict]) -> str`
- **Custom**: User-defined analyzer implementations

#### Storage & Orchestration
- **Storage**: DuckDB integration for results persistence
- **Evaluator**: Main orchestrator class managing the pipeline

## Implementation Details

### File Structure
```
src/evaluator/
├── __init__.py       # Export all public classes
├── py.typed         # Type hints marker
├── base.py          # Abstract base classes
├── sources.py       # CSVSource, YAMLSource implementations
├── formatters.py    # XMLFormatter implementation
├── judges.py        # LLM judge implementations with hierarchy
├── analyzers.py     # Analyzer base class
├── storage.py       # DuckDB integration
└── evaluator.py     # Main Evaluator orchestrator
```

### Dependencies
- `pydantic>=2.0.0` - Data validation and structured output
- `duckdb>=0.9.0` - Database storage
- `openai>=1.0.0` - LLM integration
- `pandas>=2.0.0` - CSV handling
- `pyyaml>=6.0` - YAML support
- `tqdm>=4.60.0` - Progress bars

### Judge Class Hierarchy
```python
class Judge(ABC):
    """Base class for all judges"""
    async def judge(self, formatted_input: str) -> dict: ...

class LLMJudge(Judge):
    """Shared base for LLM-based judges"""
    name: str
    model: str
    temperature: float
    system_prompt: str
    user_prompt_template: str
    
class ChatCompletionLLM(LLMJudge):
    """OpenAI chat completions with structured output"""
    response_model: Type[BaseModel]
    def extract_metrics(self, response: BaseModel) -> dict: ...
    
class ResponsesLLM(LLMJudge):
    """OpenAI response format (future implementation)"""
    # Stub for future development
    
class HeuristicJudge(Judge):
    """Deterministic Python code evaluation"""
    def evaluate(self, data: dict) -> dict: ...
```

### Evaluator Features

#### Core Functionality
- **Async Execution**: Parallel judge execution with `max_concurrent` control
- **Resumable Runs**: Check existing results in DuckDB to skip completed items using source-provided `id`
- **ID Management**: Uses `id` field from source data (not generated UUIDs) for deterministic deduplication
- **Structured Storage**: Results stored with schema:
  - `id` (from source data for cross-run comparison)
  - `judge_<name>.<metric_name>`
  - `judge_<name>.<metric_name>_annotations`

#### API Methods
- `run_sync()`: Synchronous wrapper using `asyncio.run()`
- `get_stats()`: Summary statistics from stored results
- `analyze()`: Run all configured analyzers on results

#### Output Format
- Judges use markdown-KV format for structured output
- LLM judges support pydantic models for structured extraction
- Metrics extracted via `extract_metrics()` method

### Key Requirements
- Framework should be resumable across runs using source-provided `id` fields
- Results written incrementally during execution
- Support for multiple LLM model types via OpenAI APIs
- Type hints throughout for better developer experience
- Extensible design for custom sources, formatters, judges, and analyzers
- **Source Contract**: All sources must emit data with an `id` field for evaluation tracking

## Usage Pattern
```python
evaluator = Evaluator(
    source=CSVSource("data.csv"),
    formatter=XMLFormatter(),
    judges=[CustomJudge()],
    analyzers=[CustomAnalyzer()],
    output_db="results.duckdb",
    output_table="evaluations",
    max_concurrent=10
)

evaluator.run_sync()
stats = evaluator.get_stats()
analysis = evaluator.analyze()
```
