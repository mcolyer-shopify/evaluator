# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Hidden signals modal in results viewer (press `` ` `` key)
  - Display performance metrics (.meta.* fields) for tokens and latency
  - Zebra-striped table with formatted values and responsive design
  - Toggle with backtick key, ESC, or click outside to close

## [0.3.0] - 2025-10-08

### Added
- Web-based results viewer with `evaluator viewer` CLI command
- Interactive navigation and filtering of evaluation results
- Auto-loading of most recent dataset from results directory
- Dataset switching for comparing multiple evaluations
- Automatic port conflict handling

## [0.2.1] - 2025-10-07

### Changed
- Enhanced CSV column mapping documentation with practical example

### Fixed
- Removed unused aiofiles dependency
- Fixed CI test failures by adding API key to judge instantiation

## [0.2.0] - 2025-10-07

### Fixed
- Updated pyproject.toml to use `dependency-groups.dev` instead of deprecated `tool.uv.dev-dependencies`
- Removed non-existent judge.py from CI workflow ruff commands

## [0.1.0] - 2025-10-06

### Added
- Initial LLM Evaluator Framework implementation
- Async-first framework with DuckDB storage
- OpenAI API integration for LLM evaluation
- Comprehensive test suite with mocked LLM calls
- StrictnessAnalyzer with ASCII table output
- Resumable execution via DuckDB deduplication
- Structured output extraction using Pydantic models
- Support for multiple judge types (LLM and Heuristic)
- Concurrency control with asyncio.Semaphore

### Changed
- Renamed SafetyCompletenessJudge to StrictnessComparisonJudge
- Removed asyncio dependency by using run_sync()

### Fixed
- Various bug fixes and improvements throughout development