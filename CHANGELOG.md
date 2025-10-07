# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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