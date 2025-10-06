"""Base classes for the evaluator framework."""

from abc import ABC, abstractmethod
from typing import Any, Iterator


class Source(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def load(self) -> Iterator[dict[str, Any]]:
        """Load data and return an iterator of dictionaries.

        Each dictionary MUST contain an 'id' field that uniquely identifies
        the item for deduplication and resumable execution.

        Returns:
            Iterator yielding dicts with required 'id' field plus any other data.
        """
        ...


class Formatter(ABC):
    """Abstract base class for data formatters."""

    @abstractmethod
    def format(self, data: dict[str, Any]) -> str:
        """Format data dictionary into a string."""
        ...


class Judge(ABC):
    """Abstract base class for all judges."""

    name: str

    @abstractmethod
    async def judge(self, formatted_input: str) -> dict[str, Any]:
        """Judge the formatted input and return metrics."""
        ...


class LLMJudge(Judge):
    """Abstract base class for LLM-based judges."""

    model: str
    temperature: float
    system_prompt: str
    user_prompt_template: str

    @abstractmethod
    async def judge(self, formatted_input: str) -> dict[str, Any]:
        """Judge the formatted input using an LLM."""
        ...


class HeuristicJudge(Judge):
    """Abstract base class for deterministic Python code-based judges."""

    @abstractmethod
    def evaluate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate data using deterministic logic."""
        ...

    async def judge(self, formatted_input: str) -> dict[str, Any]:
        """Default implementation that parses formatted input and evaluates."""
        # This would need parsing logic to convert formatted_input back to data
        # For now, this is a placeholder
        raise NotImplementedError("HeuristicJudge requires custom implementation")


class Analyzer(ABC):
    """Abstract base class for result analyzers."""

    @abstractmethod
    def analyze(self, results: list[dict[str, Any]]) -> str:
        """Analyze results and return a formatted analysis string."""
        ...
