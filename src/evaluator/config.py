"""Configuration management for Evaluator with CLI support."""

from dataclasses import dataclass
import argparse
from typing import Optional


@dataclass
class EvaluatorConfig:
    """Configuration for Evaluator with CLI support.

    This class provides a clean interface for managing Evaluator configuration
    from both programmatic and CLI contexts.

    Examples:
        # From CLI arguments
        config = EvaluatorConfig.from_cli()
        evaluator = Evaluator(..., **config.to_evaluator_kwargs())

        # Programmatically
        config = EvaluatorConfig(sample=100, seed=123)
        evaluator = Evaluator(..., **config.to_evaluator_kwargs())
    """

    sample: Optional[int] = None
    seed: int = 42
    max_concurrent: int = 10
    output_db: Optional[str] = None
    show_progress: bool = True
    model: Optional[str] = None

    @classmethod
    def from_cli(cls, args=None, description="Run evaluation"):
        """Create config from CLI arguments.

        Args:
            args: Optional list of arguments to parse (defaults to sys.argv)
            description: Description for the argument parser

        Returns:
            EvaluatorConfig instance with values from parsed arguments
        """
        parser = cls.create_parser(description=description)
        namespace = parser.parse_args(args)
        return cls(
            sample=namespace.sample,
            seed=namespace.seed,
            max_concurrent=namespace.max_parallel,
            output_db=namespace.output_db,
            show_progress=not namespace.no_progress,
            model=namespace.model,
        )

    @classmethod
    def create_parser(cls, description="Run evaluation"):
        """Create ArgumentParser with standard evaluator options.

        Args:
            description: Description for the argument parser

        Returns:
            argparse.ArgumentParser configured with standard options
        """
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "--sample",
            type=int,
            default=None,
            help="Number of rows to sample (default: all rows)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for sampling (default: 42)",
        )
        parser.add_argument(
            "--max-parallel",
            type=int,
            default=10,
            help="Maximum number of parallel judge calls (default: 10)",
        )
        parser.add_argument(
            "--output-db",
            type=str,
            default=None,
            help="Output database path (default: auto-generated)",
        )
        parser.add_argument(
            "--no-progress",
            action="store_true",
            help="Disable progress output",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Override the judge model (e.g., gpt-4o-mini, gpt-5)",
        )
        return parser

    def to_evaluator_kwargs(self):
        """Convert config to kwargs for Evaluator.__init__.

        Returns:
            Dict of keyword arguments ready for Evaluator initialization
        """
        return {
            "sample": self.sample,
            "seed": self.seed,
            "max_concurrent": self.max_concurrent,
            "output_db": self.output_db,
            "show_progress": self.show_progress,
        }

    def __str__(self):
        """String representation of config."""
        parts = []
        if self.sample is not None:
            parts.append(f"sample={self.sample}")
        parts.append(f"seed={self.seed}")
        parts.append(f"max_concurrent={self.max_concurrent}")
        if self.output_db is not None:
            parts.append(f"output_db={self.output_db}")
        parts.append(f"show_progress={self.show_progress}")
        return f"EvaluatorConfig({', '.join(parts)})"
