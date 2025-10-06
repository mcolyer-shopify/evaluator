"""Main evaluator orchestrator class."""

import asyncio
from typing import Any, Optional
import random
from datetime import datetime
from pathlib import Path
import time

from tqdm.asyncio import tqdm
from .base import Source, Formatter, Judge, Analyzer, LLMJudge
from .storage import ResultsStorage


class Evaluator:
    """Main orchestrator for the evaluation pipeline."""

    @staticmethod
    def _get_next_results_filename(results_dir: str = "results") -> str:
        """Generate the next available results filename in YYYYMMDD-N.duckdb format."""
        # Ensure results directory exists
        Path(results_dir).mkdir(exist_ok=True)

        # Get today's date in YYYYMMDD format
        today = datetime.now().strftime("%Y%m%d")

        # Find the next available number for today
        counter = 1
        while True:
            filename = f"{today}-{counter}.duckdb"
            filepath = Path(results_dir) / filename
            if not filepath.exists():
                return str(filepath)
            counter += 1

    def __init__(
        self,
        source: Source,
        formatter: Formatter,
        judges: list[Judge],
        analyzers: Optional[list[Analyzer]] = None,
        output_db: Optional[str] = None,
        max_concurrent: int = 10,
        sample: Optional[int] = None,
        seed: int = 42,
        show_progress: bool = True,
        warmup_cache: bool = True,
    ):
        """Initialize the evaluator.

        Args:
            source: Data source to load evaluation items from
            formatter: Formatter to convert data items to strings
            judges: List of judges to evaluate each item
            analyzers: Optional list of analyzers for result analysis
            output_db: Path to DuckDB database file (auto-generated if None)
            max_concurrent: Maximum number of concurrent judge evaluations
            sample: Number of items to sample (None for all items)
            seed: Random seed for sampling
            show_progress: Whether to display progress bars during evaluation
            warmup_cache: Whether to send a warmup request to prime prompt cache
        """
        self.source = source
        self.formatter = formatter
        self.judges = judges
        self.analyzers = analyzers or []
        self.output_db = output_db or self._get_next_results_filename()
        self.output_table = "evaluations"
        self.max_concurrent = max_concurrent
        self.sample = sample
        self.seed = seed
        self.show_progress = show_progress
        self.warmup_cache = warmup_cache

        # Create semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def run_sync(self) -> None:
        """Synchronous wrapper for the async evaluation pipeline."""
        asyncio.run(self.run())

    async def run(self) -> None:
        """Run the evaluation pipeline asynchronously."""
        # Track total evaluation time
        evaluation_start_time = time.time()

        # Set random seed for reproducible sampling
        random.seed(self.seed)

        # Display output database location if show_progress is enabled
        if self.show_progress:
            print(f"Results will be saved to: {self.output_db}")

        # Initialize storage
        with ResultsStorage(self.output_db, self.output_table) as storage:
            # Create table schema based on judges
            columns = self._get_table_schema()
            storage.create_table(columns)

            # Load all data items with progress bar
            data_items = list(self.source.load())
            if self.show_progress:
                print(f"Loading data... Loaded {len(data_items)} items")

            # Apply sampling if specified
            if self.sample is not None and self.sample < len(data_items):
                data_items = random.sample(data_items, self.sample)

            # Validate data items have required id field
            for data_item in data_items:
                if "id" not in data_item:
                    raise ValueError(
                        f"Data item missing required 'id' field: {data_item}"
                    )

            # Warm up cache if requested
            if self.warmup_cache and data_items:
                if self.show_progress:
                    print("Warming up cache...")
                await self._warmup_cache(data_items[0])

            # Create evaluation tasks (let _evaluate_item handle existence checking)
            tasks = []
            for data_item in data_items:
                task = self._evaluate_item_with_progress(data_item, storage)
                tasks.append(task)

            # Run evaluations with progress bar
            if self.show_progress:
                # Use tqdm to track completion of tasks
                completed = 0
                with tqdm(
                    total=len(tasks),
                    desc="Evaluating",
                    unit="items",
                    ncols=100,
                ) as pbar:
                    # Use as_completed to get progress updates as tasks finish
                    for coro in asyncio.as_completed(tasks):
                        await coro
                        completed += 1
                        pbar.update(1)
            else:
                # Run without progress tracking
                await asyncio.gather(*tasks)

            # Run analyzers automatically after evaluation completes
            if self.analyzers:
                if self.show_progress:
                    print("\nRunning analyzers...")

                # Get all results for analysis
                results = storage.get_all_results()

                analysis_parts = []
                for analyzer in self.analyzers:
                    try:
                        analysis = analyzer.analyze(results)
                        analysis_parts.append(analysis)
                    except Exception as e:
                        error_msg = (
                            f"Analyzer {analyzer.__class__.__name__} failed: {e}"
                        )
                        analysis_parts.append(error_msg)
                        if self.show_progress:
                            print(f"Warning: {error_msg}")

                # Display analysis results if show_progress is enabled
                if self.show_progress and analysis_parts:
                    print("\nAnalysis Results:")
                    for analysis in analysis_parts:
                        print(analysis)

            # Print completion summary if show_progress is enabled
            if self.show_progress:
                evaluation_end_time = time.time()
                total_runtime = evaluation_end_time - evaluation_start_time

                stats = self.get_stats()
                stats["performance.total_runtime_seconds"] = total_runtime

                # Calculate throughput
                if stats["total"] > 0:
                    stats["performance.items_per_second"] = (
                        stats["total"] / total_runtime
                    )

                self._print_completion_summary(stats)

    async def _evaluate_item_with_progress(
        self, data_item: dict[str, Any], storage: ResultsStorage
    ) -> dict[str, str]:
        """Evaluate a single data item (wrapper for progress tracking)."""
        await self._evaluate_item(data_item, storage)
        # Return status info for progress tracking
        return {"id": str(data_item["id"]), "status": "completed"}

    async def _evaluate_item(
        self, data_item: dict[str, Any], storage: ResultsStorage
    ) -> None:
        """Evaluate a single data item."""
        async with self._semaphore:
            # Extract ID from source data
            if "id" not in data_item:
                raise ValueError(f"Data item missing required 'id' field: {data_item}")
            item_id = str(data_item["id"])

            # Format the data
            formatted_input = self.formatter.format(data_item)

            # Initialize result with ID and input
            result = {"id": item_id, "input": formatted_input}

            # Run all judges
            for judge in self.judges:
                try:
                    # Measure judge execution time
                    start_time = time.time()
                    judge_result = await judge.judge(formatted_input)
                    end_time = time.time()

                    # Add timing metadata
                    judge_result["meta.latency_ms"] = (end_time - start_time) * 1000

                    # Add judge results with proper naming convention
                    for metric_name, value in judge_result.items():
                        column_name = f"judge_{judge.name}.{metric_name}"
                        result[column_name] = value

                except Exception as e:
                    # Log error and continue with other judges
                    error_column = f"judge_{judge.name}.error"
                    result[error_column] = str(e)

            # Insert result into storage
            storage.insert_result(result)

    def _get_table_schema(self) -> dict[str, str]:
        """Generate table schema based on judges."""
        columns = {"id": "VARCHAR PRIMARY KEY", "input": "VARCHAR"}

        for judge in self.judges:
            # Always add error column for each judge
            columns[f"judge_{judge.name}.error"] = "VARCHAR"

            # Always add meta columns for token tracking and timing
            columns[f"judge_{judge.name}.meta.tokens_total"] = "DOUBLE"
            columns[f"judge_{judge.name}.meta.tokens_input"] = "DOUBLE"
            columns[f"judge_{judge.name}.meta.tokens_output"] = "DOUBLE"
            columns[f"judge_{judge.name}.meta.tokens_cached"] = "DOUBLE"
            columns[f"judge_{judge.name}.meta.latency_ms"] = "DOUBLE"

            # For judges with extract_metrics method, introspect the actual metrics
            if hasattr(judge, "extract_metrics"):
                # Try to determine columns from the judge's response model or output_format if available
                output_format = getattr(judge, "output_format", None) or getattr(
                    judge, "response_model", None
                )
                if output_format:
                    try:
                        # Create a dummy instance to see what extract_metrics returns
                        model_fields = output_format.model_fields
                        for field_name, field_info in model_fields.items():
                            column_name = f"judge_{judge.name}.{field_name}"
                            # Determine SQL type based on Python type annotation
                            if field_info.annotation in [int, float]:
                                columns[column_name] = "DOUBLE"
                            else:
                                columns[column_name] = "VARCHAR"
                    except Exception:
                        # Fall back to generic schema if introspection fails
                        columns[f"judge_{judge.name}.response"] = "VARCHAR"
                else:
                    # For judges with extract_metrics but no response_model,
                    # try to call extract_metrics with the judge's response to see the structure
                    if hasattr(judge, "response"):
                        try:
                            dummy_metrics = judge.extract_metrics(judge.response)
                            for metric_name, metric_value in dummy_metrics.items():
                                column_name = f"judge_{judge.name}.{metric_name}"
                                if isinstance(metric_value, (int, float)):
                                    columns[column_name] = "DOUBLE"
                                else:
                                    columns[column_name] = "VARCHAR"
                        except Exception:
                            columns[f"judge_{judge.name}.response"] = "VARCHAR"
                    else:
                        # Generic fallback for judges without response models or response attribute
                        columns[f"judge_{judge.name}.response"] = "VARCHAR"
            else:
                # Generic fallback for judges without extract_metrics
                columns[f"judge_{judge.name}.response"] = "VARCHAR"

        return columns

    async def _warmup_cache(self, sample_data_item: dict[str, Any]) -> None:
        """Warm up the cache by sending a request with the common prefix."""
        # Format a sample input to prime the cache
        formatted_input = self.formatter.format(sample_data_item)

        # Send warmup requests for each judge to prime their respective caches
        for judge in self.judges:
            try:
                # Only warm up LLM judges
                if isinstance(judge, LLMJudge):
                    await judge.judge(formatted_input)
            except Exception:
                # Ignore warmup failures - they're not critical
                pass

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics from stored results."""
        with ResultsStorage(self.output_db, self.output_table) as storage:
            return storage.get_stats()

    def _print_completion_summary(self, stats: dict[str, Any]) -> None:
        """Print formatted completion summary with performance metrics."""
        print("\n" + "=" * 60)
        print("🏁 EVALUATION COMPLETE")
        print("=" * 60)

        # Basic stats
        total_items = stats.get("total", 0)
        runtime = stats.get("performance.total_runtime_seconds", 0)
        throughput = stats.get("performance.items_per_second", 0)

        print(f"📊 Items processed: {total_items:,}")

        # Error reporting
        total_errors = stats.get("total_errors", 0)
        if total_errors > 0:
            print(f"❌ Errors encountered: {total_errors:,}")

        print(f"⏱️  Total runtime: {runtime:.2f}s")
        if throughput > 0:
            print(f"🚀 Throughput: {throughput:.2f} items/second")

        # Performance metrics
        if (
            "performance.latency_mean_ms" in stats
            and stats.get("performance.latency_mean_ms") is not None
        ):
            print("\n⚡ LATENCY METRICS")
            print(
                f"   Mean:    {stats.get('performance.latency_mean_ms', 0) or 0:.0f}ms"
            )
            print(
                f"   Median:  {stats.get('performance.latency_p50_ms', 0) or 0:.0f}ms"
            )
            print(
                f"   95th %:  {stats.get('performance.latency_p95_ms', 0) or 0:.0f}ms"
            )
            print(
                f"   Range:   {stats.get('performance.latency_min_ms', 0) or 0:.0f}ms - {stats.get('performance.latency_max_ms', 0) or 0:.0f}ms"
            )

        # Token usage
        if (
            "performance.total_tokens_used" in stats
            and stats.get("performance.total_tokens_used") is not None
        ):
            total_tokens = stats.get("performance.total_tokens_used", 0) or 0
            avg_tokens = stats.get("performance.avg_tokens_per_item", 0) or 0
            print("\n🔢 TOKEN USAGE")
            print(f"   Total tokens: {total_tokens:,}")
            print(f"   Avg per item: {avg_tokens:.0f}")

            # Cache efficiency
            cache_rate = stats.get("performance.cache_hit_rate_pct", 0) or 0
            cached_tokens = stats.get("performance.total_cached_tokens", 0) or 0
            print(f"   Cache hits:   {cache_rate:.1f}% ({cached_tokens:,} tokens)")

        print("=" * 60)

    def analyze(self) -> str:
        """Run all analyzers on the stored results."""
        if not self.analyzers:
            return "No analyzers configured"

        with ResultsStorage(self.output_db, self.output_table) as storage:
            results = storage.get_all_results()

            analysis_parts = []
            for analyzer in self.analyzers:
                try:
                    analysis = analyzer.analyze(results)
                    analysis_parts.append(analysis)
                except Exception as e:
                    analysis_parts.append(
                        f"Analyzer {analyzer.__class__.__name__} failed: {e}"
                    )

            return "\n\n".join(analysis_parts)
