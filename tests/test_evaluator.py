"""Tests for the main Evaluator orchestrator."""

import pytest
import asyncio
from unittest.mock import patch

from src.evaluator.evaluator import Evaluator
from src.evaluator.base import Source, Formatter, Judge, LLMJudge, Analyzer
from src.evaluator.sources import CSVSource
from src.evaluator.formatters import XMLFormatter
from src.evaluator.judges import ChatCompletionLLM


class MockSource(Source):
    """Mock source for testing."""

    def __init__(self, data):
        self.data = data

    def load(self):
        for item in self.data:
            # Ensure each item has an id field
            if "id" not in item:
                raise ValueError(f"Source data missing required 'id' field: {item}")
            yield item


class MockFormatter(Formatter):
    """Mock formatter for testing."""

    def format(self, data):
        return f"Formatted: {data}"


class MockJudge(Judge):
    """Mock judge for testing."""

    def __init__(self, name, response):
        self.name = name
        self.response = response

    async def judge(self, formatted_input):
        return self.response

    def extract_metrics(self, response):
        """Extract metrics from response for schema generation."""
        return self.response


class MockLLMJudge(LLMJudge):
    """Mock LLM judge for testing."""

    def __init__(self, name, response):
        self.name = name
        self.model = "mock-model"
        self.temperature = 0.0
        self.system_prompt = "Mock prompt"
        self.user_prompt_template = "{formatted_input}"
        self.response = response

    async def judge(self, formatted_input):
        return self.response

    def extract_metrics(self, response):
        """Extract metrics from response for schema generation."""
        return self.response


class MockAnalyzer(Analyzer):
    """Mock analyzer for testing."""

    def analyze(self, results):
        return f"Analysis of {len(results)} results"


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_initialization(self, temp_db_path):
        """Test evaluator initialization."""
        source = MockSource([])
        formatter = MockFormatter()
        judges = [MockJudge("test", {})]

        evaluator = Evaluator(
            source=source,
            formatter=formatter,
            judges=judges,
            output_db=temp_db_path,
            max_concurrent=5,
        )

        assert evaluator.source == source
        assert evaluator.formatter == formatter
        assert evaluator.judges == judges
        assert evaluator.analyzers == []
        assert evaluator.output_db == temp_db_path
        assert evaluator.output_table == "evaluations"
        assert evaluator.max_concurrent == 5
        assert evaluator._semaphore._value == 5

    def test_initialization_with_analyzers(self, temp_db_path):
        """Test evaluator initialization with analyzers."""
        analyzer = MockAnalyzer()

        evaluator = Evaluator(
            source=MockSource([]),
            formatter=MockFormatter(),
            judges=[MockJudge("test", {})],
            analyzers=[analyzer],
            output_db=temp_db_path,
        )

        assert len(evaluator.analyzers) == 1
        assert evaluator.analyzers[0] == analyzer

    async def test_evaluate_single_item(self, temp_db_path):
        """Test evaluating a single item."""
        test_data = [{"id": 1, "text": "Test input"}]

        source = MockSource(test_data)
        formatter = MockFormatter()
        judge_response = {"score": 0.85, "confidence": 0.9}
        judge = MockJudge("quality", judge_response)

        evaluator = Evaluator(
            source=source,
            formatter=formatter,
            judges=[judge],
            output_db=temp_db_path,
        )

        await evaluator.run()

        # Verify results were stored
        stats = evaluator.get_stats()
        assert stats["total"] == 1

    async def test_evaluate_multiple_items(self, temp_db_path):
        """Test evaluating multiple items."""
        test_data = [
            {"id": 1, "text": "First test"},
            {"id": 2, "text": "Second test"},
            {"id": 3, "text": "Third test"},
        ]

        source = MockSource(test_data)
        formatter = MockFormatter()
        judge = MockJudge("test", {"score": 0.8})

        evaluator = Evaluator(
            source=source, formatter=formatter, judges=[judge], output_db=temp_db_path
        )

        await evaluator.run()

        stats = evaluator.get_stats()
        assert stats["total"] == 3

    async def test_evaluate_multiple_judges(self, temp_db_path):
        """Test evaluating with multiple judges."""
        test_data = [{"id": "multi-judge-1", "text": "Test input"}]

        source = MockSource(test_data)
        formatter = MockFormatter()

        judges = [
            MockJudge("quality", {"score": 0.8, "annotation": "Good"}),
            MockJudge("safety", {"score": 0.9, "confidence": 0.95}),
            MockJudge("coherence", {"rating": 4, "explanation": "Clear"}),
        ]

        evaluator = Evaluator(
            source=source, formatter=formatter, judges=judges, output_db=temp_db_path
        )

        await evaluator.run()

        # Get all results and verify judge outputs are properly stored
        from src.evaluator.storage import ResultsStorage

        with ResultsStorage(temp_db_path, "evaluations") as storage:
            results = storage.get_all_results()

        assert len(results) == 1
        result = results[0]

        # Check judge results are stored with proper naming
        assert "judge_quality.score" in result
        assert "judge_quality.annotation" in result
        assert "judge_safety.score" in result
        assert "judge_safety.confidence" in result
        assert "judge_coherence.rating" in result
        assert "judge_coherence.explanation" in result

    async def test_judge_error_handling(self, temp_db_path):
        """Test that judge errors are handled gracefully."""
        test_data = [{"id": "error-test", "text": "Test"}]

        source = MockSource(test_data)
        formatter = MockFormatter()

        # Create a judge that raises an exception
        failing_judge = MockJudge("failing", {})

        async def failing_judge_method(formatted_input):
            raise ValueError("Test error")

        failing_judge.judge = failing_judge_method

        working_judge = MockJudge("working", {"score": 0.8})

        evaluator = Evaluator(
            source=source,
            formatter=formatter,
            judges=[failing_judge, working_judge],
            output_db=temp_db_path,
        )

        await evaluator.run()

        # Verify evaluation completed despite error
        stats = evaluator.get_stats()
        assert stats["total"] == 1

        # Check error was recorded
        from src.evaluator.storage import ResultsStorage

        with ResultsStorage(temp_db_path, "evaluations") as storage:
            results = storage.get_all_results()

        result = results[0]
        assert "judge_failing.error" in result
        assert "Test error" in result["judge_failing.error"]
        assert "judge_working.score" in result
        assert result["judge_working.score"] == 0.8

    def test_get_table_schema(self, temp_db_path):
        """Test table schema generation."""
        judges = [
            MockJudge("quality", {"score": 0.5, "annotation": "test"}),
            MockLLMJudge("llm_judge", {"confidence": 0.9}),
        ]

        evaluator = Evaluator(
            source=MockSource([]),
            formatter=MockFormatter(),
            judges=judges,
            output_db=temp_db_path,
        )

        schema = evaluator._get_table_schema()

        # Check required columns
        assert "id" in schema
        assert schema["id"] == "VARCHAR PRIMARY KEY"

        # Check judge-specific columns
        assert "judge_quality.error" in schema
        assert "judge_quality.score" in schema
        assert "judge_quality.annotation" in schema

        assert "judge_llm_judge.error" in schema
        assert "judge_llm_judge.confidence" in schema

    def test_run_sync(self, temp_db_path):
        """Test synchronous wrapper."""
        test_data = [{"id": "sync-test", "text": "Sync test"}]

        evaluator = Evaluator(
            source=MockSource(test_data),
            formatter=MockFormatter(),
            judges=[MockJudge("test", {"score": 0.7})],
            output_db=temp_db_path,
        )

        # Should complete without error
        evaluator.run_sync()

        stats = evaluator.get_stats()
        assert stats["total"] == 1

    def test_get_stats_empty(self, temp_db_path):
        """Test getting stats when no evaluations have run."""
        evaluator = Evaluator(
            source=MockSource([]),
            formatter=MockFormatter(),
            judges=[MockJudge("test", {})],
            output_db=temp_db_path,
        )

        stats = evaluator.get_stats()
        assert stats["total"] == 0

    async def test_concurrent_evaluation(self, temp_db_path):
        """Test concurrent evaluation with semaphore."""
        # Create data that would normally process concurrently
        test_data = [{"id": i, "text": f"Test {i}"} for i in range(10)]

        # Mock judge that simulates some processing time
        async def slow_judge(formatted_input):
            await asyncio.sleep(0.01)  # Small delay
            return {"score": 0.8}

        judge = MockJudge("slow", {"score": 0.8})
        judge.judge = slow_judge

        evaluator = Evaluator(
            source=MockSource(test_data),
            formatter=MockFormatter(),
            judges=[judge],
            output_db=temp_db_path,
            max_concurrent=3,  # Limit concurrency
        )

        await evaluator.run()

        stats = evaluator.get_stats()
        assert stats["total"] == 10

    def test_analyze_no_analyzers(self, temp_db_path):
        """Test analyze method with no analyzers."""
        evaluator = Evaluator(
            source=MockSource([]),
            formatter=MockFormatter(),
            judges=[MockJudge("test", {})],
            output_db=temp_db_path,
        )

        result = evaluator.analyze()
        assert result == "No analyzers configured"

    async def test_analyze_with_analyzers(self, temp_db_path):
        """Test analyze method with analyzers."""
        test_data = [{"id": "analysis-test", "text": "Analysis test"}]

        analyzer1 = MockAnalyzer()
        analyzer2 = MockAnalyzer()

        evaluator = Evaluator(
            source=MockSource(test_data),
            formatter=MockFormatter(),
            judges=[MockJudge("test", {"score": 0.8})],
            analyzers=[analyzer1, analyzer2],
            output_db=temp_db_path,
        )

        # Run evaluation first
        await evaluator.run()

        # Then analyze
        analysis = evaluator.analyze()

        # Should contain results from both analyzers
        assert "Analysis of 1 results" in analysis
        # Should be joined with newlines
        lines = analysis.split("\n\n")
        assert len(lines) == 2

    async def test_analyzer_error_handling(self, temp_db_path):
        """Test that analyzer errors are handled gracefully."""
        test_data = [{"id": "analyzer-error-test", "text": "Test"}]

        # Create an analyzer that raises an exception
        failing_analyzer = MockAnalyzer()
        failing_analyzer.analyze = lambda results: exec(
            'raise ValueError("Analyzer error")'
        )

        working_analyzer = MockAnalyzer()

        evaluator = Evaluator(
            source=MockSource(test_data),
            formatter=MockFormatter(),
            judges=[MockJudge("test", {"score": 0.8})],
            analyzers=[failing_analyzer, working_analyzer],
            output_db=temp_db_path,
        )

        await evaluator.run()
        analysis = evaluator.analyze()

        # Should contain error message and working analyzer result
        assert "MockAnalyzer failed" in analysis
        assert "Analysis of 1 results" in analysis


@pytest.mark.integration
class TestEvaluatorIntegration:
    """Integration tests using real components."""

    async def test_csv_to_xml_evaluation(
        self, sample_csv_file, temp_db_path, mock_openai_client
    ):
        """Test full pipeline with CSV source and XML formatter."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            source = CSVSource(sample_csv_file)
            formatter = XMLFormatter()
            judge = ChatCompletionLLM()
            judge.name = "integration_test"

            evaluator = Evaluator(
                source=source,
                formatter=formatter,
                judges=[judge],
                output_db=temp_db_path,
            )

            await evaluator.run()

            # Verify results
            stats = evaluator.get_stats()
            assert stats["total"] == 3  # sample_data has 3 items

            # Verify OpenAI was called for each item plus warmup
            assert (
                mock_openai_client.chat.completions.create.call_count == 4
            )  # 1 warmup + 3 items

    async def test_resumable_execution(self, temp_db_path):
        """Test that evaluator can resume from partial execution."""
        # Note: This test demonstrates the concept but actual resumability
        # would require deterministic IDs rather than UUIDs
        test_data = [{"id": 1, "text": "First"}, {"id": 2, "text": "Second"}]

        # Mock judge that fails on second item
        call_count = 0

        async def conditional_judge(formatted_input):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Simulated failure")
            return {"score": 0.8}

        judge = MockJudge("resumable", {"score": 0.8})
        judge.judge = conditional_judge

        evaluator = Evaluator(
            source=MockSource(test_data),
            formatter=MockFormatter(),
            judges=[judge],
            output_db=temp_db_path,
        )

        # First run - should process both items, second will have error
        await evaluator.run()

        # Check that both items were attempted
        stats = evaluator.get_stats()
        assert stats["total"] == 2  # Both items attempted

        # Verify that first succeeded and second has error
        from src.evaluator.storage import ResultsStorage

        with ResultsStorage(temp_db_path, "evaluations") as storage:
            results = storage.get_all_results()

        assert len(results) == 2

        # Find results by checking for errors
        successful_results = [
            r
            for r in results
            if not any(k.endswith(".error") and v for k, v in r.items())
        ]
        error_results = [
            r for r in results if any(k.endswith(".error") and v for k, v in r.items())
        ]

        assert len(successful_results) == 1  # First item succeeded
        assert len(error_results) == 1  # Second item failed

    def test_source_missing_id_field(self, temp_db_path):
        """Test that sources missing id field raise error."""
        test_data = [{"text": "No ID field"}]  # Missing id field

        with pytest.raises(ValueError, match="Source data missing required 'id' field"):
            source = MockSource(test_data)
            list(source.load())

    async def test_progress_bar_disabled(self, temp_db_path):
        """Test evaluator with progress bar disabled."""
        test_data = [{"id": "progress-test", "text": "Test"}]

        evaluator = Evaluator(
            source=MockSource(test_data),
            formatter=MockFormatter(),
            judges=[MockJudge("test", {"score": 0.8})],
            output_db=temp_db_path,
            show_progress=False,
        )

        # Should run without error
        await evaluator.run()

        # Check that results were stored
        stats = evaluator.get_stats()
        assert stats["total"] == 1

    async def test_progress_bar_enabled(self, temp_db_path):
        """Test evaluator with progress bar enabled."""
        test_data = [
            {"id": "progress-test-1", "text": "Test 1"},
            {"id": "progress-test-2", "text": "Test 2"},
        ]

        evaluator = Evaluator(
            source=MockSource(test_data),
            formatter=MockFormatter(),
            judges=[MockJudge("test", {"score": 0.8})],
            output_db=temp_db_path,
            show_progress=True,
        )

        # Should run without error (progress bar will be visible during test)
        await evaluator.run()

        # Check that results were stored
        stats = evaluator.get_stats()
        assert stats["total"] == 2
