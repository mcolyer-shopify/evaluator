"""Tests for analyzer implementations and base analyzer functionality."""

import pytest

from src.evaluator.analyzers import Analyzer
from src.evaluator.base import Analyzer as BaseAnalyzer


class TestAnalyzersModule:
    """Test the analyzers module structure."""

    def test_analyzer_import(self):
        """Test that Analyzer can be imported from analyzers module."""
        assert Analyzer is BaseAnalyzer

    def test_analyzer_is_abstract(self):
        """Test that imported Analyzer is still abstract."""
        with pytest.raises(TypeError):
            Analyzer()


class MockAnalyzer(Analyzer):
    """Mock analyzer for testing analyzer functionality."""

    def analyze(self, results):
        if not results:
            return "No results provided"

        # Basic statistics
        total_count = len(results)

        # Find numeric columns (judge metrics)
        numeric_columns = set()
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key != "id":
                    numeric_columns.add(key)

        analysis_parts = [f"Total results: {total_count}"]

        # Calculate stats for numeric columns
        for column in sorted(numeric_columns):
            values = []
            for result in results:
                if column in result and isinstance(result[column], (int, float)):
                    values.append(result[column])

            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                analysis_parts.append(
                    f"{column}: avg={avg:.3f}, min={min_val:.3f}, max={max_val:.3f}"
                )

        return "\n".join(analysis_parts)


class CountAnalyzer(Analyzer):
    """Simple analyzer that just counts results."""

    def analyze(self, results):
        return f"Found {len(results)} evaluation results"


class ErrorAnalyzer(Analyzer):
    """Analyzer that focuses on errors in results."""

    def analyze(self, results):
        if not results:
            return "No results to analyze for errors"

        error_count = 0
        judge_errors = {}

        for result in results:
            for key, value in result.items():
                if key.endswith(".error") and value:
                    error_count += 1
                    judge_name = key.split(".")[0]
                    if judge_name not in judge_errors:
                        judge_errors[judge_name] = []
                    judge_errors[judge_name].append(value)

        if error_count == 0:
            return f"No errors found in {len(results)} results"

        analysis_parts = [f"Found {error_count} errors in {len(results)} results:"]

        for judge, errors in judge_errors.items():
            analysis_parts.append(f"  {judge}: {len(errors)} errors")
            for error in errors[:3]:  # Show first 3 errors
                analysis_parts.append(f"    - {error}")
            if len(errors) > 3:
                analysis_parts.append(f"    ... and {len(errors) - 3} more")

        return "\n".join(analysis_parts)


class TestConcreteAnalyzers:
    """Test concrete analyzer implementations."""

    def test_mock_analyzer_empty_results(self):
        """Test MockAnalyzer with empty results."""
        analyzer = MockAnalyzer()
        result = analyzer.analyze([])
        assert result == "No results provided"

    def test_mock_analyzer_with_results(self):
        """Test MockAnalyzer with sample results."""
        analyzer = MockAnalyzer()

        test_results = [
            {
                "id": "eval-001",
                "judge_quality.score": 0.85,
                "judge_quality.confidence": 0.9,
                "judge_safety.score": 0.95,
            },
            {
                "id": "eval-002",
                "judge_quality.score": 0.72,
                "judge_quality.confidence": 0.8,
                "judge_safety.score": 0.88,
            },
            {
                "id": "eval-003",
                "judge_quality.score": 0.91,
                "judge_quality.confidence": 0.85,
                "judge_safety.score": 0.92,
            },
        ]

        result = analyzer.analyze(test_results)

        assert "Total results: 3" in result
        assert "judge_quality.score:" in result
        assert "judge_quality.confidence:" in result
        assert "judge_safety.score:" in result

        # Check that averages are calculated
        assert "avg=" in result
        assert "min=" in result
        assert "max=" in result

    def test_mock_analyzer_statistics_accuracy(self):
        """Test that MockAnalyzer calculates statistics correctly."""
        analyzer = MockAnalyzer()

        test_results = [
            {"id": "1", "score": 0.8},
            {"id": "2", "score": 0.6},
            {"id": "3", "score": 1.0},
        ]

        result = analyzer.analyze(test_results)

        # Average should be (0.8 + 0.6 + 1.0) / 3 = 0.8
        assert "score: avg=0.800" in result
        assert "min=0.600" in result
        assert "max=1.000" in result

    def test_count_analyzer(self):
        """Test CountAnalyzer functionality."""
        analyzer = CountAnalyzer()

        # Test with empty results
        result = analyzer.analyze([])
        assert result == "Found 0 evaluation results"

        # Test with results
        test_results = [{"id": f"test-{i}"} for i in range(5)]
        result = analyzer.analyze(test_results)
        assert result == "Found 5 evaluation results"

    def test_error_analyzer_no_errors(self):
        """Test ErrorAnalyzer with no errors."""
        analyzer = ErrorAnalyzer()

        clean_results = [
            {"id": "eval-001", "judge_quality.score": 0.8, "judge_safety.score": 0.9},
            {"id": "eval-002", "judge_quality.score": 0.7, "judge_safety.score": 0.85},
        ]

        result = analyzer.analyze(clean_results)
        assert result == "No errors found in 2 results"

    def test_error_analyzer_with_errors(self):
        """Test ErrorAnalyzer with errors present."""
        analyzer = ErrorAnalyzer()

        results_with_errors = [
            {
                "id": "eval-001",
                "judge_quality.score": 0.8,
                "judge_quality.error": None,  # No error
                "judge_safety.error": "API timeout",
            },
            {
                "id": "eval-002",
                "judge_quality.error": "Invalid response format",
                "judge_safety.score": 0.9,
            },
            {"id": "eval-003", "judge_quality.score": 0.75, "judge_safety.score": 0.82},
        ]

        result = analyzer.analyze(results_with_errors)

        assert "Found 2 errors in 3 results:" in result
        assert "judge_quality: 1 errors" in result
        assert "judge_safety: 1 errors" in result
        assert "API timeout" in result
        assert "Invalid response format" in result

    def test_error_analyzer_many_errors(self):
        """Test ErrorAnalyzer truncation with many errors."""
        analyzer = ErrorAnalyzer()

        # Create results with many errors from same judge
        results_with_many_errors = []
        for i in range(6):
            results_with_many_errors.append(
                {"id": f"eval-{i:03d}", "judge_test.error": f"Error number {i + 1}"}
            )

        result = analyzer.analyze(results_with_many_errors)

        assert "Found 6 errors in 6 results:" in result
        assert "judge_test: 6 errors" in result
        assert "Error number 1" in result
        assert "Error number 2" in result
        assert "Error number 3" in result
        assert "... and 3 more" in result  # Should truncate after 3

    def test_error_analyzer_empty_results(self):
        """Test ErrorAnalyzer with empty results."""
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze([])
        assert result == "No results to analyze for errors"


class TestAnalyzerComposition:
    """Test using multiple analyzers together."""

    def test_multiple_analyzers(self):
        """Test using multiple analyzers on the same data."""
        test_results = [
            {
                "id": "eval-001",
                "judge_quality.score": 0.8,
                "judge_safety.error": "Connection failed",
            },
            {"id": "eval-002", "judge_quality.score": 0.9, "judge_safety.score": 0.95},
        ]

        analyzers = [CountAnalyzer(), MockAnalyzer(), ErrorAnalyzer()]

        results = []
        for analyzer in analyzers:
            analysis = analyzer.analyze(test_results)
            results.append(analysis)

        # Count analyzer result
        assert "Found 2 evaluation results" in results[0]

        # Mock analyzer result
        assert "Total results: 2" in results[1]
        assert "judge_quality.score:" in results[1]

        # Error analyzer result
        assert "Found 1 errors in 2 results:" in results[2]
        assert "Connection failed" in results[2]

    def test_analyzer_inheritance_chain(self):
        """Test that all analyzers properly inherit from base Analyzer."""
        analyzers = [MockAnalyzer(), CountAnalyzer(), ErrorAnalyzer()]

        for analyzer in analyzers:
            assert isinstance(analyzer, Analyzer)
            assert isinstance(analyzer, BaseAnalyzer)
            assert hasattr(analyzer, "analyze")
            assert callable(analyzer.analyze)


class TestAnalyzerEdgeCases:
    """Test edge cases and error conditions for analyzers."""

    def test_analyzer_with_malformed_results(self):
        """Test analyzers handle malformed results gracefully."""
        malformed_results = [
            {"id": "good-001", "score": 0.8},
            {"id": "missing-data"},  # No score
            {"malformed": "no-id"},  # No id
            {},  # Empty dict
            {"id": "mixed-types", "score": "not-a-number"},  # Wrong type
        ]

        # MockAnalyzer should handle mixed types
        mock_analyzer = MockAnalyzer()
        result = mock_analyzer.analyze(malformed_results)
        assert "Total results: 5" in result

        # CountAnalyzer should just count
        count_analyzer = CountAnalyzer()
        result = count_analyzer.analyze(malformed_results)
        assert "Found 5 evaluation results" in result

    def test_analyzer_with_none_values(self):
        """Test analyzers handle None values appropriately."""
        results_with_nones = [
            {"id": "test-001", "score": 0.8, "confidence": None},
            {"id": "test-002", "score": None, "confidence": 0.9},
            {"id": "test-003", "score": 0.7, "confidence": 0.8},
        ]

        mock_analyzer = MockAnalyzer()
        result = mock_analyzer.analyze(results_with_nones)

        # Should still calculate stats for non-None values
        assert "Total results: 3" in result
        # Score stats should be based on 0.8 and 0.7 only
        # Confidence stats should be based on 0.9 and 0.8 only

    def test_analyzer_with_large_dataset(self):
        """Test analyzer performance with larger dataset."""
        # Create a larger dataset
        large_results = []
        for i in range(100):
            large_results.append(
                {
                    "id": f"eval-{i:03d}",
                    "judge_quality.score": 0.5 + (i % 50) * 0.01,
                    "judge_safety.score": 0.7 + (i % 30) * 0.01,
                }
            )

        analyzers = [CountAnalyzer(), MockAnalyzer(), ErrorAnalyzer()]

        for analyzer in analyzers:
            result = analyzer.analyze(large_results)
            # Should complete without error
            assert isinstance(result, str)
            assert len(result) > 0

        # Count analyzer should report correct count
        count_result = CountAnalyzer().analyze(large_results)
        assert "Found 100 evaluation results" in count_result
