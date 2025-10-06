"""Tests for base classes and abstract interfaces."""

import pytest
from abc import ABC

from src.evaluator.base import (
    Source,
    Formatter,
    Judge,
    LLMJudge,
    HeuristicJudge,
    Analyzer,
)


class ConcreteSource(Source):
    """Concrete implementation of Source for testing."""

    def __init__(self, data):
        self.data = data

    def load(self):
        for item in self.data:
            yield item


class ConcreteFormatter(Formatter):
    """Concrete implementation of Formatter for testing."""

    def format(self, data):
        return f"Formatted: {data}"


class ConcreteJudge(Judge):
    """Concrete implementation of Judge for testing."""

    def __init__(self, name="test_judge"):
        self.name = name

    async def judge(self, formatted_input):
        return {"score": 0.8, "input_length": len(formatted_input)}


class ConcreteLLMJudge(LLMJudge):
    """Concrete implementation of LLMJudge for testing."""

    def __init__(self, name="llm_judge"):
        self.name = name
        self.model = "test-model"
        self.temperature = 0.5
        self.system_prompt = "Test system prompt"
        self.user_prompt_template = "Evaluate: {formatted_input}"

    async def judge(self, formatted_input):
        return {"score": 0.75, "model_used": self.model}


class ConcreteHeuristicJudge(HeuristicJudge):
    """Concrete implementation of HeuristicJudge for testing."""

    def __init__(self, name="heuristic_judge"):
        self.name = name

    def evaluate(self, data):
        # Simple heuristic: score based on text length
        text_length = len(str(data.get("text", "")))
        return {"score": min(text_length / 100.0, 1.0), "length": text_length}


class ConcreteAnalyzer(Analyzer):
    """Concrete implementation of Analyzer for testing."""

    def analyze(self, results):
        if not results:
            return "No results to analyze"

        total = len(results)
        scores = [r.get("score", 0) for r in results if "score" in r]
        avg_score = sum(scores) / len(scores) if scores else 0

        return f"Analyzed {total} results, average score: {avg_score:.2f}"


class TestBaseClasses:
    """Test that base classes are properly abstract."""

    def test_source_is_abstract(self):
        """Test that Source cannot be instantiated directly."""
        assert issubclass(Source, ABC)

        with pytest.raises(TypeError):
            Source()

    def test_formatter_is_abstract(self):
        """Test that Formatter cannot be instantiated directly."""
        assert issubclass(Formatter, ABC)

        with pytest.raises(TypeError):
            Formatter()

    def test_judge_is_abstract(self):
        """Test that Judge cannot be instantiated directly."""
        assert issubclass(Judge, ABC)

        with pytest.raises(TypeError):
            Judge()

    def test_llm_judge_is_abstract(self):
        """Test that LLMJudge cannot be instantiated directly."""
        assert issubclass(LLMJudge, ABC)

        with pytest.raises(TypeError):
            LLMJudge()

    def test_heuristic_judge_is_abstract(self):
        """Test that HeuristicJudge cannot be instantiated directly."""
        assert issubclass(HeuristicJudge, ABC)

        with pytest.raises(TypeError):
            HeuristicJudge()

    def test_analyzer_is_abstract(self):
        """Test that Analyzer cannot be instantiated directly."""
        assert issubclass(Analyzer, ABC)

        with pytest.raises(TypeError):
            Analyzer()


class TestConcreteImplementations:
    """Test concrete implementations of abstract base classes."""

    def test_concrete_source(self):
        """Test concrete Source implementation."""
        test_data = [{"id": 1, "text": "test"}, {"id": 2, "text": "data"}]
        source = ConcreteSource(test_data)

        loaded_data = list(source.load())
        assert loaded_data == test_data
        assert len(loaded_data) == 2

    def test_concrete_formatter(self):
        """Test concrete Formatter implementation."""
        formatter = ConcreteFormatter()

        test_data = {"name": "test", "value": 42}
        result = formatter.format(test_data)

        assert result == "Formatted: {'name': 'test', 'value': 42}"

    async def test_concrete_judge(self):
        """Test concrete Judge implementation."""
        judge = ConcreteJudge("test_judge")

        assert judge.name == "test_judge"

        result = await judge.judge("Test input text")

        assert result["score"] == 0.8
        assert result["input_length"] == len("Test input text")

    async def test_concrete_llm_judge(self):
        """Test concrete LLMJudge implementation."""
        judge = ConcreteLLMJudge("llm_test")

        assert judge.name == "llm_test"
        assert judge.model == "test-model"
        assert judge.temperature == 0.5
        assert judge.system_prompt == "Test system prompt"
        assert judge.user_prompt_template == "Evaluate: {formatted_input}"

        result = await judge.judge("Test input")

        assert result["score"] == 0.75
        assert result["model_used"] == "test-model"

    def test_concrete_heuristic_judge_evaluate(self):
        """Test concrete HeuristicJudge evaluate method."""
        judge = ConcreteHeuristicJudge("heuristic_test")

        assert judge.name == "heuristic_test"

        # Test with short text
        short_data = {"text": "short"}
        result = judge.evaluate(short_data)

        assert result["score"] == 0.05  # 5 chars / 100
        assert result["length"] == 5

        # Test with long text
        long_text = "x" * 150
        long_data = {"text": long_text}
        result = judge.evaluate(long_data)

        assert result["score"] == 1.0  # Capped at 1.0
        assert result["length"] == 150

    async def test_heuristic_judge_default_implementation(self):
        """Test that HeuristicJudge default judge method raises NotImplementedError."""
        judge = ConcreteHeuristicJudge()

        with pytest.raises(
            NotImplementedError, match="HeuristicJudge requires custom implementation"
        ):
            await judge.judge("formatted input")

    def test_concrete_analyzer(self):
        """Test concrete Analyzer implementation."""
        analyzer = ConcreteAnalyzer()

        # Test with empty results
        result = analyzer.analyze([])
        assert result == "No results to analyze"

        # Test with results
        test_results = [
            {"id": "1", "score": 0.8},
            {"id": "2", "score": 0.6},
            {"id": "3", "score": 0.9},
            {"id": "4", "other": "data"},  # No score
        ]

        result = analyzer.analyze(test_results)
        assert "Analyzed 4 results" in result
        assert "average score: 0.77" in result  # (0.8 + 0.6 + 0.9) / 3


class TestInheritanceRelationships:
    """Test inheritance relationships between base classes."""

    def test_llm_judge_inherits_from_judge(self):
        """Test that LLMJudge inherits from Judge."""
        assert issubclass(LLMJudge, Judge)

        judge = ConcreteLLMJudge()
        assert isinstance(judge, Judge)
        assert isinstance(judge, LLMJudge)

    def test_heuristic_judge_inherits_from_judge(self):
        """Test that HeuristicJudge inherits from Judge."""
        assert issubclass(HeuristicJudge, Judge)

        judge = ConcreteHeuristicJudge()
        assert isinstance(judge, Judge)
        assert isinstance(judge, HeuristicJudge)

    def test_judge_attributes(self):
        """Test that Judge subclasses have required attributes."""
        llm_judge = ConcreteLLMJudge("llm_test")
        heuristic_judge = ConcreteHeuristicJudge("heuristic_test")
        regular_judge = ConcreteJudge("regular_test")

        # All judges should have name attribute
        assert hasattr(llm_judge, "name")
        assert hasattr(heuristic_judge, "name")
        assert hasattr(regular_judge, "name")

        # LLM judges should have additional attributes
        assert hasattr(llm_judge, "model")
        assert hasattr(llm_judge, "temperature")
        assert hasattr(llm_judge, "system_prompt")
        assert hasattr(llm_judge, "user_prompt_template")


class TestCustomHeuristicJudge:
    """Test custom heuristic judge that implements both methods."""

    class WordCountJudge(HeuristicJudge):
        """Custom heuristic judge that counts words."""

        def __init__(self, name="word_count"):
            self.name = name

        def evaluate(self, data):
            text = str(data.get("text", ""))
            word_count = len(text.split())
            return {
                "word_count": word_count,
                "score": min(word_count / 20.0, 1.0),  # Score based on word count
            }

        async def judge(self, formatted_input):
            # Custom implementation that parses formatted input
            # For this test, assume formatted input contains the data we need
            if "text" in formatted_input:
                # Simple parsing - in reality this would be more sophisticated
                text_start = formatted_input.find("text>") + 5
                text_end = formatted_input.find("</text>")
                if text_start > 4 and text_end > text_start:
                    text = formatted_input[text_start:text_end]
                    return self.evaluate({"text": text})

            # Fallback evaluation
            return self.evaluate({"text": formatted_input})

    def test_custom_heuristic_judge_evaluate(self):
        """Test custom heuristic judge evaluate method."""
        judge = self.WordCountJudge("word_counter")

        # Test with different word counts
        data1 = {"text": "one two three"}
        result1 = judge.evaluate(data1)
        assert result1["word_count"] == 3
        assert result1["score"] == 0.15  # 3/20

        data2 = {"text": "word " * 25}  # 25 words
        result2 = judge.evaluate(data2)
        assert result2["word_count"] == 25
        assert result2["score"] == 1.0  # Capped at 1.0

    async def test_custom_heuristic_judge_judge_method(self):
        """Test custom heuristic judge judge method."""
        judge = self.WordCountJudge("word_counter")

        # Test with XML-like formatted input
        formatted_input = "<data><text>hello world test</text></data>"
        result = await judge.judge(formatted_input)

        assert result["word_count"] == 3
        assert result["score"] == 0.15

        # Test with plain text fallback
        result2 = await judge.judge("plain text input")
        assert "word_count" in result2
        assert "score" in result2


class TestPolymorphism:
    """Test polymorphic behavior of base classes."""

    async def test_judge_polymorphism(self):
        """Test that different Judge implementations can be used polymorphically."""
        judges = [
            ConcreteJudge("regular"),
            ConcreteLLMJudge("llm"),
            ConcreteHeuristicJudge("heuristic"),
        ]

        # All should be callable as judges
        for judge in judges:
            assert hasattr(judge, "judge")
            assert hasattr(judge, "name")

            if isinstance(judge, HeuristicJudge) and not hasattr(
                judge, "_custom_judge_impl"
            ):
                # Default HeuristicJudge raises NotImplementedError
                with pytest.raises(NotImplementedError):
                    await judge.judge("test")
            else:
                result = await judge.judge("test input")
                assert isinstance(result, dict)

    def test_source_polymorphism(self):
        """Test that different Source implementations can be used polymorphically."""
        sources = [
            ConcreteSource([{"test": "data"}]),
            ConcreteSource([]),  # Empty source
        ]

        for source in sources:
            assert hasattr(source, "load")
            data = list(source.load())
            assert isinstance(data, list)

    def test_formatter_polymorphism(self):
        """Test that different Formatter implementations can be used polymorphically."""
        formatters = [ConcreteFormatter()]

        test_data = {"key": "value"}

        for formatter in formatters:
            assert hasattr(formatter, "format")
            result = formatter.format(test_data)
            assert isinstance(result, str)

    def test_analyzer_polymorphism(self):
        """Test that different Analyzer implementations can be used polymorphically."""
        analyzers = [ConcreteAnalyzer()]

        test_results = [{"score": 0.8}, {"score": 0.6}]

        for analyzer in analyzers:
            assert hasattr(analyzer, "analyze")
            result = analyzer.analyze(test_results)
            assert isinstance(result, str)
