"""Shared fixtures and configuration for tests."""

import asyncio
from unittest.mock import AsyncMock, Mock
import pytest
import pandas as pd
import yaml
from pydantic import BaseModel

from src.evaluator.sources import CSVSource, YAMLSource
from src.evaluator.formatters import XMLFormatter
from src.evaluator.storage import ResultsStorage


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        {"id": 1, "text": "This is a test", "category": "A"},
        {"id": 2, "text": "Another test case", "category": "B"},
        {"id": 3, "text": "Third test item", "category": "A"},
    ]


@pytest.fixture
def sample_csv_file(sample_data, tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def sample_yaml_file(sample_data, tmp_path):
    """Create a temporary YAML file with sample data."""
    yaml_file = tmp_path / "test_data.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(sample_data, f)
    return str(yaml_file)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_results.duckdb")


@pytest.fixture
def csv_source(sample_csv_file):
    """Create a CSVSource instance."""
    return CSVSource(sample_csv_file)


@pytest.fixture
def yaml_source(sample_yaml_file):
    """Create a YAMLSource instance."""
    return YAMLSource(sample_yaml_file)


@pytest.fixture
def xml_formatter():
    """Create an XMLFormatter instance."""
    return XMLFormatter()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client with realistic responses."""
    client = AsyncMock()

    # Mock usage data
    mock_usage = Mock()
    mock_usage.total_tokens = 150
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50

    # Mock prompt_tokens_details with cached_tokens
    mock_prompt_tokens_details = Mock()
    mock_prompt_tokens_details.cached_tokens = 25
    mock_usage.prompt_tokens_details = mock_prompt_tokens_details

    # Mock chat completion response
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Test response from OpenAI"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    # Set up mock for regular chat completions
    client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Mock structured output response
    mock_structured_response = Mock()
    mock_structured_choice = Mock()
    mock_structured_message = Mock()
    mock_parsed_response = Mock()
    mock_parsed_response.model_dump.return_value = {
        "score": 0.85,
        "confidence": 0.9,
        "annotation": "Well structured response",
    }
    mock_structured_message.parsed = mock_parsed_response
    mock_structured_choice.message = mock_structured_message
    mock_structured_response.choices = [mock_structured_choice]
    mock_structured_response.usage = mock_usage

    # Set up mock for structured completions
    client.beta.chat.completions.parse = AsyncMock(
        return_value=mock_structured_response
    )

    return client


class MockResponseModel(BaseModel):
    """Mock Pydantic model for structured responses."""

    score: float
    confidence: float
    annotation: str


@pytest.fixture
def mock_response_model():
    """Mock response model for structured output testing."""
    return MockResponseModel


@pytest.fixture
def sample_judge_results():
    """Sample judge results for testing."""
    return {
        "score": 0.75,
        "confidence": 0.8,
        "annotation": "Good quality response",
        "raw_response": "This is a test judge response",
    }


@pytest.fixture
def results_storage(temp_db_path):
    """Create a ResultsStorage instance with temporary database."""
    return ResultsStorage(temp_db_path, "test_evaluations")


@pytest.fixture
def sample_results():
    """Sample evaluation results for testing."""
    return [
        {
            "id": "test-1",
            "judge_test.score": 0.8,
            "judge_test.confidence": 0.9,
            "judge_test.annotation": "Good result",
        },
        {
            "id": "test-2",
            "judge_test.score": 0.6,
            "judge_test.confidence": 0.7,
            "judge_test.annotation": "Average result",
        },
        {
            "id": "test-3",
            "judge_test.score": 0.9,
            "judge_test.confidence": 0.95,
            "judge_test.annotation": "Excellent result",
        },
    ]


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup happens automatically with tmp_path fixture
