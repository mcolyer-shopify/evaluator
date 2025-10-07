"""Tests for judge implementations with mocked LLM calls."""

from unittest.mock import Mock, patch
from pydantic import BaseModel

from src.evaluator.judges import ChatCompletionLLM, ResponsesLLM


class MockResponseModel(BaseModel):
    """Test response model for structured output."""

    score: float
    confidence: float
    annotation: str


class TestChatCompletionLLM:
    """Tests for ChatCompletionLLM judge."""

    def test_initialization(self, mock_openai_client):
        """Test judge initialization."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client
            judge = ChatCompletionLLM(api_key="test-key")

            assert judge.name == "chat_completion"
            assert judge.model == "gpt-4o-mini"
            assert judge.temperature == 0.0
            assert judge.client == mock_openai_client
            mock_client_class.assert_called_once_with(api_key="test-key")

    def test_initialization_no_api_key(self, mock_openai_client):
        """Test judge initialization without API key."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client
            ChatCompletionLLM()

            mock_client_class.assert_called_once_with()

    async def test_judge_without_structured_output(self, mock_openai_client):
        """Test judge method without structured output."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            judge = ChatCompletionLLM()
            judge.prompt = "You are a helpful assistant. Evaluate: {formatted_input}"

            result = await judge.judge("Test input text")

            # Verify the client was called correctly
            mock_openai_client.chat.completions.create.assert_called_once()
            call_args = mock_openai_client.chat.completions.create.call_args

            assert call_args[1]["model"] == "gpt-4o-mini"
            assert call_args[1]["temperature"] == 0.0

            messages = call_args[1]["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            expected_content = "You are a helpful assistant. Evaluate: Test input text"
            assert messages[0]["content"] == expected_content

            # Verify result includes response and token metadata
            expected_result = {
                "response": "Test response from OpenAI",
                "meta.tokens_total": 150,
                "meta.tokens_input": 100,
                "meta.tokens_output": 50,
                "meta.tokens_cached": 25,
            }
            assert result == expected_result

    async def test_judge_with_structured_output(self, mock_openai_client):
        """Test judge method with structured output."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            judge = ChatCompletionLLM()
            judge.output_format = MockResponseModel
            judge.prompt = "Rate this content. {formatted_input}"

            result = await judge.judge("Test input")

            # Verify structured completion was called
            mock_openai_client.beta.chat.completions.parse.assert_called_once()
            call_args = mock_openai_client.beta.chat.completions.parse.call_args

            assert call_args[1]["model"] == "gpt-4o-mini"
            assert call_args[1]["response_format"] == MockResponseModel

            # Verify result extraction includes judge results and token metadata
            expected_result = {
                "score": 0.85,
                "confidence": 0.9,
                "annotation": "Well structured response",
                "meta.tokens_total": 150,
                "meta.tokens_input": 100,
                "meta.tokens_output": 50,
                "meta.tokens_cached": 25,
            }
            assert result == expected_result

    async def test_judge_without_system_prompt(self, mock_openai_client):
        """Test judge method without system prompt."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            judge = ChatCompletionLLM()
            judge.prompt = "{formatted_input}"  # Just the input, no system prompt

            await judge.judge("Test input")

            call_args = mock_openai_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]

            # Should only have user message
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    async def test_judge_structured_output_fallback(self, mock_openai_client):
        """Test structured output fallback when parsing fails."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            # Mock usage for the fallback response
            mock_usage = Mock()
            mock_usage.total_tokens = 75
            mock_usage.prompt_tokens = 50
            mock_usage.completion_tokens = 25
            mock_prompt_tokens_details = Mock()
            mock_prompt_tokens_details.cached_tokens = 10
            mock_usage.prompt_tokens_details = mock_prompt_tokens_details

            # Mock failed parsing (parsed is None)
            mock_structured_response = Mock()
            mock_structured_choice = Mock()
            mock_structured_message = Mock()
            mock_structured_message.parsed = None
            mock_structured_message.content = "Fallback response"
            mock_structured_choice.message = mock_structured_message
            mock_structured_response.choices = [mock_structured_choice]
            mock_structured_response.usage = mock_usage

            mock_openai_client.beta.chat.completions.parse.return_value = (
                mock_structured_response
            )

            judge = ChatCompletionLLM()
            judge.output_format = MockResponseModel

            result = await judge.judge("Test input")

            # Should fall back to raw response with token metadata
            expected_result = {
                "response": "Fallback response",
                "meta.tokens_total": 75,
                "meta.tokens_input": 50,
                "meta.tokens_output": 25,
                "meta.tokens_cached": 10,
            }
            assert result == expected_result

    def test_extract_metrics_default(self, mock_openai_client):
        """Test default extract_metrics implementation."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client
            judge = ChatCompletionLLM(api_key="test-key")

            # Mock response model
            mock_response = Mock()
            mock_response.model_dump.return_value = {"score": 0.7, "confidence": 0.8}

            result = judge.extract_metrics(mock_response)

            assert result == {"score": 0.7, "confidence": 0.8}
            mock_response.model_dump.assert_called_once()

    def test_custom_judge_attributes(self, mock_openai_client):
        """Test that judge attributes can be customized."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            judge = ChatCompletionLLM()
            judge.name = "custom_judge"
            judge.model = "gpt-4"
            judge.temperature = 0.5

            assert judge.name == "custom_judge"
            assert judge.model == "gpt-4"
            assert judge.temperature == 0.5


class TestResponsesLLM:
    """Tests for ResponsesLLM judge."""

    def test_initialization(self, mock_openai_client):
        """Test judge initialization."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client
            judge = ResponsesLLM(api_key="test-key")

            assert judge.name == "responses"
            assert judge.model == "gpt-5"
            assert judge.reasoning_effort == "medium"
            assert judge.client == mock_openai_client
            mock_client_class.assert_called_once_with(api_key="test-key")

    def test_initialization_no_api_key(self, mock_openai_client):
        """Test judge initialization without API key."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client
            ResponsesLLM()

            mock_client_class.assert_called_once_with()

    async def test_judge_without_structured_output(self, mock_openai_client):
        """Test judge method without structured output."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            # Mock the responses.create call
            mock_response = Mock()
            mock_response.output_text = "This is the judge response"
            # Add mock usage data (Responses API format)
            mock_usage = Mock()
            mock_usage.total_tokens = 120
            mock_usage.input_tokens = 80
            mock_usage.output_tokens = 40
            mock_input_tokens_details = Mock()
            mock_input_tokens_details.cached_tokens = 15
            mock_usage.input_tokens_details = mock_input_tokens_details
            mock_response.usage = mock_usage
            mock_openai_client.responses.create.return_value = mock_response

            judge = ResponsesLLM()
            judge.prompt = "You are a helpful assistant. Evaluate: {formatted_input}"

            result = await judge.judge("Test input text")

            # Verify responses.create was called with correct parameters
            mock_openai_client.responses.create.assert_called_once()
            call_args = mock_openai_client.responses.create.call_args

            assert call_args[1]["model"] == "gpt-5"
            assert call_args[1]["reasoning"]["effort"] == "medium"
            assert len(call_args[1]["input"]) == 1
            assert call_args[1]["input"][0]["role"] == "user"
            expected_content = "You are a helpful assistant. Evaluate: Test input text"
            assert call_args[1]["input"][0]["content"] == expected_content

            # Verify result includes response and token metadata
            expected_result = {
                "response": "This is the judge response",
                "meta.tokens_total": 120,
                "meta.tokens_input": 80,
                "meta.tokens_output": 40,
                "meta.tokens_cached": 15,
            }
            assert result == expected_result

    async def test_judge_with_structured_output(self, mock_openai_client):
        """Test judge method with structured output."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            # Mock the responses.create call with JSON output
            mock_response = Mock()
            mock_response.output_text = '{"score": 0.85, "confidence": 0.9, "annotation": "Well structured response"}'
            # Add mock usage data
            mock_usage = Mock()
            mock_usage.total_tokens = 180
            mock_usage.input_tokens = 110
            mock_usage.output_tokens = 70
            mock_input_tokens_details = Mock()
            mock_input_tokens_details.cached_tokens = 30
            mock_usage.input_tokens_details = mock_input_tokens_details
            mock_response.usage = mock_usage
            mock_openai_client.responses.create.return_value = mock_response

            judge = ResponsesLLM()
            judge.output_format = MockResponseModel
            judge.prompt = "You are a helpful assistant. {formatted_input}"

            result = await judge.judge("Test input")

            # Verify responses.create was called with structured output
            mock_openai_client.responses.create.assert_called_once()
            call_args = mock_openai_client.responses.create.call_args

            assert call_args[1]["model"] == "gpt-5"
            assert "text" in call_args[1]
            assert "format" in call_args[1]["text"]
            assert call_args[1]["text"]["format"]["type"] == "json_schema"
            assert call_args[1]["text"]["format"]["strict"] is True
            assert call_args[1]["text"]["format"]["name"] == "structured_response"

            # Verify result extraction includes judge results and token metadata
            expected_result = {
                "score": 0.85,
                "confidence": 0.9,
                "annotation": "Well structured response",
                "meta.tokens_total": 180,
                "meta.tokens_input": 110,
                "meta.tokens_output": 70,
                "meta.tokens_cached": 30,
            }
            assert result == expected_result

    async def test_judge_without_system_prompt(self, mock_openai_client):
        """Test judge method without system prompt."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            # Mock the responses.create call
            mock_response = Mock()
            mock_response.output_text = "Response without system prompt"
            mock_openai_client.responses.create.return_value = mock_response

            judge = ResponsesLLM()
            judge.prompt = "{formatted_input}"  # Just the input, no system prompt

            await judge.judge("Test input")

            call_args = mock_openai_client.responses.create.call_args
            input_messages = call_args[1]["input"]

            # Should only have user message
            assert len(input_messages) == 1
            assert input_messages[0]["role"] == "user"

    async def test_judge_structured_output_parsing_error(self, mock_openai_client):
        """Test structured output fallback when JSON parsing fails."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            # Mock the responses.create call with invalid JSON
            mock_response = Mock()
            mock_response.output_text = "Invalid JSON response"
            mock_openai_client.responses.create.return_value = mock_response

            judge = ResponsesLLM()
            judge.output_format = MockResponseModel

            result = await judge.judge("Test input")

            # Should fallback to raw response with parsing error
            assert result["response"] == "Invalid JSON response"
            assert "parsing_error" in result

    async def test_judge_custom_reasoning_effort(self, mock_openai_client):
        """Test judge with custom reasoning effort."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            # Mock the responses.create call
            mock_response = Mock()
            mock_response.output_text = "Response with high effort"
            mock_openai_client.responses.create.return_value = mock_response

            judge = ResponsesLLM()
            judge.reasoning_effort = "high"

            await judge.judge("Test input")

            call_args = mock_openai_client.responses.create.call_args
            assert call_args[1]["reasoning"]["effort"] == "high"

    def test_extract_metrics_default(self):
        """Test default extract_metrics implementation."""
        judge = ResponsesLLM(api_key="test-key")

        mock_response = MockResponseModel(
            score=0.7, confidence=0.8, annotation="Test annotation"
        )

        result = judge.extract_metrics(mock_response)

        assert result == {
            "score": 0.7,
            "confidence": 0.8,
            "annotation": "Test annotation",
        }

    async def test_judge_excessive_newlines_handling(self, mock_openai_client):
        """Test handling of excessive newlines in structured output."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            # Mock response with excessive newlines after JSON
            json_with_newlines = (
                '{"score": 0.75, "confidence": 0.85, "annotation": "Test with newlines"}'
                + "\n" * 1000
            )
            mock_response = Mock()
            mock_response.output_text = json_with_newlines
            # Add mock usage data
            mock_usage = Mock()
            mock_usage.total_tokens = 160
            mock_usage.input_tokens = 90
            mock_usage.output_tokens = 70
            mock_input_tokens_details = Mock()
            mock_input_tokens_details.cached_tokens = 20
            mock_usage.input_tokens_details = mock_input_tokens_details
            mock_response.usage = mock_usage
            mock_openai_client.responses.create.return_value = mock_response

            judge = ResponsesLLM()
            judge.output_format = MockResponseModel

            result = await judge.judge("Test input")

            # Should successfully parse despite excessive newlines, including token metadata
            expected_result = {
                "score": 0.75,
                "confidence": 0.85,
                "annotation": "Test with newlines",
                "meta.tokens_total": 160,
                "meta.tokens_input": 90,
                "meta.tokens_output": 70,
                "meta.tokens_cached": 20,
            }
            assert result == expected_result


class TestCustomJudgeImplementation:
    """Test custom judge implementation using ChatCompletionLLM as base."""

    class CustomJudge(ChatCompletionLLM):
        """Custom judge for testing inheritance."""

        name = "custom_test"
        prompt = "You are a strict evaluator. Rate this on a scale of 1-10: {formatted_input}"

        class OutputFormat(BaseModel):
            """Test output format with auto-discovery."""

            score: float
            confidence: float

        def extract_metrics(self, response):
            """Custom metrics extraction."""
            base_metrics = super().extract_metrics(response)
            base_metrics["custom_metric"] = base_metrics.get("score", 0) * 2
            return base_metrics

    async def test_custom_judge(self, mock_openai_client):
        """Test custom judge implementation."""
        with patch("src.evaluator.judges.AsyncOpenAI") as mock_client_class:
            mock_client_class.return_value = mock_openai_client

            judge = self.CustomJudge()
            judge.output_format = MockResponseModel

            result = await judge.judge("Test content")

            # Verify custom attributes
            call_args = mock_openai_client.beta.chat.completions.parse.call_args
            messages = call_args[1]["messages"]

            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            expected_content = (
                "You are a strict evaluator. Rate this on a scale of 1-10: Test content"
            )
            assert messages[0]["content"] == expected_content

            # Verify custom metrics extraction
            assert result["custom_metric"] == 0.85 * 2  # score * 2

            # Note: Auto-discovery worked initially, but we override it in this test with MockResponseModel
            assert judge.output_format is MockResponseModel

    def test_auto_discovery_mechanism(self):
        """Test that OutputFormat is auto-discovered and extract_metrics is generated."""

        class TestAutoDiscoveryJudge(ChatCompletionLLM):
            name = "test_auto"

            class OutputFormat(BaseModel):
                test_field: str
                number_field: int

        # Verify auto-discovery worked
        assert (
            TestAutoDiscoveryJudge.output_format is TestAutoDiscoveryJudge.OutputFormat
        )

        # Verify default extract_metrics was generated
        judge = TestAutoDiscoveryJudge(api_key="test-key")
        mock_response = TestAutoDiscoveryJudge.OutputFormat(
            test_field="hello", number_field=42
        )

        result = judge.extract_metrics(mock_response)
        expected = {"test_field": "hello", "number_field": 42}
        assert result == expected
