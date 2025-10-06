"""Judge implementations for LLM-based evaluation."""

from typing import Any, Type, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel

from .base import LLMJudge


class ChatCompletionLLM(LLMJudge):
    """LLM judge using OpenAI chat completions with structured output."""

    # Class attributes that should be overridden by subclasses
    name: str = "chat_completion"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    prompt: str = "{formatted_input}"
    output_format: Optional[Type[BaseModel]] = None

    def __init_subclass__(cls, **kwargs):
        """Auto-discover OutputFormat nested class and set up default extract_metrics."""
        super().__init_subclass__(**kwargs)

        # Auto-discover OutputFormat nested class
        for (
            attr_name
        ) in cls.__dict__:  # Use __dict__ instead of dir() to avoid special attributes
            if attr_name == "OutputFormat":
                attr = getattr(cls, attr_name)
                if isinstance(attr, type) and hasattr(
                    attr, "model_dump"
                ):  # Check if it's a Pydantic model
                    cls.output_format = attr
                    break

        # Generate default extract_metrics if not explicitly defined in this class
        if "extract_metrics" not in cls.__dict__:

            def extract_metrics(self, response):
                """Default implementation - extracts all fields from the response model."""
                return response.model_dump()

            cls.extract_metrics = extract_metrics

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize with optional OpenAI API key and base URL."""
        # If no parameters provided, let OpenAI client use environment variables
        if api_key is None and base_url is None:
            self.client = AsyncOpenAI()
        else:
            # Use provided parameters
            kwargs = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            self.client = AsyncOpenAI(**kwargs)

    async def judge(self, formatted_input: str) -> dict[str, Any]:
        """Judge the formatted input using OpenAI chat completions."""
        # Format the prompt with the input
        formatted_prompt = self.prompt.format(formatted_input=formatted_input)
        messages = [{"role": "user", "content": formatted_prompt}]

        # Make API call with structured output if output_format is provided
        if self.output_format:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format=self.output_format,
            )

            parsed_response = response.choices[0].message.parsed
            if parsed_response:
                # Extract metrics using the extract_metrics method
                result = self.extract_metrics(parsed_response)
            else:
                # Fallback to raw response
                result = {"response": response.choices[0].message.content}

            # Add token usage metadata if available - with defensive handling for different API formats
            if hasattr(response, "usage") and response.usage:
                try:
                    # Normalize to consistent column names across APIs
                    if hasattr(response.usage, "total_tokens"):
                        result["meta.tokens_total"] = response.usage.total_tokens
                    elif hasattr(response.usage, "total"):
                        result["meta.tokens_total"] = response.usage.total

                    if hasattr(response.usage, "prompt_tokens"):
                        result["meta.tokens_input"] = response.usage.prompt_tokens
                    elif hasattr(response.usage, "input_tokens"):
                        result["meta.tokens_input"] = response.usage.input_tokens

                    if hasattr(response.usage, "completion_tokens"):
                        result["meta.tokens_output"] = response.usage.completion_tokens
                    elif hasattr(response.usage, "output_tokens"):
                        result["meta.tokens_output"] = response.usage.output_tokens

                    # Add cached tokens if available in prompt_tokens_details
                    cached_tokens = 0
                    if (
                        hasattr(response.usage, "prompt_tokens_details")
                        and response.usage.prompt_tokens_details
                    ):
                        if hasattr(
                            response.usage.prompt_tokens_details, "cached_tokens"
                        ):
                            cached_tokens = (
                                response.usage.prompt_tokens_details.cached_tokens
                            )
                    elif hasattr(response.usage, "cached_tokens"):
                        cached_tokens = response.usage.cached_tokens

                    result["meta.tokens_cached"] = cached_tokens

                except AttributeError as e:
                    # Log the issue but don't fail the evaluation
                    print(f"Warning: Could not extract token usage from response: {e}")
                    result["meta.tokens_total"] = 0
                    result["meta.tokens_input"] = 0
                    result["meta.tokens_output"] = 0
                    result["meta.tokens_cached"] = 0

            return result
        else:
            # Standard chat completion without structured output
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )

            result = {"response": response.choices[0].message.content}

            # Add token usage metadata if available - with defensive handling for different API formats
            if hasattr(response, "usage") and response.usage:
                try:
                    # Normalize to consistent column names across APIs
                    if hasattr(response.usage, "total_tokens"):
                        result["meta.tokens_total"] = response.usage.total_tokens
                    elif hasattr(response.usage, "total"):
                        result["meta.tokens_total"] = response.usage.total

                    if hasattr(response.usage, "prompt_tokens"):
                        result["meta.tokens_input"] = response.usage.prompt_tokens
                    elif hasattr(response.usage, "input_tokens"):
                        result["meta.tokens_input"] = response.usage.input_tokens

                    if hasattr(response.usage, "completion_tokens"):
                        result["meta.tokens_output"] = response.usage.completion_tokens
                    elif hasattr(response.usage, "output_tokens"):
                        result["meta.tokens_output"] = response.usage.output_tokens

                    # Add cached tokens if available in prompt_tokens_details
                    cached_tokens = 0
                    if (
                        hasattr(response.usage, "prompt_tokens_details")
                        and response.usage.prompt_tokens_details
                    ):
                        if hasattr(
                            response.usage.prompt_tokens_details, "cached_tokens"
                        ):
                            cached_tokens = (
                                response.usage.prompt_tokens_details.cached_tokens
                            )
                    elif hasattr(response.usage, "cached_tokens"):
                        cached_tokens = response.usage.cached_tokens

                    result["meta.tokens_cached"] = cached_tokens

                except AttributeError as e:
                    # Log the issue but don't fail the evaluation
                    print(f"Warning: Could not extract token usage from response: {e}")
                    result["meta.tokens_total"] = 0
                    result["meta.tokens_input"] = 0
                    result["meta.tokens_output"] = 0
                    result["meta.tokens_cached"] = 0

            return result

    def extract_metrics(self, response: BaseModel) -> dict[str, Any]:
        """Extract metrics from structured response. Override in subclasses."""
        # Default implementation: convert all fields to dict
        return response.model_dump()


class ResponsesLLM(LLMJudge):
    """LLM judge using OpenAI Responses API with reasoning capabilities."""

    # Class attributes that should be overridden by subclasses
    name: str = "responses"
    model: str = "gpt-5"
    reasoning_effort: str = "medium"
    prompt: str = "{formatted_input}"
    output_format: Optional[Type[BaseModel]] = None

    def __init_subclass__(cls, **kwargs):
        """Auto-discover OutputFormat nested class and set up default extract_metrics."""
        super().__init_subclass__(**kwargs)

        # Auto-discover OutputFormat nested class
        for (
            attr_name
        ) in cls.__dict__:  # Use __dict__ instead of dir() to avoid special attributes
            if attr_name == "OutputFormat":
                attr = getattr(cls, attr_name)
                if isinstance(attr, type) and hasattr(
                    attr, "model_dump"
                ):  # Check if it's a Pydantic model
                    cls.output_format = attr
                    break

        # Generate default extract_metrics if not explicitly defined in this class
        if "extract_metrics" not in cls.__dict__:

            def extract_metrics(self, response):
                """Default implementation - extracts all fields from the response model."""
                return response.model_dump()

            cls.extract_metrics = extract_metrics

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize with optional OpenAI API key and base URL."""
        # If no parameters provided, let OpenAI client use environment variables
        if api_key is None and base_url is None:
            self.client = AsyncOpenAI()
        else:
            # Use provided parameters
            kwargs = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            self.client = AsyncOpenAI(**kwargs)

    async def judge(self, formatted_input: str) -> dict[str, Any]:
        """Judge using OpenAI Responses API with reasoning capabilities."""
        # Format the prompt with the input
        formatted_prompt = self.prompt.format(formatted_input=formatted_input)
        input_messages = [{"role": "user", "content": formatted_prompt}]

        # Prepare request parameters
        request_params = {
            "model": self.model,
            "input": input_messages,
        }

        # Only add reasoning effort for models that support it (gpt-5 and newer)
        if self.model.startswith("gpt-5"):
            request_params["reasoning"] = {"effort": self.reasoning_effort}

        # Add structured output if output_format is provided
        if self.output_format:
            # Get the schema and ensure it meets Structured Outputs requirements
            schema = self.output_format.model_json_schema()

            # Ensure additionalProperties is set to False for all objects
            def ensure_no_additional_properties(obj):
                if isinstance(obj, dict):
                    if obj.get("type") == "object":
                        obj["additionalProperties"] = False
                    for key, value in obj.items():
                        ensure_no_additional_properties(value)
                elif isinstance(obj, list):
                    for item in obj:
                        ensure_no_additional_properties(item)

            ensure_no_additional_properties(schema)

            request_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "structured_response",
                    "schema": schema,
                    "strict": True,
                }
            }

        # Make API call
        response = await self.client.responses.create(**request_params)

        # Extract output text - use output_text property if available
        if hasattr(response, "output_text"):
            output_text = response.output_text
        else:
            # Fallback to extracting from output items
            output_text = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text

        # If we have a output_format, parse the output as JSON
        if self.output_format and output_text.strip():
            try:
                import json

                # Handle excessive newlines issue - trim after final }
                cleaned_output = output_text.strip()
                if cleaned_output.endswith("}"):
                    # Find the position of the last } and trim everything after
                    last_brace_pos = cleaned_output.rfind("}")
                    cleaned_output = cleaned_output[: last_brace_pos + 1]

                # Use raw_decode to get only the first valid JSON object
                decoder = json.JSONDecoder()
                parsed_data, _ = decoder.raw_decode(cleaned_output)

                # Create instance of output format
                structured_response = self.output_format(**parsed_data)
                # Extract metrics using the extract_metrics method
                result = self.extract_metrics(structured_response)
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback to raw output if parsing fails
                result = {"response": output_text, "parsing_error": str(e)}
        else:
            # Return raw response
            result = {"response": output_text}

        # Add token usage metadata if available - with defensive handling for different API formats
        if hasattr(response, "usage") and response.usage:
            try:
                # Handle Responses API format (input_tokens, output_tokens, total_tokens)
                if hasattr(response.usage, "total_tokens"):
                    result["meta.tokens_total"] = response.usage.total_tokens
                elif hasattr(response.usage, "input_tokens") and hasattr(
                    response.usage, "output_tokens"
                ):
                    result["meta.tokens_total"] = (
                        response.usage.input_tokens + response.usage.output_tokens
                    )

                if hasattr(response.usage, "input_tokens"):
                    result["meta.tokens_input"] = response.usage.input_tokens
                elif hasattr(response.usage, "prompt_tokens"):
                    result["meta.tokens_input"] = response.usage.prompt_tokens

                if hasattr(response.usage, "output_tokens"):
                    result["meta.tokens_output"] = response.usage.output_tokens
                elif hasattr(response.usage, "completion_tokens"):
                    result["meta.tokens_output"] = response.usage.completion_tokens

                # Add cached tokens - Responses API uses input_tokens_details.cached_tokens
                cached_tokens = 0
                if (
                    hasattr(response.usage, "input_tokens_details")
                    and response.usage.input_tokens_details
                    and hasattr(response.usage.input_tokens_details, "cached_tokens")
                ):
                    cached_tokens = response.usage.input_tokens_details.cached_tokens
                elif (
                    hasattr(response.usage, "prompt_tokens_details")
                    and response.usage.prompt_tokens_details
                    and hasattr(response.usage.prompt_tokens_details, "cached_tokens")
                ):
                    cached_tokens = response.usage.prompt_tokens_details.cached_tokens
                elif hasattr(response.usage, "cached_tokens"):
                    cached_tokens = response.usage.cached_tokens

                result["meta.tokens_cached"] = cached_tokens

            except AttributeError as e:
                # Log the issue but don't fail the evaluation
                print(f"Warning: Could not extract token usage from response: {e}")
                result["meta.tokens_total"] = 0
                result["meta.tokens_prompt"] = 0
                result["meta.tokens_completion"] = 0
                result["meta.tokens_cached"] = 0

        return result

    def extract_metrics(self, response: BaseModel) -> dict[str, Any]:
        """Extract metrics from structured response. Override in subclasses."""
        # Default implementation: convert all fields to dict
        return response.model_dump()
