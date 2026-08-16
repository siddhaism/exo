import asyncio
from collections.abc import AsyncGenerator

from exo.api.adapters.ollama import (
    collect_ollama_chat_response,
    ollama_generate_request_to_text_generation,
    ollama_request_to_text_generation,
)
from exo.api.types.ollama_api import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaMessage,
)
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId


async def _chunks(
    values: list[ErrorChunk],
) -> AsyncGenerator[ErrorChunk, None]:
    for value in values:
        yield value


async def _collect_nonstreaming_response(
    values: list[ErrorChunk],
) -> OllamaChatResponse:
    responses = [
        response
        async for response in collect_ollama_chat_response(
            CommandId("test-command"),
            "test/model",
            _chunks(values),
        )
    ]
    assert len(responses) == 1
    return OllamaChatResponse.model_validate_json(responses[0])


def test_chat_adapter_preserves_json_schema() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    request = OllamaChatRequest(
        model=ModelId("test/model"),
        messages=[OllamaMessage(role="user", content="Return a value")],
        format=schema,
    )

    task = ollama_request_to_text_generation(request)

    assert task.response_format == schema


def test_generate_adapter_preserves_json_mode() -> None:
    request = OllamaGenerateRequest(
        model=ModelId("test/model"),
        prompt="Return JSON",
        format="json",
    )

    task = ollama_generate_request_to_text_generation(request)

    assert task.response_format == "json"


def test_nonstreaming_chat_returns_valid_error_for_empty_stream() -> None:
    response = asyncio.run(_collect_nonstreaming_response([]))

    assert response.model == "test/model"
    assert response.done is True
    assert response.done_reason == "error"
    assert response.message.content is not None
    assert "before the runner produced" in response.message.content


def test_nonstreaming_chat_serializes_error_chunk() -> None:
    response = asyncio.run(
        _collect_nonstreaming_response(
            [
                ErrorChunk(
                    model=ModelId("test/model"),
                    error_message="runner creation failed",
                )
            ]
        )
    )

    assert response.done_reason == "error"
    assert response.message.content is not None
    assert "runner creation failed" in response.message.content
