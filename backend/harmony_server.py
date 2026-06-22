from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from jsonschema import ValidationError, validate
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)
from pydantic import BaseModel, Field


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[ToolDefinition] = Field(default_factory=list)
    stream: bool = False


@dataclass(frozen=True)
class Generation:
    tokens: list[int]
    stop_token: int | None


class TokenGenerator(Protocol):
    def generate(
        self,
        prompt_tokens: Sequence[int],
        stop_tokens: Sequence[int],
        max_tokens: int,
    ) -> Generation: ...


class HarmonyServerError(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class MlxGenerator:
    """The only boundary that imports MLX and loads model weights."""

    def __init__(self, model_id: str) -> None:
        if platform.machine() != "arm64":
            raise RuntimeError(
                "GPT-OSS MLX serving requires an arm64 Python interpreter."
            )

        import mlx.core as mx
        from mlx_lm import load

        if not mx.metal.is_available():
            raise RuntimeError("GPT-OSS MLX serving requires Apple Metal.")
        self._mx = mx
        self._model, _tokenizer = load(model_id)

    def generate(
        self,
        prompt_tokens: Sequence[int],
        stop_tokens: Sequence[int],
        max_tokens: int,
    ) -> Generation:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        stop = set(stop_tokens)
        generated: list[int] = []
        prompt = self._mx.array(list(prompt_tokens))
        for token, _logprobs in generate_step(
            prompt,
            self._model,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            token_id = int(token)
            if token_id in stop:
                return Generation(generated, token_id)
            generated.append(token_id)
        return Generation(generated, None)


def openai_messages_to_harmony(
    messages: Sequence[ChatMessage],
    tools: Sequence[ToolDefinition],
    reasoning_effort: ReasoningEffort,
) -> Conversation:
    system = SystemContent.new().with_reasoning_effort(reasoning_effort)
    result = [Message.from_role_and_content(Role.SYSTEM, system)]

    instructions = [
        message.content
        for message in messages
        if message.role in {"system", "developer"} and message.content
    ]
    developer = DeveloperContent.new()
    if instructions:
        developer.with_instructions("\n\n".join(instructions))
    if tools:
        developer.with_function_tools(
            [
                ToolDescription.new(
                    tool.function.name,
                    tool.function.description,
                    tool.function.parameters,
                )
                for tool in tools
            ]
        )
    result.append(Message.from_role_and_content(Role.DEVELOPER, developer))

    call_names: dict[str, str] = {}
    for message in messages:
        if message.role in {"system", "developer"}:
            continue
        if message.role == "user":
            result.append(
                Message.from_role_and_content(Role.USER, message.content or "")
            )
            continue
        if message.role == "assistant":
            if message.tool_calls:
                if message.content:
                    result.append(
                        Message.from_role_and_content(
                            Role.ASSISTANT, message.content
                        ).with_channel("commentary")
                    )
                for call in message.tool_calls:
                    call_names[call.id] = call.function.name
                    result.append(
                        Message.from_role_and_content(
                            Role.ASSISTANT,
                            call.function.arguments,
                        )
                        .with_channel("commentary")
                        .with_recipient(f"functions.{call.function.name}")
                        .with_content_type("<|constrain|>json")
                    )
            elif message.content is not None:
                result.append(
                    Message.from_role_and_content(
                        Role.ASSISTANT, message.content
                    ).with_channel("final")
                )
            continue

        if not message.tool_call_id or message.tool_call_id not in call_names:
            raise HarmonyServerError(
                "Tool result does not match a prior tool call.",
                "invalid_request",
            )
        author = f"functions.{call_names[message.tool_call_id]}"
        result.append(
            Message.from_author_and_content(
                Author.new(Role.TOOL, author),
                message.content or "",
            )
            .with_channel("commentary")
            .with_recipient("assistant")
        )

    return Conversation.from_messages(result)


def _message_text(message: Message) -> str:
    if not message.content or any(
        not isinstance(item, TextContent) for item in message.content
    ):
        raise HarmonyServerError(
            "Harmony returned invalid message content.",
            "invalid_harmony",
            502,
        )
    return "".join(item.text for item in message.content)


def parse_harmony_completion(
    encoding: HarmonyEncoding,
    generation: Generation,
    tools: Sequence[ToolDefinition],
) -> tuple[dict[str, Any], str]:
    try:
        messages = encoding.parse_messages_from_completion_tokens(
            generation.tokens,
            Role.ASSISTANT,
            strict=True,
        )
    except Exception as error:
        raise HarmonyServerError(
            f"Malformed Harmony output: {error}",
            "invalid_harmony",
            502,
        ) from error

    return_token = encoding.encode("<|return|>", allowed_special="all")[0]
    call_token = encoding.encode("<|call|>", allowed_special="all")[0]
    tool_schemas = {tool.function.name: tool.function.parameters for tool in tools}

    if generation.stop_token == call_token:
        calls = [message for message in messages if message.recipient is not None]
        if len(calls) != 1:
            raise HarmonyServerError(
                "Harmony tool action did not contain exactly one recipient.",
                "invalid_harmony",
                502,
            )
        action = calls[0]
        prefix = "functions."
        if action.channel != "commentary" or not action.recipient.startswith(prefix):
            raise HarmonyServerError(
                "Harmony returned a malformed tool recipient.", "invalid_harmony", 502
            )
        name = action.recipient.removeprefix(prefix)
        if name not in tool_schemas:
            raise HarmonyServerError(
                f"Harmony requested unknown tool {name!r}.", "invalid_harmony", 502
            )
        raw_arguments = _message_text(action)
        try:
            arguments = json.loads(raw_arguments)
            validate(arguments, tool_schemas[name])
        except (json.JSONDecodeError, ValidationError, TypeError) as error:
            raise HarmonyServerError(
                f"Harmony supplied invalid arguments for {name}.",
                "invalid_harmony",
                502,
            ) from error
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, separators=(",", ":")),
                        },
                    }
                ],
            },
            "tool_calls",
        )

    if generation.stop_token == return_token:
        finals = [message for message in messages if message.channel == "final"]
        if not finals:
            raise HarmonyServerError(
                "Harmony return action omitted a final message.", "invalid_harmony", 502
            )
        return {
            "role": "assistant",
            "content": "".join(_message_text(message) for message in finals),
        }, "stop"

    raise HarmonyServerError(
        "Harmony completion ended without a final action.",
        "invalid_harmony",
        502,
    )


def error_payload(error: HarmonyServerError) -> dict[str, Any]:
    return {
        "error": {
            "message": str(error),
            "type": "server_error"
            if error.status_code >= 500
            else "invalid_request_error",
            "code": error.code,
        }
    }


def create_app(
    model_id: str,
    max_tokens: int = 512,
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    generator: TokenGenerator | None = None,
) -> FastAPI:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.generator = generator or await asyncio.to_thread(
            MlxGenerator, model_id
        )
        app.state.generation_lock = asyncio.Lock()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(HarmonyServerError)
    async def harmony_error(
        _request: Request, error: HarmonyServerError
    ) -> JSONResponse:
        return JSONResponse(error_payload(error), status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        error = HarmonyServerError(
            "Invalid chat completion request.", "invalid_request"
        )
        return JSONResponse(error_payload(error), status_code=400)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "local",
                    "fraude_protocol": "harmony",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completion(
        body: ChatCompletionRequest, request: Request
    ) -> dict[str, Any]:
        if body.model != model_id:
            raise HarmonyServerError(
                f"Model {body.model!r} is not loaded.", "model_not_found", 404
            )
        if body.stream:
            raise HarmonyServerError(
                "Streaming is not supported by this server.", "unsupported_streaming"
            )

        conversation = openai_messages_to_harmony(
            body.messages, body.tools, reasoning_effort
        )
        prompt_tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        async with request.app.state.generation_lock:
            generated = await asyncio.to_thread(
                request.app.state.generator.generate,
                prompt_tokens,
                encoding.stop_tokens_for_assistant_actions(),
                max_tokens,
            )
        message, finish_reason = parse_harmony_completion(
            encoding, generated, body.tools
        )
        completion_tokens = len(generated.tokens) + (generated.stop_token is not None)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_tokens) + completion_tokens,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve GPT-OSS through MLX and Harmony."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="medium",
    )
    args = parser.parse_args()
    app = create_app(
        args.model,
        args.max_tokens,
        ReasoningEffort(args.reasoning_effort.capitalize()),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
