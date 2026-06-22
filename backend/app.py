import asyncio
import json
import os
import re
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from jsonschema import ValidationError, validate
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel


DEFAULT_MODEL_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
DEFAULT_CONTEXT_WINDOW = 32_768
MAX_TOOL_ROUNDS = 5
SSE_CHUNK_SIZE = 256
SYSTEM_MESSAGE = (
    "You are a coding assistant with filesystem tools. Use tools when needed. "
    "write_file creates new files only and cannot modify existing files. "
    "create_plan saves a draft plan when the user explicitly asks to create or save one. "
    "get_plan retrieves a saved plan by its UUID. "
    "Never write a file unless the current user message explicitly asks you to. "
    "Do not invent example files during greetings or ordinary conversation."
)

_RAW_CALL = re.compile(
    r"^(.*?)\s*(?:<tool_call>\s*)?<function=([^>\s]+)>"
    r"(.*?)</function>\s*(?:</tool_call>)?\s*$",
    re.DOTALL,
)
_RAW_PARAMETER = re.compile(
    r"<parameter=([^>\s]+)>(.*?)</parameter>",
    re.DOTALL,
)


class GatewayError(Exception):
    def __init__(self, message: str, code: str = "gateway_error") -> None:
        super().__init__(message)
        self.code = code


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    stream_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentResult:
    text: str
    usage: dict[str, int]


class AgentGateway:
    def __init__(
        self,
        model_client: httpx.AsyncClient,
        mcp_session: ClientSession,
        tools: list[Any],
        model_url: str,
        context_window: int,
        preferred_model: str | None = None,
    ) -> None:
        self.model_client = model_client
        self.mcp_session = mcp_session
        self.model_url = model_url.rstrip("/")
        self.context_window = context_window
        self.preferred_model = preferred_model
        self.tool_definitions = [openai_tool(tool) for tool in tools]
        self.tool_schemas = {
            tool.name: tool.inputSchema
            for tool in tools
        }
        self.turn_lock = asyncio.Lock()

    async def list_models(self) -> dict[str, Any]:
        try:
            response = await self.model_client.get(f"{self.model_url}/v1/models")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise GatewayError(
                f"Model server is unavailable: {error}",
                "model_unavailable",
            ) from error

        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise GatewayError("Model server returned an invalid model list.", "invalid_response")
        if (
            self.preferred_model
            and "gpt-oss" in self.preferred_model.casefold()
            and response.headers.get("server", "").casefold().startswith("basehttp/")
        ):
            raise GatewayError(
                "GPT-OSS requires backend/harmony_server.py; "
                "stock mlx_lm.server does not apply the Harmony protocol.",
                "incompatible_model_server",
            )

        data = []
        for item in payload["data"]:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                if self.preferred_model and item["id"] != self.preferred_model:
                    continue
                model = dict(item)
                model["context_window"] = self.context_window
                data.append(model)
        if not data:
            message = (
                f"Configured model {self.preferred_model!r} is unavailable."
                if self.preferred_model
                else "Model server returned no usable models."
            )
            raise GatewayError(message, "model_unavailable")
        return {"object": "list", "data": data}

    async def run_agent(
        self,
        model: str,
        request_messages: list[ChatMessage],
        is_cancelled: Callable[[], Awaitable[bool]],
    ) -> AgentResult:
        if self.preferred_model and model != self.preferred_model:
            raise GatewayError(
                f"Model {model!r} is not configured for this gateway.",
                "model_unavailable",
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            *(message.model_dump() for message in request_messages),
        ]

        for _ in range(MAX_TOOL_ROUNDS):
            if await is_cancelled():
                raise asyncio.CancelledError

            payload = await self._call_model(model, messages)
            message = read_assistant_message(payload)
            usage = read_usage(payload)
            calls = normalize_tool_calls(message, self.tool_schemas)

            if not calls:
                content = message.get("content")
                if not isinstance(content, str):
                    raise GatewayError(
                        "Model server returned neither text nor a tool call.",
                        "invalid_response",
                    )
                if "<function=" in content or "<tool_call>" in content:
                    raise GatewayError(
                        "Qwen returned malformed tool-call markup.",
                        "invalid_tool_call",
                    )
                if "<|channel|>" in content or "<|message|>" in content:
                    raise GatewayError(
                        "Model server returned raw Harmony markup. Start "
                        "backend/harmony_server.py instead of mlx_lm.server for GPT-OSS.",
                        "incompatible_model_server",
                    )
                if is_raw_json_tool_call(content):
                    raise GatewayError(
                        "Qwen returned malformed JSON tool-call output.",
                        "invalid_tool_call",
                    )
                if usage is None:
                    raise GatewayError(
                        "Model server omitted token usage from the final response.",
                        "invalid_response",
                    )
                return AgentResult(content, usage)

            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": calls,
                }
            )
            for call in calls:
                if await is_cancelled():
                    raise asyncio.CancelledError
                function = call["function"]
                arguments = json.loads(function["arguments"])
                try:
                    result = await self.mcp_session.call_tool(
                        function["name"],
                        arguments,
                    )
                except Exception as error:
                    raise GatewayError(
                        f"MCP tool {function['name']} failed: {error}",
                        "tool_error",
                    ) from error
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_result_text(result),
                    }
                )

        raise GatewayError(
            f"Model server exceeded the {MAX_TOOL_ROUNDS}-round tool limit.",
            "tool_round_limit",
        )

    async def _call_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            response = await self.model_client.post(
                f"{self.model_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "tools": self.tool_definitions,
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise GatewayError(
                f"Model server generation failed: {error}",
                "generation_failed",
            ) from error
        if not isinstance(payload, dict):
            raise GatewayError("Model server returned invalid JSON.", "invalid_response")
        return payload


def openai_tool(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def read_assistant_message(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GatewayError("Qwen returned no completion choice.", "invalid_response")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise GatewayError("Qwen returned an invalid completion choice.", "invalid_response")
    return choice["message"]


def read_usage(payload: dict[str, Any]) -> dict[str, int] | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(not isinstance(usage.get(key), int) for key in keys):
        return None
    return {key: usage[key] for key in keys}


def normalize_tool_calls(
    message: dict[str, Any],
    tool_schemas: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    structured = message.get("tool_calls")
    if structured:
        if not isinstance(structured, list):
            raise GatewayError("Qwen returned invalid tool calls.", "invalid_tool_call")
        return [normalize_structured_call(call, tool_schemas) for call in structured]

    content = message.get("content")
    if not isinstance(content, str) or "<function=" not in content:
        return []
    return [parse_qwen_tool_call(content, tool_schemas)]


def normalize_structured_call(
    call: Any,
    tool_schemas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        raise GatewayError("Qwen returned an invalid tool call.", "invalid_tool_call")
    function = call["function"]
    name = function.get("name")
    raw_arguments = function.get("arguments")
    if not isinstance(name, str) or name not in tool_schemas:
        raise GatewayError(f"Qwen requested unknown tool {name!r}.", "invalid_tool_call")
    try:
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else raw_arguments
        )
        validate(arguments, tool_schemas[name])
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        raise GatewayError(
            f"Qwen supplied invalid arguments for {name}.",
            "invalid_tool_call",
        ) from error
    return make_tool_call(name, arguments, call.get("id"))


def parse_qwen_tool_call(
    content: str,
    tool_schemas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    match = _RAW_CALL.fullmatch(content)
    if not match:
        raise GatewayError("Qwen returned malformed tool-call markup.", "invalid_tool_call")
    prefix, name, parameter_block = match.groups()
    if "<function=" in prefix or "<tool_call>" in prefix:
        raise GatewayError("Qwen returned multiple tool calls.", "invalid_tool_call")
    if name not in tool_schemas:
        raise GatewayError(f"Qwen requested unknown tool {name!r}.", "invalid_tool_call")

    arguments: dict[str, str] = {}
    matches = list(_RAW_PARAMETER.finditer(parameter_block))
    remainder = _RAW_PARAMETER.sub("", parameter_block)
    if remainder.strip():
        raise GatewayError("Qwen returned malformed tool arguments.", "invalid_tool_call")
    for parameter in matches:
        key, value = parameter.groups()
        if key in arguments:
            raise GatewayError(
                f"Qwen repeated tool argument {key!r}.",
                "invalid_tool_call",
            )
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        arguments[key] = value
    try:
        validate(arguments, tool_schemas[name])
    except ValidationError as error:
        raise GatewayError(
            f"Qwen supplied invalid arguments for {name}.",
            "invalid_tool_call",
        ) from error
    return make_tool_call(name, arguments)


def make_tool_call(
    name: str,
    arguments: Any,
    call_id: Any = None,
) -> dict[str, Any]:
    return {
        "id": call_id if isinstance(call_id, str) else f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def tool_result_text(result: Any) -> str:
    parts = [
        item.text
        for item in result.content
        if getattr(item, "type", None) == "text" and isinstance(item.text, str)
    ]
    text = "\n".join(parts) or "Tool returned no text."
    return f"Tool error: {text}" if result.isError else text


def is_raw_json_tool_call(content: str) -> bool:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("name"), str)
        and isinstance(payload.get("parameters"), dict)
    )


def sse_data(payload: Any) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def error_payload(error: GatewayError) -> dict[str, Any]:
    return {
        "error": {
            "message": str(error),
            "type": "server_error",
            "code": error.code,
        }
    }


def configured_context_window() -> int:
    raw = os.environ.get("FRAUDE_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError("FRAUDE_CONTEXT_WINDOW must be a positive integer.") from error
    if value <= 0:
        raise RuntimeError("FRAUDE_CONTEXT_WINDOW must be a positive integer.")
    return value


def configured_model() -> str:
    return os.environ.get(
        "FRAUDE_MODEL",
        os.environ.get("QWEN_MODEL", DEFAULT_MODEL_ID),
    )


def configured_model_url() -> str:
    return os.environ.get(
        "FRAUDE_MODEL_API_BASE_URL",
        os.environ.get("QWEN_API_BASE_URL", DEFAULT_MODEL_URL),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    workspace = Path(os.environ.get("FRAUDE_WORKSPACE", Path.cwd())).resolve()
    server_path = Path(__file__).with_name("mcp_server.py").resolve()
    child_env = dict(os.environ)
    child_env["FRAUDE_WORKSPACE"] = str(workspace)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env=child_env,
        cwd=workspace,
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            async with httpx.AsyncClient(timeout=120) as client:
                app.state.gateway = AgentGateway(
                    client,
                    session,
                    tools,
                    configured_model_url(),
                    configured_context_window(),
                    configured_model(),
                )
                yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(GatewayError)
async def handle_gateway_error(_request: Request, error: GatewayError) -> JSONResponse:
    return JSONResponse(error_payload(error), status_code=502)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    _request: Request,
    _error: RequestValidationError,
) -> JSONResponse:
    error = GatewayError("Invalid chat completion request.", "invalid_request")
    return JSONResponse(error_payload(error), status_code=400)


@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    gateway: AgentGateway = request.app.state.gateway
    return await gateway.list_models()


@app.post("/v1/chat/completions")
async def chat_completion(
    body: ChatCompletionRequest,
    request: Request,
) -> StreamingResponse:
    gateway: AgentGateway = request.app.state.gateway

    async def events() -> AsyncIterator[str]:
        try:
            async with gateway.turn_lock:
                result = await gateway.run_agent(
                    body.model,
                    body.messages,
                    request.is_disconnected,
                )
            for start in range(0, len(result.text), SSE_CHUNK_SIZE):
                yield sse_data(
                    {
                        "choices": [
                            {"delta": {"content": result.text[start:start + SSE_CHUNK_SIZE]}}
                        ]
                    }
                )
            yield sse_data({"choices": [], "usage": result.usage})
            yield sse_data("[DONE]")
        except asyncio.CancelledError:
            raise
        except GatewayError as error:
            yield sse_data(error_payload(error))
        except Exception:
            error = GatewayError("Unexpected gateway failure.", "internal_error")
            yield sse_data(error_payload(error))

    return StreamingResponse(events(), media_type="text/event-stream")
