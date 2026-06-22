import asyncio
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from typing import Any

import httpx

from app import (
    AgentGateway,
    AgentResult,
    ChatMessage,
    GatewayError,
    app,
    configured_model,
    configured_model_url,
    is_raw_json_tool_call,
    normalize_tool_calls,
    parse_qwen_tool_call,
)


TOOL_SCHEMAS = {
    "read_file": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "write_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
}


def tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} description",
        inputSchema=TOOL_SCHEMAS[name],
    )


def completion(
    message: dict[str, Any],
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"choices": [{"message": message}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


class QwenToolParserTest(unittest.TestCase):
    def test_parses_observed_qwen_markup_without_opening_wrapper(self) -> None:
        call = parse_qwen_tool_call(
            """<function=write_file>
<parameter=path>
backend/new.txt
</parameter>
<parameter=content>
hello
</parameter>
</function>
</tool_call>""",
            TOOL_SCHEMAS,
        )

        self.assertEqual(call["function"]["name"], "write_file")
        self.assertEqual(
            json.loads(call["function"]["arguments"]),
            {"path": "backend/new.txt", "content": "hello"},
        )

    def test_accepts_plain_prose_before_tool_markup(self) -> None:
        call = parse_qwen_tool_call(
            """I'll create reverse.py.

<function=write_file>
<parameter=path>
reverse.py
</parameter>
<parameter=content>
items = []
items.append("a")
</parameter>
</function>""",
            TOOL_SCHEMAS,
        )

        self.assertEqual(call["function"]["name"], "write_file")
        self.assertEqual(
            json.loads(call["function"]["arguments"]),
            {
                "path": "reverse.py",
                "content": 'items = []\nitems.append("a")',
            },
        )

    def test_rejects_trailing_prose_after_tool_markup(self) -> None:
        with self.assertRaisesRegex(GatewayError, "malformed"):
            parse_qwen_tool_call(
                "<function=read_file><parameter=path>x</parameter>"
                "</function> done",
                TOOL_SCHEMAS,
            )

    def test_rejects_unknown_tool(self) -> None:
        with self.assertRaisesRegex(GatewayError, "unknown tool"):
            parse_qwen_tool_call("<function=delete_file></function>", TOOL_SCHEMAS)

    def test_rejects_duplicate_parameter(self) -> None:
        markup = (
            "<function=read_file>"
            "<parameter=path>a</parameter>"
            "<parameter=path>b</parameter>"
            "</function>"
        )
        with self.assertRaisesRegex(GatewayError, "repeated"):
            parse_qwen_tool_call(markup, TOOL_SCHEMAS)

    def test_preserves_meaningful_content_whitespace(self) -> None:
        call = parse_qwen_tool_call(
            """<function=write_file>
<parameter=path>
new.txt
</parameter>
<parameter=content>
  indented

</parameter>
</function>""",
            TOOL_SCHEMAS,
        )

        self.assertEqual(
            json.loads(call["function"]["arguments"])["content"],
            "  indented\n",
        )

    def test_validates_structured_tool_arguments(self) -> None:
        with self.assertRaisesRegex(GatewayError, "invalid arguments"):
            normalize_tool_calls(
                {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({"path": "only-path"}),
                            },
                        }
                    ]
                },
                TOOL_SCHEMAS,
            )

    def test_detects_bare_json_tool_protocol(self) -> None:
        self.assertTrue(
            is_raw_json_tool_call(
                '{"name":"write_file","parameters":{"path":"example.txt"}}'
            )
        )
        self.assertFalse(is_raw_json_tool_call('{"message":"ordinary JSON"}'))


class GatewayConfigurationTest(unittest.TestCase):
    def test_generic_model_settings_take_precedence_over_legacy_qwen_settings(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FRAUDE_MODEL": "gpt-oss",
                "QWEN_MODEL": "qwen",
                "FRAUDE_MODEL_API_BASE_URL": "http://harmony.test",
                "QWEN_API_BASE_URL": "http://qwen.test",
            },
            clear=True,
        ):
            self.assertEqual(configured_model(), "gpt-oss")
            self.assertEqual(configured_model_url(), "http://harmony.test")

    def test_legacy_qwen_settings_remain_supported(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "QWEN_MODEL": "qwen",
                "QWEN_API_BASE_URL": "http://qwen.test",
            },
            clear=True,
        ):
            self.assertEqual(configured_model(), "qwen")
            self.assertEqual(configured_model_url(), "http://qwen.test")


class FakeMcpSession:
    def __init__(self, is_error: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.is_error = is_error

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"result for {name}")],
            isError=self.is_error,
        )


class AgentGatewayTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.responses: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={"object": "list", "data": [{"id": "qwen"}]},
                )
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json=self.responses.pop(0))

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.mcp = FakeMcpSession()
        self.gateway = AgentGateway(
            self.client,
            self.mcp,  # type: ignore[arg-type]
            [tool("read_file"), tool("write_file")],
            "http://qwen.test",
            4096,
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_executes_structured_tool_and_returns_final_usage(self) -> None:
        self.responses.extend(
            [
                completion(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_read",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "package.json"}),
                                },
                            }
                        ],
                    }
                ),
                completion(
                    {"role": "assistant", "content": "The package is fraude-code."},
                    {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                ),
            ]
        )

        result = await self.gateway.run_agent(
            "qwen",
            [ChatMessage(role="user", content="Read package.json")],
            never_cancelled,
        )

        self.assertEqual(result.text, "The package is fraude-code.")
        self.assertEqual(result.usage["total_tokens"], 25)
        self.assertEqual(self.mcp.calls, [("read_file", {"path": "package.json"})])
        system_message = self.requests[0]["messages"][0]["content"]
        self.assertIn("create_plan", system_message)
        self.assertIn("get_plan", system_message)
        self.assertEqual(self.requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(self.requests[1]["messages"][-1]["content"], "result for read_file")

    async def test_executes_raw_qwen_write_markup(self) -> None:
        self.responses.extend(
            [
                completion(
                    {
                        "role": "assistant",
                        "content": (
                            "<function=write_file>"
                            "<parameter=path>new.txt</parameter>"
                            "<parameter=content>hello</parameter>"
                            "</function></tool_call>"
                        ),
                    }
                ),
                completion(
                    {"role": "assistant", "content": "Created new.txt."},
                    {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34},
                ),
            ]
        )

        result = await self.gateway.run_agent(
            "qwen",
            [ChatMessage(role="user", content="Create new.txt")],
            never_cancelled,
        )

        self.assertEqual(result.text, "Created new.txt.")
        self.assertEqual(
            self.mcp.calls,
            [("write_file", {"path": "new.txt", "content": "hello"})],
        )

    async def test_stops_after_tool_round_limit(self) -> None:
        tool_response = completion(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "package.json"}),
                        },
                    }
                ],
            }
        )
        self.responses.extend([tool_response] * 5)

        with self.assertRaisesRegex(GatewayError, "5-round"):
            await self.gateway.run_agent(
                "qwen",
                [ChatMessage(role="user", content="Keep reading")],
                never_cancelled,
            )

    async def test_honors_cancellation_before_model_call(self) -> None:
        async def cancelled() -> bool:
            return True

        with self.assertRaises(asyncio.CancelledError):
            await self.gateway.run_agent(
                "qwen",
                [ChatMessage(role="user", content="Stop")],
                cancelled,
            )

    async def test_models_include_context_window(self) -> None:
        payload = await self.gateway.list_models()

        self.assertEqual(payload["data"][0]["context_window"], 4096)

    async def test_models_filter_to_configured_model(self) -> None:
        gateway = AgentGateway(
            self.client,
            self.mcp,  # type: ignore[arg-type]
            [tool("read_file")],
            "http://qwen.test",
            4096,
            "qwen",
        )

        payload = await gateway.list_models()

        self.assertEqual([model["id"] for model in payload["data"]], ["qwen"])

    async def test_rejects_stock_mlx_server_for_gpt_oss(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Server": "BaseHTTP/0.6 Python/3.12.8"},
                json={"object": "list", "data": [{"id": "gpt-oss"}]},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gateway = AgentGateway(
            client,
            self.mcp,  # type: ignore[arg-type]
            [tool("read_file")],
            "http://mlx.test",
            4096,
            "gpt-oss",
        )
        try:
            with self.assertRaisesRegex(GatewayError, "harmony_server.py"):
                await gateway.list_models()
        finally:
            await client.aclose()

    async def test_rejects_raw_harmony_markup_from_model_server(self) -> None:
        self.responses.append(
            completion(
                {
                    "role": "assistant",
                    "content": (
                        "<|channel|>analysis<|message|>private<|end|>"
                        "<|start|>assistant<|channel|>final<|message|>hello"
                    ),
                },
                {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            )
        )

        with self.assertRaisesRegex(GatewayError, "harmony_server.py"):
            await self.gateway.run_agent(
                "qwen",
                [ChatMessage(role="user", content="hi")],
                never_cancelled,
            )


class FakeEndpointGateway:
    def __init__(self) -> None:
        self.turn_lock = asyncio.Lock()

    async def run_agent(self, *_args: Any) -> AgentResult:
        return AgentResult(
            "hello",
            {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )


class EndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_endpoint_emits_frontend_sse_contract(self) -> None:
        app.state.gateway = FakeEndpointGateway()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway.test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )

        self.assertEqual(response.status_code, 200)
        records = [record for record in response.text.split("\n\n") if record]
        self.assertEqual(
            json.loads(records[0].removeprefix("data: ")),
            {"choices": [{"delta": {"content": "hello"}}]},
        )
        self.assertEqual(
            json.loads(records[1].removeprefix("data: "))["usage"]["total_tokens"],
            3,
        )
        self.assertEqual(records[2], "data: [DONE]")

    async def test_invalid_request_uses_openai_error_shape(self) -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway.test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen", "messages": [{"role": "tool"}]},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")


async def never_cancelled() -> bool:
    return False


if __name__ == "__main__":
    unittest.main()
