import json
import unittest
from typing import Any

import httpx
from openai_harmony import HarmonyEncodingName, ReasoningEffort, load_harmony_encoding

from harmony_server import (
    ChatMessage,
    FunctionCall,
    FunctionDefinition,
    Generation,
    HarmonyServerError,
    ToolCall,
    ToolDefinition,
    create_app,
    openai_messages_to_harmony,
    parse_harmony_completion,
)


READ_TOOL = ToolDefinition(
    type="function",
    function=FunctionDefinition(
        name="read_file",
        description="Read a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
)


class ConversationConversionTest(unittest.TestCase):
    def test_converts_instructions_tools_and_prior_tool_exchange(self) -> None:
        conversation = openai_messages_to_harmony(
            [
                ChatMessage(role="system", content="Use filesystem tools."),
                ChatMessage(role="user", content="Read package.json"),
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionCall(
                                name="read_file",
                                arguments='{"path":"package.json"}',
                            ),
                        )
                    ],
                ),
                ChatMessage(
                    role="tool",
                    tool_call_id="call_1",
                    content='{"name":"fraude-code"}',
                ),
            ],
            [READ_TOOL],
            ReasoningEffort.MEDIUM,
        )

        rendered = conversation.to_dict()["messages"]
        self.assertEqual(rendered[0]["role"], "system")
        self.assertEqual(rendered[1]["role"], "developer")
        developer = rendered[1]["content"][0]
        self.assertEqual(developer["instructions"], "Use filesystem tools.")
        self.assertEqual(
            developer["tools"]["functions"]["tools"][0]["name"], "read_file"
        )
        self.assertEqual(rendered[3]["recipient"], "functions.read_file")
        self.assertEqual(rendered[3]["channel"], "commentary")
        self.assertEqual(rendered[4]["role"], "tool")
        self.assertEqual(rendered[4]["name"], "functions.read_file")
        self.assertEqual(rendered[4]["recipient"], "assistant")

    def test_rejects_unmatched_tool_result(self) -> None:
        with self.assertRaisesRegex(HarmonyServerError, "does not match"):
            openai_messages_to_harmony(
                [ChatMessage(role="tool", tool_call_id="missing", content="nope")],
                [READ_TOOL],
                ReasoningEffort.MEDIUM,
            )


class HarmonyParsingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def fixture(self, raw: str) -> Generation:
        tokens = self.encoding.encode(raw, allowed_special="all")
        return Generation(tokens[:-1], tokens[-1])

    def test_extracts_final_without_exposing_analysis(self) -> None:
        generation = self.fixture(
            "<|channel|>analysis<|message|>private reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Public answer.<|return|>"
        )

        message, finish_reason = parse_harmony_completion(self.encoding, generation, [])

        self.assertEqual(message, {"role": "assistant", "content": "Public answer."})
        self.assertEqual(finish_reason, "stop")
        self.assertNotIn("private reasoning", json.dumps(message))

    def test_converts_filesystem_action_to_openai_tool_call(self) -> None:
        generation = self.fixture(
            "<|channel|>commentary to=functions.read_file <|constrain|>json"
            '<|message|>{"path":"package.json"}<|call|>'
        )

        message, finish_reason = parse_harmony_completion(
            self.encoding,
            generation,
            [READ_TOOL],
        )

        self.assertEqual(finish_reason, "tool_calls")
        function = message["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "read_file")
        self.assertEqual(json.loads(function["arguments"]), {"path": "package.json"})

    def test_accepts_analysis_before_tool_action(self) -> None:
        generation = self.fixture(
            "<|channel|>analysis<|message|>need the file<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.read_file "
            '<|constrain|>json<|message|>{"path":"package.json"}<|call|>'
        )

        message, _ = parse_harmony_completion(self.encoding, generation, [READ_TOOL])

        self.assertEqual(message["content"], None)
        self.assertNotIn("need the file", json.dumps(message))

    def test_rejects_malformed_recipient(self) -> None:
        generation = self.fixture(
            "<|channel|>commentary to=read_file <|constrain|>json"
            '<|message|>{"path":"package.json"}<|call|>'
        )
        with self.assertRaisesRegex(HarmonyServerError, "recipient"):
            parse_harmony_completion(self.encoding, generation, [READ_TOOL])

    def test_rejects_malformed_arguments(self) -> None:
        generation = self.fixture(
            "<|channel|>commentary to=functions.read_file <|constrain|>json"
            "<|message|>{not-json}<|call|>"
        )
        with self.assertRaisesRegex(HarmonyServerError, "invalid arguments"):
            parse_harmony_completion(self.encoding, generation, [READ_TOOL])

    def test_rejects_completion_without_final_action(self) -> None:
        tokens = self.encoding.encode(
            "<|channel|>analysis<|message|>unfinished<|end|>",
            allowed_special="all",
        )
        with self.assertRaisesRegex(HarmonyServerError, "without a final action"):
            parse_harmony_completion(self.encoding, Generation(tokens, None), [])


class FakeGenerator:
    def __init__(self, generation: Generation) -> None:
        self.generation = generation
        self.calls: list[tuple[list[int], list[int], int]] = []

    def generate(
        self, prompt_tokens: Any, stop_tokens: Any, max_tokens: int
    ) -> Generation:
        self.calls.append((list(prompt_tokens), list(stop_tokens), max_tokens))
        return self.generation


class HarmonyEndpointTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        tokens = self.encoding.encode(
            "<|channel|>final<|message|>Hello.<|return|>",
            allowed_special="all",
        )
        self.generator = FakeGenerator(Generation(tokens[:-1], tokens[-1]))
        self.app = create_app("gpt-oss", max_tokens=77, generator=self.generator)

    async def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async with self.app.router.lifespan_context(self.app):
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await client.request(method, path, **kwargs)

    async def test_lists_only_loaded_model(self) -> None:
        response = await self.request("GET", "/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()["data"]], ["gpt-oss"])

    async def test_rejects_model_mismatch_with_openai_error(self) -> None:
        response = await self.request(
            "POST",
            "/v1/chat/completions",
            json={"model": "other", "messages": []},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")

    async def test_rejects_streaming_with_openai_error(self) -> None:
        response = await self.request(
            "POST",
            "/v1/chat/completions",
            json={"model": "gpt-oss", "messages": [], "stream": True},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "unsupported_streaming")

    async def test_reports_exact_usage_from_rendered_and_generated_tokens(self) -> None:
        response = await self.request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gpt-oss",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        usage = response.json()["usage"]
        prompt_count = len(self.generator.calls[0][0])
        completion_count = len(self.generator.generation.tokens) + 1
        self.assertEqual(usage["prompt_tokens"], prompt_count)
        self.assertEqual(usage["completion_tokens"], completion_count)
        self.assertEqual(usage["total_tokens"], prompt_count + completion_count)
        self.assertEqual(self.generator.calls[0][2], 77)

    async def test_validation_error_uses_openai_shape(self) -> None:
        response = await self.request(
            "POST",
            "/v1/chat/completions",
            json={"messages": []},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")

    async def test_malformed_harmony_uses_openai_error_shape(self) -> None:
        tokens = self.encoding.encode(
            "<|channel|>analysis<|message|>unfinished<|end|>",
            allowed_special="all",
        )
        self.generator.generation = Generation(tokens, None)

        response = await self.request(
            "POST",
            "/v1/chat/completions",
            json={"model": "gpt-oss", "messages": [{"role": "user", "content": "Hi"}]},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "invalid_harmony")


if __name__ == "__main__":
    unittest.main()
