# Fraude Code frontend/backend contract

This document is the complete backend handoff for the v1 Ink frontend. The
frontend owns terminal rendering, prompt editing, in-memory conversation
history, elapsed time, cancellation input, and context-percentage display. The
backend owns model identity, generation, and exact token usage. Context capacity
comes from backend metadata when available and otherwise from frontend
configuration.

## Connection

- The frontend reads `FRAUDE_API_BASE_URL`, defaulting to
  `http://127.0.0.1:8000`.
- The frontend reads `FRAUDE_CONTEXT_WINDOW` as the effective context limit,
  defaulting to `32768`. Server metadata takes precedence when available.
- If `FRAUDE_API_KEY` is set, every request includes
  `Authorization: Bearer <value>`.
- Responses must use UTF-8.
- The backend may be stateless. The frontend sends the full conversation with
  every completion request.

## 1. Model metadata

The frontend sends:

```http
GET /v1/models
Accept: application/json
```

The backend must return an OpenAI-compatible model list. A backend may include
the `context_window` extension as a positive token count:

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
      "object": "model",
      "context_window": 32768
    }
  ]
}
```

The frontend accepts `context_window_tokens` as a compatibility alias. Stock
`mlx_lm.server` responses that contain only `id`, `object`, and `created` are
supported: the frontend uses `FRAUDE_CONTEXT_WINDOW`, or `32768` when the
variable is unset. If `FRAUDE_MODEL` is set, its value must match an `id`;
otherwise the first usable model is selected. The selected `id` is rendered on
the cover. The effective context window and final usage are used to calculate:

```text
context left = clamp(0, 100, (context_window - total_tokens) / context_window * 100)
```

## 2. Streaming completion

For each submitted prompt, the frontend sends:

```http
POST /v1/chat/completions
Content-Type: application/json
Accept: text/event-stream
```

```json
{
  "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
  "messages": [
    {"role": "user", "content": "add rate limiting"},
    {"role": "assistant", "content": "I will inspect the routes."},
    {"role": "user", "content": "continue"}
  ],
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Only `user` and `assistant` text messages are sent in v1. The frontend does not
send tool definitions and does not render tool calls. The Python gateway owns
the MCP agent loop and must therefore produce assistant text rather than
`tool_calls` deltas.

The response must be `text/event-stream`. Text reaches the frontend in
`choices[0].delta.content`:

```text
data: {"choices":[{"delta":{"content":"I’ll add "}}]}

data: {"choices":[{"delta":{"content":"a limiter."}}]}

```

Exact usage reaches the frontend in a usage chunk. It may accompany a choice or
arrive in a choice-less final chunk:

```text
data: {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":8,"total_tokens":50}}

data: [DONE]

```

SSE records may be split across arbitrary network chunks. Records end with a
blank line. The terminal marker `data: [DONE]` completes the turn.

### Live token count

Incremental token reporting is **TBD**. The frontend displays `— tokens` while
generation is active and uses exact usage only after the backend provides it.
No estimate is fabricated by the frontend.

The v1 Python gateway buffers model/tool rounds before emitting the final text
as SSE chunks. The frontend spinner remains active while those rounds run.

## 3. Cancellation

Escape aborts the frontend request and closes the response stream. The backend
must detect the disconnected request and stop generation promptly. It does not
need a separate cancellation endpoint. Any text received before cancellation is
kept in the transcript and may be included in the next conversation request.

## 4. Errors

Non-2xx responses must use this OpenAI-compatible shape:

```json
{
  "error": {
    "message": "Model is not loaded",
    "type": "server_error",
    "code": "model_unavailable"
  }
}
```

An error may also arrive in an SSE data record:

```text
data: {"error":{"message":"Generation failed","type":"server_error"}}

```

`message` is required; `type` and `code` are optional. The frontend renders the
message as a transcript error and restores prompt input. Transport failures are
handled the same way.

## Running a local model and gateway

### GPT-OSS through the Harmony server

GPT-OSS requires official Harmony rendering and parsing. Start the repo-owned
non-streaming model server with arm64 Python:

Do not substitute `mlx_lm.server` for this command; the stock server does not
parse GPT-OSS Harmony actions.

```bash
uv run --python /usr/local/bin/python3.12 \
  --project backend --extra harmony \
  python backend/harmony_server.py \
  --model mlx-community/gpt-oss-20b-MXFP4-Q8 \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 512 --reasoning-effort medium
```

Then start the gateway and UI with
`FRAUDE_MODEL=mlx-community/gpt-oss-20b-MXFP4-Q8`. Set
`FRAUDE_MODEL_API_BASE_URL` when the model server is not at
`http://127.0.0.1:8080`.

### Qwen through the stock MLX server

Start the stock MLX server:

```bash
mlx_lm.server \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --port 8080
```

Start the Python gateway in a second terminal. It launches the filesystem MCP
server over stdio, exposes only the configured model, and serves the frontend
API on port 8000. `FRAUDE_MODEL` takes precedence over legacy `QWEN_MODEL`;
both default to the non-DWQ Qwen model shown below:

```bash
uv run --project backend uvicorn app:app \
  --app-dir backend \
  --host 127.0.0.1 \
  --port 8000
```

Then start Fraude Code in a third terminal. The 32K context is an operational
limit for the local memory budget; it is intentionally below the model
configuration's theoretical 262K maximum:

```bash
FRAUDE_MODEL=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
FRAUDE_CONTEXT_WINDOW=32768 \
npm start
```

## Backend acceptance checklist

- `GET /v1/models` returns at least one model with an ID. Context capacity is
  optional because the frontend has a configurable fallback.
- Streaming completion emits ordered text deltas and terminates with `[DONE]`.
- The final stream includes exact prompt, completion, and total token counts.
- Disconnecting a stream stops its generation work.
- Errors use the documented JSON shape and do not emit unstructured terminal
  output into the HTTP response.
- The service accepts the complete conversation on every request and does not
  require frontend session storage or a backend-specific session identifier.
