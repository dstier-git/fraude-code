# Running a local model through MLX

Fraude supports the existing stock Qwen server and the repository-owned GPT-OSS
Harmony server. Both expose the same non-streaming OpenAI endpoint to the Python
gateway on `127.0.0.1:8080`.

## GPT-OSS with Harmony

GPT-OSS must use its Harmony prompt and action protocol. Start the bundled
server from the repository root with arm64 Python:

> Do not use `mlx_lm.server` for GPT-OSS. It returns raw Harmony control tokens
> that the stock server does not parse.

```bash
uv run --python /usr/local/bin/python3.12 \
  --project backend --extra harmony \
  python backend/harmony_server.py \
  --model mlx-community/gpt-oss-20b-MXFP4-Q8 \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 512 --reasoning-effort medium
```

The server loads one model, serializes generation, and intentionally supports
only non-streaming chat completions. It fails at startup if Python is not arm64
or Metal is unavailable.

In another terminal, use the same model ID for the gateway and Node UI:

```bash
export FRAUDE_MODEL=mlx-community/gpt-oss-20b-MXFP4-Q8
uv run --project backend uvicorn app:app \
  --app-dir backend --host 127.0.0.1 --port 8000
```

```bash
FRAUDE_MODEL=mlx-community/gpt-oss-20b-MXFP4-Q8 \
FRAUDE_CONTEXT_WINDOW=32768 \
npm start
```

`FRAUDE_MODEL_API_BASE_URL` overrides the model-server URL. The legacy
`QWEN_MODEL` and `QWEN_API_BASE_URL` variables remain supported at lower
precedence.

## Qwen3-Coder-30B-A3B

One-time install is already done (`mlx-lm` via `uv`, forced onto arm64 Python because the
default `python3` is x86_64 anaconda and has no MLX wheels). Model:
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` (~16 GB, cached in `~/.cache/huggingface`).

## Every session: raise the GPU memory limit FIRST

```bash
sudo sysctl iogpu.wired_limit_mb=19456
```

- Resets on reboot — re-run before using the model.
- **Must be run from a real Terminal window.** `sudo` needs a TTY, so it will NOT work from a
  non-interactive shell (a script, a cron job, or Claude Code's `!` prefix). This is the one
  step that can't be automated/scripted on this machine.

## Then run the model — three ways (a script is NOT required)

**1. Interactive CLI (simplest):**
```bash
mlx_lm.chat --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

**2. One-shot generate:**
```bash
mlx_lm.generate --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --prompt "..." --max-tokens 256
```

**3. OpenAI-compatible server (for harnesses/apps):**
```bash
mlx_lm.server --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --port 8080
# → POST http://127.0.0.1:8080/v1/chat/completions  (supports tools / streaming)
```

The model itself does **not** need to be wrapped in a script. The example TUI
(`mlx_chat.py`) just talks to option 3's `/v1` endpoint — it's optional, not required.

### Connect Fraude Code

Fraude Code now connects through its Python agent gateway. Keep `mlx_lm.server`
on port 8080, then start the gateway from the repository root in a second
terminal:

```bash
uv run --project backend uvicorn app:app \
  --app-dir backend \
  --host 127.0.0.1 \
  --port 8000
```

The gateway launches the filesystem MCP server over stdio, selects the non-DWQ
Qwen model by default, and reports the configured context window. Override the
selection with `FRAUDE_MODEL` (or legacy `QWEN_MODEL`) when needed. In a third
terminal run:

```bash
FRAUDE_MODEL=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
FRAUDE_CONTEXT_WINDOW=32768 \
npm start
```

The model configuration declares a 262K maximum, but 32K is the intended local
operating limit under this machine's memory budget.

## Notes
- Server holds ~16 GB idle; keep context ~16–32K so the KV cache stays under the ~19 GB cap.
- ~47 tok/s generation on this M4. Ollama is untouched and coexists.
