# Running the local model (Qwen3-Coder-30B-A3B via MLX)

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

Stock `mlx_lm.server` does not include context-window metadata in `/v1/models`,
so Fraude Code uses an explicit effective limit. In a second terminal run:

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
