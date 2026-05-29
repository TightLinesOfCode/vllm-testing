# Driving Claude Code with the local vLLM (Nemotron-3-Super) endpoint

## Quickstart

```bash
# 1. Install Claude Code + claude-code-router
npm install -g @anthropic-ai/claude-code @musistudio/claude-code-router

# 2. Write the router config (points at the local Nemotron endpoint)
mkdir -p ~/.claude-code-router
cat > ~/.claude-code-router/config.json <<'EOF'
{
  "LOG": false,
  "API_TIMEOUT_MS": 600000,
  "Providers": [
    {
      "name": "vllm",
      "api_base_url": "http://192.168.1.101:8989/v1/chat/completions",
      "api_key": "dummy",
      "models": ["nemotron-3-super"]
    }
  ],
  "Router": {
    "default": "vllm,nemotron-3-super",
    "background": "vllm,nemotron-3-super",
    "think": "vllm,nemotron-3-super",
    "longContext": "vllm,nemotron-3-super",
    "webSearch": "vllm,nemotron-3-super"
  }
}
EOF

# 3. Launch Claude Code on the local model
ccr code

# 4. (Optional) Run autonomously toward a goal — flags pass straight through to claude.
#    --permission-mode acceptEdits auto-approves edits; --max-turns bounds the run.
ccr code -p --permission-mode acceptEdits --max-turns 40 \
  "Implement X; run the tests after each change; stop when they pass."
```

For scripted runs you can instead `eval "$(ccr activate)"` to export the router env into
your shell, then call `claude` directly with any flag from `claude --help`.

The rest of this doc explains each step, replication on other machines, and the vLLM server setup.

---

Claude Code speaks only the **native Anthropic Messages API** (`POST /v1/messages`), while
the local vLLM server speaks the **OpenAI API** (`/v1/chat/completions`). We bridge them
with **claude-code-router (CCR)**, which correctly translates Anthropic ⇄ OpenAI for both
streaming and tool calls:

```
Claude Code  ──>  claude-code-router :3456  ──(OpenAI /v1/chat/completions)──>  vLLM @ 192.168.1.101:8989
```

The vLLM server (`vllm/vllm-openai:v0.20.0`, served id `nemotron-3-super`,
`--tool-call-parser qwen3_coder` + `--reasoning-parser super_v3` + the model's built-in chat
template) emits valid tool calls **and** keeps reasoning out of responses (see
[appendix](#appendix-vllm-backend-launch-command) to stand one up on a new machine).

> **Why not LiteLLM?** A LiteLLM proxy returns correct content, but its Anthropic streaming
> translation emits a **duplicate `message_start` event** on every response, which hangs
> Claude Code right after the first prompt. CCR streams a single, correct `message_start`
> and works.

> **Replicating on another machine — substitute these two values everywhere below:**
> | Placeholder | This setup | Replace with |
> | --- | --- | --- |
> | `HOST:PORT` | `192.168.1.101:8989` | wherever your vLLM is reachable from the new machine |
> | `MODEL` | `nemotron-3-super` | the served model id from Step 0 |
>
> If the new machine will reach the *existing* vLLM server, only the host/IP may differ. If
> you're also rebuilding vLLM there, see the appendix first, then come back to Step 0.

## What to install where (client vs. server)

There are two roles, which may or may not be the same machine:

| Role | What runs there | Install on a new machine? |
| --- | --- | --- |
| **Client** — where you run Claude Code | Claude Code + **CCR** + `~/.claude-code-router/config.json` | **Yes, always.** CCR is the local bridge; it must live wherever you type `ccr code`. |
| **Server** — the GPU host | vLLM serving the model | **Only if** the model isn't already running/reachable. If your existing endpoint (e.g. `192.168.1.101:8989`) is reachable over the network, skip this and just point the client's config at it. |

So **every client machine** needs CCR + the config (Prerequisites → Step 3). The vLLM
**appendix** is only needed when that machine is also hosting the model.

```bash
# On a new client machine — install both CLIs at once:
npm install -g @anthropic-ai/claude-code @musistudio/claude-code-router
```

## Prerequisites

- A running vLLM (or any OpenAI-compatible) server with **tool calling enabled**, reachable
  from this machine. See the appendix to launch one.
- **Node.js ≥ 18** and **npm** — `node -v && npm -v` (install from <https://nodejs.org> or via `nvm`).
- **Claude Code CLI**:
  ```bash
  npm install -g @anthropic-ai/claude-code
  claude --version
  ```

## Step 0 — confirm the endpoint and model name

```bash
curl -s http://HOST:PORT/v1/models
```

Note the `"id"` in the response (e.g. `nemotron-3-super`) — that's the **MODEL** name you'll
put in the config below.

## 1. Install claude-code-router

```bash
npm install -g @musistudio/claude-code-router
ccr -v        # confirm it's on PATH
```

## 2. Create `~/.claude-code-router/config.json`

```bash
mkdir -p ~/.claude-code-router
```

Then write this file (replace `HOST:PORT` and `MODEL` with your values from Step 0):

```json
{
  "LOG": false,
  "API_TIMEOUT_MS": 600000,
  "Providers": [
    {
      "name": "vllm",
      "api_base_url": "http://HOST:PORT/v1/chat/completions",
      "api_key": "dummy",
      "models": ["MODEL"]
    }
  ],
  "Router": {
    "default": "vllm,MODEL",
    "background": "vllm,MODEL",
    "think": "vllm,MODEL",
    "longContext": "vllm,MODEL",
    "webSearch": "vllm,MODEL"
  }
}
```

For reference, this setup's concrete values are `http://192.168.1.101:8989/v1/chat/completions`
and `nemotron-3-super`.

- `api_base_url` is the **full** path ending in `/v1/chat/completions`.
- **No `transformer`** — CCR sends OpenAI-format requests by default, which is what vLLM
  wants. (The `Anthropic` transformer would wrongly send Anthropic format to vLLM.)
- `api_key: "dummy"` — vLLM ignores it; CCR just wants a non-empty value.
- `API_TIMEOUT_MS` is generous because a large local model can be slow on big prompts.
- `Router` values use the format `"providerName,modelName"`. With one model, every category
  points at the same `vllm,MODEL`.

## 3. Launch Claude Code through CCR

From your project directory:

```bash
ccr code
```

This starts the CCR service (port 3456) and launches Claude Code routed through it. No
`ANTHROPIC_*` env vars needed — CCR sets them. After editing `config.json`, apply changes
with `ccr restart`. Other handy commands: `ccr status`, `ccr stop`.

## Verification

1. **Backend tool calling** — confirm the vLLM endpoint itself returns a structured
   `tool_calls` array (this is the hard requirement for Claude Code):

   ```bash
   curl -s http://HOST:PORT/v1/chat/completions \
     -H 'Content-Type: application/json' -d '{
       "model":"MODEL","max_tokens":256,
       "messages":[{"role":"user","content":"What is the weather in Paris?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
       "tool_choice":"auto"}'
   ```

   Expect a `tool_calls` array and `"finish_reason":"tool_calls"`. If it's missing, your
   server lacks `--enable-auto-tool-choice` / a working `--tool-call-parser` (see appendix).

2. **Router streams clean Anthropic SSE** — with the service running (`ccr status` shows
   Running):

   ```bash
   curl -sN http://127.0.0.1:3456/v1/messages \
     -H 'Content-Type: application/json' -H 'x-api-key: dummy' \
     -H 'anthropic-version: 2023-06-01' -d '{
       "model":"MODEL","max_tokens":64,"stream":true,
       "messages":[{"role":"user","content":"say hi"}]}'
   ```

   Expect exactly **one** `event: message_start`, then content and `message_stop`.

3. **End to end** — in the Claude Code session, ask it to `ls` or read a file. If it
   actually invokes the tool (not just narrates), the loop works.

## FAQ: is it really using my local model?

Claude Code will **display "Claude Opus 4.8"** in its status line, and if you ask it *"what
model are you?"* it will **answer "Opus 4.8"** — even when it's running entirely on your
local model. Neither is evidence of the real backend:

- The **status-line label** is cosmetic: Claude Code doesn't know CCR is transparently
  rerouting its API calls, so it shows its own default model name.
- **Asking the model its identity doesn't work**: Claude Code injects a system prompt that
  says *"You are powered by the model named Opus 4.8…"*. Your local model just reads that
  and parrots it back. A model has no reliable way to know its own identity.

**The only reliable check is where the requests go.** Temporarily set `"LOG": true` in
`~/.claude-code-router/config.json`, run `ccr restart`, then watch live:

```bash
tail -f ~/.claude-code-router/logs/*.log | grep --line-buffered requestUrl
```

Send a prompt in Claude Code — you'll see requests hit `http://HOST:PORT/v1/chat/completions`
with `"model":"MODEL"` (e.g. `http://192.168.1.101:8989/...` → `nemotron-3-super`). That
confirms your local model is doing the work. Set `"LOG"` back to `false` + `ccr restart`
when done (logs grow on disk and contain your prompts/responses).

## Notes / gotchas

- **Reasoning leak** — fixed by running vLLM with `--reasoning-parser super_v3` (see the
  appendix). Without it, the model's `<think>…</think>` reasoning bleeds into plain-text
  replies. If your endpoint still leaks `</think>`, the backend is missing that flag.
- **Quality** — Claude Code's prompts are tuned for Claude models; expect more retries and
  lower quality than Claude, though 120B Nemotron is on the capable end.
- **Token budget** — keep `max_tokens` generous so reasoning + tool calls don't get
  truncated mid-call (the 128K context helps).

## Appendix: vLLM backend launch command

Only needed if you're standing up the vLLM server on a new machine (otherwise point CCR at
your existing endpoint). This is the **verified** config: tool calling **and** clean
reasoning both work together. It requires a GPU host with Docker + NVIDIA Container Toolkit,
an `HF_TOKEN` exported in the environment, and one file mounted into the container:

- `~/super_v3_reasoning_parser.py` — the reasoning parser (keeps `<think>…</think>` out of
  responses). Download it from the model's (public) HF repo:

  ```bash
  curl -L -o ~/super_v3_reasoning_parser.py \
    https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/resolve/main/super_v3_reasoning_parser.py
  ```

This is NVIDIA's official combo (`qwen3_coder` tool parser + `super_v3` reasoning parser +
the model's **built-in** chat template — no custom hermes template). The earlier
hermes-template approach worked for tool calls but **broke them once a reasoning parser was
added**; this combo is the one that does both.

```bash
docker run -d --restart always --gpus all \
  --shm-size=16GB \
  --name nemotron-super \
  -e HF_TOKEN=$HF_TOKEN \
  -e MAMBA_CACHE_RS_ROUNDING=1 \
  -e MAMBA_CACHE_PHILOX_ROUNDS=5 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/super_v3_reasoning_parser.py:/opt/super_v3_reasoning_parser.py:ro \
  -p 8989:8000 \
  vllm/vllm-openai:v0.20.0 \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super \
    --quantization nvfp4 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 131072 \
    --enable-chunked-prefill \
    --max-num-seqs 64 \
    --max-num-batched-tokens 65536 \
    --async-scheduling \
    --moe-backend marlin \
    --attention-backend TRITON_ATTN \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --reasoning-parser-plugin /opt/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --host 0.0.0.0 \
    --port 8000
```

Key flags:
- **Tool calling**: `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder`. No
  `--chat-template` flag — it uses the model's built-in `chat_template.jinja`, which is
  designed to pair with `qwen3_coder` and the reasoning parser.
- **Reasoning**: `--reasoning-parser super_v3` + `--reasoning-parser-plugin` (the downloaded
  plugin). `super_v3` is a thin subclass of vLLM's DeepSeek-R1 parser that strips
  `<think>…</think>` into a separate `reasoning_content` field.

> **Why not the hermes template + `--tool-call-parser hermes`?** That combo emits valid tool
> calls on its own, but adding a reasoning parser on top of it **breaks tool-call parsing**
> (the `<tool_call>` block leaks into text and isn't extracted — Claude Code then hangs).
> Verified on this model in vLLM `v0.20.0`. The `qwen3_coder` + built-in-template combo
> above is the one where tool calls *and* reasoning both work. After any (re)start, confirm
> both via Verification step 1 (a `tool_calls` array) and a clean `content` (no `</think>`).

## References

- [claude-code-router](https://github.com/musistudio/claude-code-router)
- [vLLM Nemotron-3-Super blog](https://vllm.ai/blog/2026-03-11-nemotron-3-super)
- [NVIDIA Nemotron-3-Super vLLM cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/vllm_cookbook.ipynb)
- [Claude Code LLM gateway docs](https://code.claude.com/docs/en/llm-gateway.md)
- [Claude Code environment variables](https://code.claude.com/docs/en/env-vars.md)
