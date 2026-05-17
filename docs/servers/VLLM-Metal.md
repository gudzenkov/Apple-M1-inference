# vLLM-Metal Qwen3.6 Server

Use this path for Qwen3.6-35B-A3B when the client needs an OpenAI-compatible chat endpoint, Qwen thinking output, and structured tool calls.

Hard truth: vLLM-Metal currently targets text-only serving on Apple Silicon. Run Qwen3.6-35B-A3B as a language model; do not expect local vision input parity here.

## Recommendation

- Chat/tools/thinking: `vllm-metal` with `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ`.
- Quantization choice: use 4-bit DWQ. The 6-bit build is better quality on paper but too large for the current free unified memory budget once vLLM KV cache and the embedding server are resident.
- Embeddings: separate llama.cpp server with `Qwen/Qwen3-Embedding-0.6B-GGUF` by default; use 4B or 8B only when memory allows.
- Devcontainer access: use `host.docker.internal`, never container `localhost`, for host-side macOS servers.

## Install

Install vLLM-Metal on the macOS host, not inside the devcontainer:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
vllm --version
```

The install script creates `~/.venv-vllm-metal` by default.

Current vLLM-Metal builds may only auto-force the Qwen3.x multimodal wrapper onto the text backbone for FP8 checkpoints. Qwen3.6 DWQ needs the same text-only path. Apply the local compatibility patch after install or after reinstalling vLLM-Metal:

```bash
scripts/patch-vllm-metal-qwen36-text-backbone.sh
```

## Serve Qwen3.6-35B-A3B

Foreground server:

```bash
cd "$(git rev-parse --show-toplevel)"
scripts/start-vllm-metal-qwen36.sh
```

Equivalent explicit command:

```bash
source ~/.venv-vllm-metal/bin/activate

HF_HUB_DISABLE_XET=1 \
VLLM_METAL_MULTIMODAL_MODE=text-only-compat \
VLLM_METAL_MEMORY_FRACTION=0.50 \
vllm serve mlx-community/Qwen3.6-35B-A3B-4bit-DWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b-a3b \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Increase `MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`, or `VLLM_MAX_NUM_BATCHED_TOKENS` only after a successful short-context smoke test. Qwen3.6 supports long context, but local Mac memory pressure is the real limit.

## Host Smoke Tests

```bash
curl -fsS http://127.0.0.1:8000/v1/models

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Return exactly OK."}],
    "max_tokens": 16,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false},
    "stream": false
  }'
```

Tool-call smoke test:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "What is the weather in Warsaw? Use the tool."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 128,
    "stream": false
  }'
```

The server only emits tool calls. Your client still must execute the tool and send the result back.

## Devcontainer Access

Inside a Docker Desktop or OrbStack devcontainer:

```bash
export OPENAI_BASE_URL=http://host.docker.internal:8000/v1
export OPENAI_API_KEY=local
export QWEN_CHAT_MODEL=qwen3.6-35b-a3b

curl -fsS "$OPENAI_BASE_URL/models"
```

Use the model name returned by `/v1/models`; the repo script serves `qwen3.6-35b-a3b`.

## Troubleshooting

- `vllm: command not found`: activate `~/.venv-vllm-metal/bin/activate` or reinstall vLLM-Metal.
- `Missing ... vision_tower ...` while loading Qwen3.6 DWQ: run `scripts/patch-vllm-metal-qwen36-text-backbone.sh`; the checkpoint is text-only but its config is exposed through a multimodal wrapper.
- Out of memory during load: keep the 4-bit DWQ model, reduce `MAX_MODEL_LEN`, close other GPU-heavy apps, or lower `VLLM_METAL_MEMORY_FRACTION`. Do not switch to 6-bit unless you have substantially more free unified memory.
- Tool calls arrive as plain text: confirm the server was started with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
- No reasoning output: confirm `--reasoning-parser qwen3` is present and the client is not stripping reasoning fields.
- Port conflict on `8000`: set `VLLM_PORT=8010` and update `OPENAI_BASE_URL` accordingly.

## References

- [vLLM-Metal README](https://github.com/vllm-project/vllm-metal)
- [vLLM-Metal supported models](https://raw.githubusercontent.com/vllm-project/vllm-metal/main/docs/supported_models.md)
- [Qwen3.6-35B-A3B 4-bit DWQ model card](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ)
- [vLLM tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/)
