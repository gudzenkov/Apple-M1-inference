# vLLM-Metal Qwen3.8-27B Server

Use this path for `Qwen3.8-27B` when a client needs an OpenAI-compatible chat endpoint, Qwen reasoning output, and structured tool calls.

vLLM-Metal currently serves this checkpoint most reliably as a text-only model on Apple Silicon. Use `--language-model-only`; do not expect local vision-input parity from this setup.

## Recommendation

- Model: `mlx-community/Qwen3.8-27B-4bit`.
- Context: `262144` tokens is the model's native 256 Ki-token window. It needs no YaRN or other RoPE override.
- KV cache: use the quality-oriented TurboQuant `q8_0`/`q4_0` configuration below to retain the native window without consuming the entire recommended Metal working set.
- Concurrency: keep `--max-num-seqs 1` for a single-user coding server. Qwen3.8 has hybrid linear/full attention, and each concurrent sequence also needs linear-attention state.
- Embeddings: run a separate llama.cpp server with `Qwen/Qwen3-Embedding-0.6B-GGUF`; use 4B or 8B only when memory allows.
- Devcontainer access: use `host.docker.internal`, not container `localhost`, for the host-side macOS server.

TurboQuant changes KV-cache precision and can affect quality. Validate it on representative long-context work. The uncompressed-KV alternative is documented below.

## Install

Install the latest vLLM-Metal build on the macOS host, not inside the devcontainer:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
vllm --version
python -c 'import importlib.metadata as m; print(m.version("vllm-metal"))'
```

Current builds recognize the Qwen3.8 conditional-generation wrapper in automatic text-backbone mode. Do not apply the old Qwen3.6 compatibility patch or set the removed `text-only-compat` mode.

## Serve the Native 256 Ki Context

This is the recommended 262,144-token, no-RoPE configuration:

```bash
source ~/.venv-vllm-metal/bin/activate

MODEL_ID=mlx-community/Qwen3.8-27B-4bit
SERVED_MODEL=qwen3.8-27b

VLLM_METAL_MULTIMODAL_MODE=auto \
VLLM_METAL_MEMORY_FRACTION=auto \
vllm serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$SERVED_MODEL" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --language-model-only \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --additional-config '{"turboquant": true, "k_quant": "q8_0", "v_quant": "q4_0"}'
```

The startup log must report all of the following before this setup is considered active:

- `max_model_len=262144` or an equivalent engine summary.
- `turboquant=True`, `k_quant=q8_0` in the paged-attention allocation line.
- A GPU KV-cache capacity of at least 262,144 tokens.
- Maximum concurrency of at least `1.00x` for 262,144 tokens per request.

Do not add `VLLM_ALLOW_LONG_MAX_MODEL_LEN`, `--hf-overrides`, `rope_parameters`, or YaRN for this configuration. Those controls are for extending the model beyond its native 262,144-token window.

## Uncompressed KV on the 64 GiB M1 Max

The measured uncompressed-KV startup result was:

```text
262144 tokens need 16.32 GiB KV
16.03 GiB KV is available at VLLM_METAL_MEMORY_FRACTION=1.0
estimated maximum model length: 257152
```

Therefore, exact native 262,144 does not fit with uncompressed KV on this machine. Decimal 256k does fit, but only narrowly. To use it, remove the `--additional-config` line from the recommended command and change these settings:

```bash
VLLM_METAL_MEMORY_FRACTION=1.0

--max-model-len 256000
```

Also remove `--gpu-memory-utilization 0.92`: a numeric `VLLM_METAL_MEMORY_FRACTION` takes precedence over it.

`256000` is 1,152 tokens below the engine's measured limit and only 6,144 tokens below the model's native maximum. It leaves little KV-allocation margin. `250000` is the safer uncompressed setting if other Metal workloads remain active.

On this M1 Max, MLX reports a 48 GiB `max_recommended_working_set_size` out of 64 GiB physical RAM. `VLLM_METAL_MEMORY_FRACTION=1.0` targets that 48 GiB Metal limit, not all physical RAM. It should not by itself reserve all memory from macOS, but it leaves no headroom inside MLX's recommended working set. Close other GPU-heavy applications and watch memory pressure during a full-context test.

Inspect the limits on the current host with:

```bash
python - <<'PY'
import mlx.core as mx

info = mx.device_info()
for key in ("memory_size", "max_recommended_working_set_size"):
    print(f"{key}: {info[key] / 1024**3:.2f} GiB")
PY
```

`--max-model-len` covers the complete sequence: prompt, retained reasoning, and generated answer. A client cannot send a 256k prompt and then generate beyond that same limit.

## Metal Memory Controls

On current vLLM-Metal builds:

- `VLLM_METAL_MEMORY_FRACTION=auto` delegates to `--gpu-memory-utilization`; vLLM currently defaults it to `0.92`.
- A numeric `VLLM_METAL_MEMORY_FRACTION` overrides `--gpu-memory-utilization`.
- The fraction applies to MLX's `max_recommended_working_set_size`, not total physical RAM.
- `--max-num-seqs 1` limits concurrency and reduces hybrid linear-attention state reservations; it does not shrink the per-token KV needed by one full-length request.

Do not set conflicting values for the environment variable and CLI flag. If an installed build logs that `--gpu-memory-utilization` is ignored, update vLLM-Metal; older builds did not connect that flag to the Metal paged-KV planner.

## Host Smoke Tests

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1

curl -fsS "$OPENAI_BASE_URL/models" | jq

curl -fsS "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Return exactly OK."}],
    "max_tokens": 16,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false},
    "stream": false
  }' | jq
```

Responses API smoke test:

```bash
curl -fsS "$OPENAI_BASE_URL/responses" \
  --connect-timeout 2 \
  --max-time 120 \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "input": "Return exactly OK.",
    "max_output_tokens": 16,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false},
    "include_reasoning": false,
    "stream": false
  }' | jq
```

The timeout bounds the client wait, not necessarily the backend request lifetime. If it expires, inspect `/metrics` for `num_requests_running` and `generation_tokens_total` before sending another test.

Tool-call smoke test:

```bash
curl -fsS "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
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
  }' | jq
```

The server emits tool calls. The client still executes each tool and sends its result back to the model.

## Devcontainer Access

Inside a Docker Desktop or OrbStack devcontainer:

```bash
export OPENAI_BASE_URL=http://host.docker.internal:8000/v1
export OPENAI_API_KEY=local
export QWEN_CHAT_MODEL=qwen3.8-27b

curl -fsS "$OPENAI_BASE_URL/models"
```

Use the model name returned by `/v1/models`.

## Troubleshooting

- `vllm: command not found`: activate `~/.venv-vllm-metal/bin/activate` or reinstall vLLM-Metal.
- Missing vision-tower fields during text-only loading: update vLLM-Metal, keep `VLLM_METAL_MULTIMODAL_MODE=auto`, and pass `--language-model-only`.
- Native context still fails with TurboQuant: confirm the startup log says `turboquant=True`, keep `--max-num-seqs 1`, and check that no numeric `VLLM_METAL_MEMORY_FRACTION` is overriding `--gpu-memory-utilization`.
- Uncompressed 262,144 fails with the 16.32/16.03 GiB error: this is expected on the measured 64 GiB M1 Max; use TurboQuant or `--max-model-len 256000`.
- Tool calls arrive as plain text: confirm `--enable-auto-tool-choice --tool-call-parser qwen3_coder` is present.
- No separate reasoning output: confirm `--reasoning-parser qwen3` is present and the client preserves reasoning fields.
- Port conflict on `8000`: choose another `--port` value and update `OPENAI_BASE_URL`.

## References

- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.8-27B vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [MLX 4-bit model card](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
- [vLLM-Metal](https://github.com/vllm-project/vllm-metal)
- [vLLM tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/)
