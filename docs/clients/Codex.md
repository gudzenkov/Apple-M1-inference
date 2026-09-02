# Codex with Local vLLM-Metal

Codex uses the OpenAI Responses API. Point it at the Qwen3.8 server's `/v1` base URL; Codex adds `/responses` itself.

## Prerequisites

1. Start Qwen3.8-27B with the [vLLM-Metal guide](../servers/VLLM-Metal.md).
2. Confirm that `/v1/models` returns the served name `qwen3.8-27b`.
3. Set any non-empty API key. A server started without `--api-key` does not validate its value.

From the devcontainer:

```bash
export OPENAI_API_KEY=local
curl --connect-timeout 2 --max-time 10 \
  http://host.docker.internal:8000/v1/models | jq
```

Use `http://127.0.0.1:8000/v1` instead when Codex runs directly on the macOS host.

## Configuration

Put the provider in the Codex user config. Inside the devcontainer this is `/home/vscode/.codex/config.toml`, normally written as `~/.codex/config.toml`:

```toml
model_provider = "vllm"
model = "qwen3.8-27b"
model_context_window = 262144
model_auto_compact_token_limit = 230000
model_reasoning_effort = "low"
model_reasoning_summary = "none"
model_supports_reasoning_summaries = false

[model_providers.vllm]
name = "Local vLLM-Metal"
base_url = "http://host.docker.internal:8000/v1"
env_key = "OPENAI_API_KEY"
requires_openai_auth = false
wire_api = "responses"
supports_websockets = false
supports_standalone_web_search = false
request_max_retries = 0
stream_max_retries = 0
```

Do not rely on a repository `.codex/config.toml` for `model_provider` or `[model_providers.*]`. Codex ignores those provider settings at project scope; keep them in the user config.

The context values must match the running server:

| vLLM `--max-model-len` | `model_context_window` | Suggested compaction limit |
| ---: | ---: | ---: |
| `262144` | `262144` | `230000` |
| `256000` | `256000` | `224000` |
| `250000` | `250000` | `218000` |

The compaction limits retain about 32k tokens for tool results, reasoning, and the final answer. Do not use `250000` as the compaction threshold on a 256k server; the remaining 6k tokens are too little for a typical coding turn.

Start with `model_reasoning_effort = "low"`. Raise it only after the smoke test is stable; `xhigh` can make even a small first turn expensive on a local model.

Zero retries make initial diagnosis explicit and avoid automatically submitting more work to a busy local server. Increase them later if transient connection failures are more important than predictable resource use.

## Responses API Smoke Test

Test the exact API Codex uses before starting Codex:

```bash
OPENAI_BASE_URL=http://host.docker.internal:8000/v1

curl --fail-with-body --show-error --silent \
  --connect-timeout 2 --max-time 120 \
  "$OPENAI_BASE_URL/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
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

This deliberately disables thinking and caps output. It verifies routing and response compatibility without testing Codex's tool loop.

Check the config parser without sending a model request:

```bash
codex --strict-config --version
```

Then start an interactive session with `codex --strict-config` and use a bounded first prompt such as `Reply with OK. Do not use tools.` Do not enable `--search`; the local provider does not implement OpenAI's hosted web-search tool.

## A Turn Keeps Using the GPU

Cancel the originating interactive Codex turn with `Ctrl-C`. A client timeout or disconnect is not proof that the backend request stopped. Check the server after cancellation:

```bash
curl -fsS http://host.docker.internal:8000/metrics \
  | rg 'num_requests_running|generation_tokens_total'
```

If a request remains running while the generated-token count does not change, stop and restart the vLLM server before submitting another turn. Do not stack another smoke test onto a server that still reports an active request.

## References

- [Codex configuration reference](https://developers.openai.com/codex/config-reference/)
- [OpenAI Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
- [Qwen3.8 vLLM-Metal server](../servers/VLLM-Metal.md)
