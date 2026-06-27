# LM Studio Local Server

Use LM Studio when you want a manual local OpenAI-compatible server without the vLLM-Metal or llama.cpp command flow. It is a good option for quick checks and for models already downloaded in the LM Studio UI.

Hard truth: LM Studio is convenient, but reproducibility depends on the exact model file loaded in the UI. Always verify `/v1/models` before wiring clients to it.

## Recommended Use

- Chat: load a local Qwen chat model and expose it on `http://127.0.0.1:1234/v1`.
- Embeddings: load a local Qwen embedding model, such as Qwen3 Embedding 0.6B, and expose `/v1/embeddings` on the same base URL.
- Devcontainer access: use `http://host.docker.internal:1234/v1`.

Keep only one role per LM Studio process when possible. For the split setup, run chat through vLLM-Metal on `:8000` and embeddings through either llama.cpp on `:8001` or LM Studio on `:1234`.

## Start

1. Open LM Studio on macOS.
2. Download or select the local model.
3. Open the local server panel.
4. Start the OpenAI-compatible server on port `1234`.
5. Confirm the served model name from `/v1/models`.

## Host Smoke Tests

```bash
curl -fsS http://127.0.0.1:1234/v1/models
```

Chat model test:

```bash
curl -fsS http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-from-/v1/models>",
    "messages": [{"role": "user", "content": "Return exactly OK."}],
    "max_tokens": 16,
    "temperature": 0,
    "stream": false
  }'
```

Embedding model test:

```bash
curl -fsS http://127.0.0.1:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-from-/v1/models>",
    "input": "test text"
  }'
```

## Devcontainer Access

```bash
export OPENAI_BASE_URL=http://host.docker.internal:1234/v1
export OPENAI_API_KEY=local

curl -fsS "$OPENAI_BASE_URL/models"
```

For embeddings-only use:

```bash
export OPENAI_EMBEDDINGS_BASE_URL=http://host.docker.internal:1234/v1
export QWEN_EMBEDDING_MODEL="<model-from-/v1/models>"
```

## Troubleshooting

- `Connection refused`: the LM Studio server is not running or the port is different.
- `404`: the client base URL is missing `/v1`.
- `model not found`: use the exact ID returned by `/v1/models`.
- Empty or invalid embeddings: confirm the loaded model is an embedding model, not a chat model.
