# llama.cpp Embedding Server

Use llama.cpp for Qwen3 embeddings and as a GGUF fallback. Keep embeddings on a separate process from the chat model.

## Qwen3 Embeddings

Recommended default for running alongside Qwen3.6-35B-A3B:

- Model repo: `Qwen/Qwen3-Embedding-0.6B-GGUF`
- File: `Qwen3-Embedding-0.6B-Q8_0.gguf`
- Served model name: `qwen3-embedding`
- Endpoint: `http://127.0.0.1:8001/v1/embeddings`

Use `Qwen/Qwen3-Embedding-4B-GGUF` with `Qwen3-Embedding-4B-Q4_K_M.gguf` when embedding quality matters more than keeping memory free for the chat model. Reserve the 8B embedding model for embedding-only or memory-rich sessions.

Start on the macOS host:

```bash
cd "$(git rev-parse --show-toplevel)"
scripts/start-qwen3-embedding-llama.cpp.sh
```

Equivalent explicit command:

```bash
llama-server \
  --hf-repo Qwen/Qwen3-Embedding-0.6B-GGUF \
  --hf-file Qwen3-Embedding-0.6B-Q8_0.gguf \
  --alias qwen3-embedding \
  --host 0.0.0.0 \
  --port 8001 \
  --embedding \
  --pooling last \
  -ngl 99 \
  -c 32768
```

Host smoke test:

```bash
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding","input":"test text"}'
```

Devcontainer smoke test:

```bash
curl -fsS http://host.docker.internal:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding","input":"test text"}'
```

## GGUF Chat Fallback

If vLLM-Metal is broken for a given Qwen3.5 build, llama.cpp can serve the GGUF quantization:

```bash
llama-server \
  --hf-repo unsloth/Qwen3.5-35B-A3B-GGUF \
  --hf-file Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --alias qwen3.5-35b-a3b-gguf \
  --host 0.0.0.0 \
  --port 8010 \
  --ctx-size 32768 \
  --jinja \
  --cache-prompt \
  -ngl 99
```

This is a fallback, not the preferred agent endpoint. Tool-call and reasoning extraction are less predictable than the vLLM-Metal path for Qwen3.5.

## Troubleshooting

- `llama-server: command not found`: install llama.cpp, for example `brew install llama.cpp`.
- Empty or wrong embeddings: confirm `--embedding --pooling last` are present.
- Port conflict on `8001`: set `QWEN_EMBEDDING_PORT=8011` and update `OPENAI_EMBEDDINGS_BASE_URL`.
- Devcontainer cannot connect: use `host.docker.internal`, not `127.0.0.1`.

## References

- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Qwen llama.cpp docs](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
- [Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)
- [Qwen3-Embedding-4B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF)
- [Qwen3-Embedding-8B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF)
