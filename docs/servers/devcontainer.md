# Mac Host for Devcontainers

Run Metal inference on macOS host, invoke it from the docker container over the VM boundary.

This runbook covers devcontainers running in:
- OrbStack (Linux VM)
- Docker Desktop (Linux VM)

## 1. Start inference servers on macOS host

Recommended Qwen3.6 agent setup:

```bash
cd "$(git rev-parse --show-toplevel)"

# Chat/tools/thinking on :8000
scripts/start-vllm-metal-qwen36.sh
```

In a second terminal:

```bash
cd "$(git rev-parse --show-toplevel)"

# Qwen3 0.6B embeddings on :8001
scripts/start-qwen3-embedding-llama.cpp.sh
```

Legacy MLX runtimes remain available:

```bash
cd "$(git rev-parse --show-toplevel)"
uv sync

# Standard MLX server on :8000
uv run mlx-openai-server start

# Or TurboQuant Optiq server on :8080
uv run mlx-openai-optiq-server start
```

## 2. Verify host endpoints before entering container

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8001/v1/models
curl -s http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding","input":"test"}'
```

Use the port for the runtime you started:
- `8000` -> `vLLM-Metal` chat/tools/thinking or legacy `mlx-openai-server`
- `8001` -> `llama.cpp` Qwen3 embeddings
- `8080` -> legacy `mlx-openai-optiq-server`; this port is often used by other local tools, so probe it before relying on it

## 3. Expose host endpoint to devcontainer

For both OrbStack and Docker Desktop devcontainers, use `host.docker.internal` from inside the container.

Example `devcontainer.json`:

```json
{
  "name": "localfirst",
  "containerEnv": {
    "OPENAI_BASE_URL": "http://host.docker.internal:8000/v1",
    "OPENAI_API_KEY": "local",
    "OPENAI_EMBEDDINGS_BASE_URL": "http://host.docker.internal:8001/v1",
    "QWEN_CHAT_MODEL": "qwen3.6-35b-a3b",
    "QWEN_EMBEDDING_MODEL": "qwen3-embedding"
  },
  "remoteEnv": {
    "OPENAI_BASE_URL": "http://host.docker.internal:8000/v1",
    "OPENAI_API_KEY": "local",
    "OPENAI_EMBEDDINGS_BASE_URL": "http://host.docker.internal:8001/v1"
  }
}
```

If your client needs separate chat and embedding providers, use `OPENAI_BASE_URL` for chat and `OPENAI_EMBEDDINGS_BASE_URL` for embeddings. Many SDKs do not have a standard embedding base-url variable, so wire it explicitly in client config when needed.

## 4. Validate connectivity from inside the devcontainer

```bash
curl -s http://host.docker.internal:8000/v1/models
curl -s http://host.docker.internal:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding","input":"test"}'
```

Both should succeed for the recommended split-server setup.

## 5. OrbStack

#### Bridge networking
Containers in OrbStack have domain names at `container-name.orb.local` or `service.project.orb.local` for Compose with zero configuration or port numbers required.

You can also use the `host.docker.internal` domain to connect to a server running on Mac.

#### Host networking
OrbStack supports host networking, allowing you to avoid having to deal with port forwarding.
Host networking, or `--net host`, allows containers to inherit the host's network namespace instead of being an independent host on a bridge network
`localhost` also works in the other direction, so you can connect directly to servers running on macOS instead of using `host.docker.internal`

## 6. VS Code Dev Containers caveat

Limitation:
- VS Code Dev Containers uses internal port forwarding for `vscode-server`.
- With host networking, forwarded container localhost ports can collide with host localhost routing.
- This can trigger repeated forwarding loops in logs (for example `Port forwarding ... > 44527 > 44527`).

Recommended default for VS Code Dev Containers:
- Keep default container networking (no host network mode).
- Use `host.docker.internal` for host services from inside container.
- Keep `LLM_BASE_URL` as `http://host.docker.internal:<port>/v1`.

## Troubleshooting

- `Connection refused` from container:
  - Server is not running on host. Re-run host smoke tests first.
  - Wrong port or runtime mismatch.
- `404` on requests:
  - Base URL missing `/v1`.
  - Client calling `/v1/completions` instead of `/v1/chat/completions`.
  - Embedding client pointed at the chat endpoint instead of `:8001`.
- DNS issue for `host.docker.internal`:
  - You are probably not on OrbStack/Docker Desktop VM networking path.
  - Add explicit host mapping only if needed by your setup.
- Repeating `Port forwarding ... > 44527 > 44527` logs in Dev Containers:
  - Check `devcontainer.json` and remove `runArgs: ["--network=host"]`.
  - Rebuild/reopen container after config change.

## References

- [MLX server runbook](./MLX.md)
- [vLLM-Metal Qwen3.6 server](./VLLM-Metal.md)
- [llama.cpp embedding server](./llama.cpp.md)
- [OrbStack host networking docs](https://docs.orbstack.dev/docker/host-networking)
- [Docker Desktop networking docs](https://docs.docker.com/desktop/features/networking/networking-how-tos/)
