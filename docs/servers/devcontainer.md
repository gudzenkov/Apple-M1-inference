# Mac Host for Devcontainers

Run MLX/Metal inference on macOS host, invoke it from the docker container over the VM boundary.

This runbook covers devcontainers running in:
- OrbStack (Linux VM)
- Docker Desktop (Linux VM)

## 1. Start the inference server on macOS host

Pick one runtime and keep it running on host.

### Option A: MLX / MLX-Optiq (recommended in this repo)
```bash
cd "$(git rev-parse --show-toplevel)"
uv sync

# Standard MLX server on :8000
uv run mlx-openai-server start

# Or TurboQuant Optiq server on :8080
uv run mlx-openai-optiq-server start
```

### Option B: Ollama
```bash
brew install ollama
brew services start ollama
ollama pull <model>
```

## 2. Verify host endpoint before entering container

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:11434/v1/models
```

Use the port for the runtime you started:
- `8000` -> `mlx-openai-server`
- `8080` -> `mlx-openai-optiq-server`
- `11434` -> Ollama

## 3. Expose host endpoint to devcontainer

For both OrbStack and Docker Desktop devcontainers, use `host.docker.internal` from inside the container.

Example `devcontainer.json`:

```json
{
  "name": "localfirst",
  "containerEnv": {
    "LLM_BASE_URL": "http://host.docker.internal:8080/v1",
    "LLM_API_KEY": "local"
  },
  "remoteEnv": {
    "LLM_BASE_URL": "http://host.docker.internal:8080/v1",
    "LLM_API_KEY": "local"
  }
}
```

If your client expects OpenAI-style names, map the same value to `OPENAI_BASE_URL` and set `OPENAI_API_KEY=local`.

## 4. Validate connectivity from inside the devcontainer

```bash
curl -s http://host.docker.internal:8080/v1/models
curl -s http://host.docker.internal:8000/v1/models
curl -s http://host.docker.internal:11434/v1/models
```

One of these should return the model list from the runtime you started on host.

## 5. OrbStack-only shortcut (optional)

OrbStack supports host networking for containers (`--net host`). With that mode, container `localhost` can talk to host `localhost` directly.

Example `devcontainer.json` override for Orb only:

```json
{
  "runArgs": ["--network=host"],
  "containerEnv": {
    "LLM_BASE_URL": "http://localhost:8080/v1",
    "LLM_API_KEY": "local"
  }
}
```

Do not use this as a cross-provider default. `host.docker.internal` is the safest shared path across OrbStack and Docker Desktop.

## Troubleshooting

- `Connection refused` from container:
  - Server is not running on host. Re-run `uv run mlx-openai-server status` or `uv run mlx-openai-optiq-server status`.
  - Wrong port or runtime mismatch.
- `404` on requests:
  - Base URL missing `/v1`.
  - Client calling `/v1/completions` instead of `/v1/chat/completions`.
- DNS issue for `host.docker.internal`:
  - You are probably not on OrbStack/Docker Desktop VM networking path.
  - Add explicit host mapping only if needed by your setup.

## References

- [MLX server runbook](./MLX.md)
- [Ollama runbook](./ollama.md)
- [OrbStack host networking docs](https://docs.orbstack.dev/docker/host-networking)
- [Docker Desktop networking docs](https://docs.docker.com/desktop/features/networking/networking-how-tos/)
