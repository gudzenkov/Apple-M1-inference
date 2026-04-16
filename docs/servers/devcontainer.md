# Mac Host for Devcontainers

Run MLX/Metal inference on macOS host, invoke it from the docker container over the VM boundary.

This runbook covers devcontainers running in:
- OrbStack (Linux VM)
- Docker Desktop (Linux VM)

## 1. Start the inference server on macOS host

Pick one MLX runtime and keep it running on host.

```bash
cd "$(git rev-parse --show-toplevel)"
uv sync

# Standard MLX server on :8000
uv run mlx-openai-server start

# Or TurboQuant Optiq server on :8080
uv run mlx-openai-optiq-server start
```

## 2. Verify host endpoint before entering container

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8080/v1/models
```

Use the port for the runtime you started:
- `8000` -> `mlx-openai-server`
- `8080` -> `mlx-openai-optiq-server`

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
```

One of these should return the model list from the runtime you started on host.

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
  - Server is not running on host. Re-run `uv run mlx-openai-server status` or `uv run mlx-openai-optiq-server status`.
  - Wrong port or runtime mismatch.
- `404` on requests:
  - Base URL missing `/v1`.
  - Client calling `/v1/completions` instead of `/v1/chat/completions`.
- DNS issue for `host.docker.internal`:
  - You are probably not on OrbStack/Docker Desktop VM networking path.
  - Add explicit host mapping only if needed by your setup.
- Repeating `Port forwarding ... > 44527 > 44527` logs in Dev Containers:
  - Check `devcontainer.json` and remove `runArgs: ["--network=host"]`.
  - Rebuild/reopen container after config change.

## References

- [MLX server runbook](./MLX.md)
- [OrbStack host networking docs](https://docs.orbstack.dev/docker/host-networking)
- [Docker Desktop networking docs](https://docs.docker.com/desktop/features/networking/networking-how-tos/)
