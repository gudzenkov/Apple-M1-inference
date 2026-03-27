# Claude Code Router (CCR) Notes

Claude Code Router lets you define custom routes for Claude-style models. Use it to point agentic workflows at the local MLX or Ollama OpenAI-compatible endpoints instead of the hosted Claude service.

## High-level steps

1. Open CCR (usually under the Claude workspace or the `router` section of the web app).
2. Create a new route and choose the OpenAI-compatible transport.
3. For MLX, set the base URL to `http://localhost:8000/v1`; for the TurboQuant Optiq server use `http://localhost:8080/v1`. For Ollama, point to `http://localhost:11434/v1`.
4. Add any required headers (e.g., `Authorization: Bearer local`) if your client enforces an API key; MLX/Ollama accept any non-empty token.
5. Set the default model ID to one of the Qwen models (e.g., `mlx-community/Qwen3.5-9B-OptiQ-4bit` or `qwen3.5:9b`).
6. Save the route and select it whenever you need Claude to call the local runtime.

## Validation

- Use `curl http://localhost:<port>/v1/models` to verify the endpoint responds before enabling the route.
- Ask CCR to run a quick prompt and inspect the response; the router will surface request failures if the base URL is unreachable.

## Tips

- Keep multiple routes if you switch between the TurboQuant and standard MLX services.
- When debugging, check the CCR route logs along with the local server logs at `logs/mlx-server.log`, `logs/mlx-optiq-server.log`, or `ollama serve` output.
