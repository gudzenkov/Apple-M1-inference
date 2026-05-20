# Claude Code Router (CCR) Notes

Claude Code Router lets you define custom routes for Claude-style models. Use it to point agentic workflows at the local vLLM-Metal OpenAI-compatible endpoint instead of a hosted service.

## High-level steps

1. Open CCR (usually under the Claude workspace or the `router` section of the web app).
2. Create a new route and choose the OpenAI-compatible transport.
3. For vLLM-Metal on the host, set the base URL to `http://localhost:8000/v1`; from a devcontainer, use `http://host.docker.internal:8000/v1`.
4. Add any required headers (e.g., `Authorization: Bearer local`) if your client enforces an API key; the local server accepts any non-empty token.
5. Set the default model ID to `qwen3.6-35b-a3b`.
6. Save the route and select it whenever you need Claude to call the local runtime.

## Validation

- Use `curl http://localhost:<port>/v1/models` to verify the endpoint responds before enabling the route.
- Ask CCR to run a quick prompt and inspect the response; the router will surface request failures if the base URL is unreachable.

## Tips

- Keep a separate legacy route only if you still switch to MLX/Optiq experiments.
- When debugging, check the CCR route logs along with the vLLM-Metal server terminal.
