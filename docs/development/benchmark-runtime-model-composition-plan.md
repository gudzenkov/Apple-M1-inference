# Benchmark Refactor Plan: Runtime/Model Composition + Handlers + MLX/Ollama Metrics Policy

## Summary

Refactor benchmark into:

1. Composed runtime/model spec from split configs (`configs/models.yaml` + `configs/bench.yaml`)
2. Runtime handlers for request building, cache flow, stream parsing, usage extraction, and memory measurement
3. Server-side MLX reasoning controls mapped to chat-template kwargs with capability gating

Locked decisions:

- Ollama usage path: native `/api/chat`
- Config composition sources: `configs/models.yaml` (catalog/capabilities) + `configs/bench.yaml` (runtime benchmark policy)
- Parser strategy: `yaml.safe_load` with strict validation
- Profile model: profile represents `runtime + model-family capability mapping` (extensible for `qwen`, `glm`, `deepseek`, ...)
- Unsupported reasoning policy: fail fast
- `usage_mode` is removed as a public control; metrics strategy is runtime-defined

## Implementation Changes

### 1) Config composition layer

- Replace legacy schema assumptions with split schema:
  - `models.yaml`: model inventory, aliases, runtime placement, family/capabilities
  - `bench.yaml`: runtime benchmark policy + runtime wire options + optional per-model overrides
- Profile abstraction:
  - Profile binds runtime behavior to model-family capability mapping
  - Example profile keys: `mlx/qwen`, `mlx-optiq/qwen`, `ollama/qwen`
- Compose resolved spec per run with precedence:
  1. CLI
  2. bench model override
  3. bench runtime defaults
  4. model profile defaults
- Composed fields used by handlers:
  - `reasoning.{requested,effective,supported,format}`
  - `cache.mode`
  - `transport.mode`
  - `stream.enabled`
  - endpoint/timeouts
  - runtime request options (for example `num_ctx`)

### 2) Runtime handler split

- Introduce handler interface and move runtime conditionals out of `runner.py`.
- MLX handler:
  - managed lifecycle
  - prefill cache endpoints
  - server-authoritative usage/perf parsing
- Ollama handler:
  - native `/api/chat` request/stream/usage parsing
  - server duration/count parsing (`*_duration`, `*_count`) with ns-to-sec normalization
  - client-derived TTFT and RSS memory metrics
- Runner orchestrates dataset cases and delegates prepare/request/parse/teardown to handlers.

### 3) MLX server-side reasoning support

- Update both MLX servers’ `ChatRequest` schema to accept reasoning controls (`reasoning`, `reasoning_effort`, normalized internal field).
- Add request-template application path in server prompt construction:
  - derive `chat_template_kwargs` from resolved reasoning format
  - apply through tokenizer template call
- Capability-gated mapping by profile/model capability:
  - `qwen` format example: `off -> enable_thinking=False`, `on -> enable_thinking=True`
- If model/profile does not support reasoning and request asks `reasoning=on`, return clear 4xx error.

### 4) Thinking/reasoning parity policy

- Benchmark API keeps one abstract control: `reasoning_mode`.
- Handler maps control to runtime-specific wire format:
  - Ollama native: `think`
  - MLX: server reasoning fields -> template kwargs
- Persist requested/effective reasoning state and source in output.

### 5) Cache, streaming, and metrics normalization

- Normalize metric sections: client timing, server usage, throughput, provenance.
- TTFT policy:
  - MLX: use server `perf.ttft_sec` as authoritative server TTFT
  - Ollama: compute TTFT from first streamed token timestamp on client
- Parse server-side usage independently of stream text parsing.
- For Ollama native usage, ingest and convert:
  - `total_duration`, `load_duration`, `prompt_eval_duration`, `eval_duration`
  - `prompt_eval_count`, `eval_count`
- Memory policy:
  - Ollama RSS baseline before load/warmup
  - RSS after load/warmup
  - request-time peak RSS sampling
  - report both absolute peak and delta metrics with provenance
- Apply source tagging per metric (`server`, `client_derived`, `estimated`).

#### Metrics mapping table (current MLX server impl vs Ollama native)

| Normalized output field | MLX source (`/v1/chat/completions`) | Ollama source (`/api/chat`) | Units / conversion | Source tag |
|---|---|---|---|---|
| `usage.server.prompt_tokens` | `usage.prompt_tokens` | `prompt_eval_count` | integer tokens | `server` |
| `usage.server.completion_tokens` | `usage.completion_tokens` | `eval_count` | integer tokens | `server` |
| `usage.server.total_tokens` | `usage.total_tokens` | `prompt_eval_count + eval_count` | integer tokens | `server` |
| `timing.server.total_time_sec` | `perf.total_time_sec` | `total_duration` | Ollama: ns -> sec (`/ 1e9`) | `server` |
| `timing.server.ttft_sec` | `perf.ttft_sec` | n/a | sec | `server` |
| `timing.client.ttft_sec` | optional validation metric | first streamed token timestamp (client) | sec from client clock delta | `client_derived` |
| `timing.server.load_time_sec` | n/a | `load_duration` | Ollama: ns -> sec (`/ 1e9`) | `server` |
| `timing.server.prompt_eval_sec` | n/a | `prompt_eval_duration` | Ollama: ns -> sec (`/ 1e9`) | `server` |
| `timing.server.eval_sec` | n/a | `eval_duration` | Ollama: ns -> sec (`/ 1e9`) | `server` |
| `throughput.prompt_tps` | `perf.prompt_tps` | `prompt_eval_count / prompt_eval_duration_sec` | derived | `server` (derived) |
| `throughput.generation_tps` | `perf.generation_tps` | `eval_count / eval_duration_sec` | derived | `server` (derived) |
| `memory.peak_gb` | `perf.peak_memory_gb` | RSS peak sampling around request | GB | MLX:`server`, Ollama:`client_derived` |
| `memory.delta_gb` | n/a | `peak_rss_gb - baseline_rss_gb` | GB | `client_derived` |

## Public Interfaces (Breaking)

### A) `configs/models.yaml`

- Model catalog + capabilities only.
- Runtime+family capability profile mapping is explicit and reusable.
- No benchmark transport/cache/usage policy fields in `models.yaml`.

### B) `configs/bench.yaml`

- Runtime benchmark policy and runtime wire options.
- Optional per-model benchmark overrides.
- No duplicated capability metadata that already lives in model/profile config.

### C) CLI

- Explicit controls:
  - `--reasoning-mode {auto,off,on}`
  - `--cache-mode {auto,prefill,request,none}`
  - `--stream {auto,on,off}`
  - `--transport {auto,ollama-native,openai-compat}`
- `--usage-mode` removed.

### D) Output schema

- Replace flat fields with structured sections:
  - `timing.client`
  - `timing.server`
  - `usage.server`
  - `usage.normalized`
  - `reasoning`
  - `cache`
  - `throughput`
  - `sources`
  - `retrieval`
  - `memory`
- Keep summary reports, regenerated from new schema.

## Test Plan

1. Config composition tests:
   - precedence
   - validation errors
   - runtime+family profile resolution
2. Handler tests:
   - MLX and Ollama request building
   - streaming parse
   - usage extraction
   - ns-to-sec conversion
   - Ollama RSS baseline/peak/delta calculation
3. MLX server tests:
   - reasoning field acceptance
   - template kwargs mapping
   - unsupported-model fail-fast behavior
4. Integration smoke:
   - MLX `reasoning=off` and `reasoning=on` with reasoning-capable model show distinct behavior + valid metrics
   - MLX `reasoning=on` with unsupported model returns expected error
   - Ollama native run yields populated server usage timing + client TTFT/memory
5. Schema tests:
   - new JSON contract
   - summary generation

## Assumptions

- Ollama usage behavior follows official docs: <https://docs.ollama.com/api/usage>
- Ollama native API remains primary path for usage-rich benchmarking.
- Initial reasoning-template mapping targets Qwen; additional formats are added by profile/config, not heuristics.
- Benchmark default stream remains `on`; long-context default remains 64k unless overridden.
- No silent disable for unsupported reasoning; explicit failure is required for comparability.
