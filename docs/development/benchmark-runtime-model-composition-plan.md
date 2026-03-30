# Benchmark Refactor Plan: Runtime/Model Composition + Handlers + MLX Server-Side Reasoning

## Summary

Refactor benchmark into:

1. Per-runtime/per-model composed config from `models.yaml`
2. Runtime handlers for request building, cache flow, stream parsing, and usage extraction
3. Server-side MLX reasoning support that maps request reasoning controls into model-specific chat-template controls (on-par intent with Ollama, capability-gated per model)

Locked decisions:

- Ollama usage path: native `/api/chat`
- Config composition source: `configs/models.yaml`
- Parser strategy: PyYAML
- Reasoning capability source: explicit per-model config
- Unsupported reasoning policy: fail fast
- CLI/output compatibility: breaking cleanup

## Implementation Changes

### 1) Config composition layer

- Replace custom `models.yaml` parser with `yaml.safe_load` and strict validation.
- Extend model schema with optional benchmark block, including reasoning capability + format, cache strategy, transport, stream, usage mode.
- Compose resolved spec per run with precedence:
  1. CLI
  2. model runtime override
  3. model defaults
  4. runtime defaults
- Add explicit composed fields used by handlers:
  - `reasoning.{requested,effective,format,supported}`
  - `cache.mode`
  - `transport.mode`
  - `stream.enabled`
  - `usage.mode`
  - endpoint/timeouts
  - model runtime options (for example `num_ctx`)

### 2) Runtime handler split

- Introduce handler interface and move runtime conditionals out of `runner.py`.
- MLX handler: managed lifecycle + prefill cache endpoints + MLX perf parsing.
- Ollama handler: native `/api/chat` request/stream/usage parsing from official usage fields (`*_duration`, `*_count`) with ns-to-sec normalization.
- Runner only orchestrates dataset cases and delegates prepare/request/parse/teardown to handlers.

### 3) MLX server-side reasoning support

- Update both MLX servers’ `ChatRequest` schema to accept reasoning controls (`reasoning`, `reasoning_effort`, and normalized internal field).
- Add request-template application path in server prompt construction: derive `chat_template_kwargs` from reasoning config and apply through tokenizer template call.
- Implement capability-gated mapping by model config:
  - format type example: `qwen_enable_thinking`
  - `off` -> `enable_thinking=False`
  - `on` -> `enable_thinking=True`
- If model is configured as reasoning-unsupported and request asks reasoning `on`, return clear 4xx error.
- Keep format pluggable via config so non-Qwen models can define their own template mapping.

### 4) Thinking/reasoning parity policy

- Benchmark API keeps one abstract control: `reasoning_mode`.
- Handler maps control to runtime-specific wire format:
  - Ollama native: `think`
  - MLX: server reasoning fields converted to template kwargs
- Persist requested/effective reasoning state and source in output.

### 5) Cache, streaming, and metrics normalization

- Normalize metric sections: client timing, server usage, throughput, provenance.
- Always compute TTFT from first streamed token timestamp when streaming is enabled.
- Parse server-side usage independently of stream text parsing.
- For Ollama native usage, ingest and convert:
  - `total_duration`, `load_duration`, `prompt_eval_duration`, `eval_duration`
  - `prompt_eval_count`, `eval_count`
- Apply source tagging per metric (`server`, `client_derived`, `estimated`).

## Public Interfaces (Breaking)

### A) `configs/models.yaml`

- Add structured per-model `bench` configuration:
  - `bench.defaults`
  - `bench.runtime.<runtime>`
  - `bench.reasoning` capability and format definition (required for reasoning-enabled models)

### B) CLI

- Replace legacy toggles with explicit modes:
  - `--reasoning-mode {auto,off,on}`
  - `--cache-mode {auto,prefill,request,none}`
  - `--stream {auto,on,off}`
  - `--usage-mode {auto,server,client,hybrid}`
  - `--transport {auto,ollama-native,openai-compat}`

### C) Output schema

- Replace flat fields with structured sections:
  - `timing.client`
  - `usage.server`
  - `usage.normalized`
  - `reasoning`
  - `cache`
  - `throughput`
  - `sources`
  - `retrieval`
  - `memory`
- Keep summary reports, regenerated from the new schema.

## Test Plan

1. Config composition tests:
   - precedence
   - validation errors
   - required reasoning capability metadata
2. Handler tests:
   - MLX and Ollama request building
   - streaming parse
   - usage extraction
   - ns-to-sec conversion
3. MLX server tests:
   - reasoning field acceptance
   - template kwargs mapping
   - unsupported-model fail-fast behavior
4. Integration smoke:
   - MLX `reasoning=off` and `reasoning=on` with reasoning-capable model show distinct behavior + valid metrics
   - MLX `reasoning=on` with unsupported model returns expected error
   - Ollama native run yields populated normalized usage timing
5. Schema tests:
   - new JSON contract
   - summary generation

## Assumptions

- Ollama usage behavior follows official docs: <https://docs.ollama.com/api/usage>
- Ollama native API is the primary path for usage-rich benchmarking.
- Initial reasoning-template mapping targets configured Qwen-like formats first; additional formats are added through config, not heuristics.
- Benchmark defaults remain `stream=on`; long-context default remains 64k unless overridden.
- No silent disable for unsupported reasoning; explicit failure is required for comparability.
