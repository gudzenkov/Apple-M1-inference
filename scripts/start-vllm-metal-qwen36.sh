#!/usr/bin/env bash
set -euo pipefail

MODEL_ID_FROM_ENV="${MODEL_ID:-}"
MODEL_ID="${QWEN_CHAT_MODEL_ID:-${MODEL_ID_FROM_ENV:-mlx-community/Qwen3.6-35B-A3B-4bit-DWQ}}"
if [ -n "${QWEN_CHAT_SERVED_MODEL:-}" ]; then
	SERVED_MODEL="${QWEN_CHAT_SERVED_MODEL}"
elif [ -n "${MODEL_ID_FROM_ENV}" ]; then
	SERVED_MODEL="${MODEL_ID}"
else
	SERVED_MODEL="qwen3.6-35b-a3b"
fi
HOST="${HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-${PORT:-8000}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${MAX_NUM_SEQS:-1}}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS:-512}}"
LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_METAL_MEMORY_FRACTION="${VLLM_METAL_MEMORY_FRACTION:-0.50}"
export VLLM_METAL_MULTIMODAL_MODE="${VLLM_METAL_MULTIMODAL_MODE:-text-only-compat}"

if command -v vllm >/dev/null 2>&1; then
	VLLM_BIN="$(command -v vllm)"
elif [ -x "${HOME}/.venv-vllm-metal/bin/vllm" ]; then
	VLLM_BIN="${HOME}/.venv-vllm-metal/bin/vllm"
else
	printf 'vllm not found. Install vLLM-Metal on macOS host first:\n' >&2
	printf '  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash\n' >&2
	exit 127
fi

args=(
	serve "${MODEL_ID}"
	--host "${HOST}"
	--port "${PORT}"
	--served-model-name "${SERVED_MODEL}"
	--max-model-len "${MAX_MODEL_LEN}"
	--max-num-seqs "${MAX_NUM_SEQS}"
	--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
	--generation-config vllm
	--reasoning-parser qwen3
	--enable-auto-tool-choice
	--tool-call-parser qwen3_coder
)

if [ "${LANGUAGE_MODEL_ONLY}" = "true" ]; then
	args+=(--language-model-only)
fi

exec "${VLLM_BIN}" "${args[@]}" "$@"
