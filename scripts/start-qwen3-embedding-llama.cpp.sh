#!/usr/bin/env bash
set -euo pipefail

HF_REPO="${QWEN_EMBEDDING_HF_REPO:-Qwen/Qwen3-Embedding-0.6B-GGUF}"
GGUF_FILE="${QWEN_EMBEDDING_GGUF_FILE:-Qwen3-Embedding-0.6B-Q8_0.gguf}"
SERVED_MODEL="${QWEN_EMBEDDING_MODEL:-qwen3-embedding}"
HOST="${HOST:-0.0.0.0}"
PORT="${QWEN_EMBEDDING_PORT:-${PORT:-8001}}"
CONTEXT_SIZE="${QWEN_EMBEDDING_CONTEXT:-32768}"
GPU_LAYERS="${LLAMA_CPP_GPU_LAYERS:-99}"

if ! command -v llama-server >/dev/null 2>&1; then
	printf 'llama-server not found. Install llama.cpp on macOS host first, for example:\n' >&2
	printf '  brew install llama.cpp\n' >&2
	exit 127
fi

exec llama-server \
	--hf-repo "${HF_REPO}" \
	--hf-file "${GGUF_FILE}" \
	--alias "${SERVED_MODEL}" \
	--host "${HOST}" \
	--port "${PORT}" \
	--embedding \
	--pooling last \
	-ngl "${GPU_LAYERS}" \
	-c "${CONTEXT_SIZE}" \
	"$@"
