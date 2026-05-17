#!/usr/bin/env bash
set -euo pipefail

CHAT_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
CHAT_MODEL="${QWEN_CHAT_MODEL:-qwen3.6-35b-a3b}"
EMBEDDINGS_BASE_URL="${OPENAI_EMBEDDINGS_BASE_URL:-http://127.0.0.1:8001/v1}"
EMBEDDING_MODEL="${QWEN_EMBEDDING_MODEL:-qwen3-embedding}"

CHAT_BASE_URL="${CHAT_BASE_URL%/}"
EMBEDDINGS_BASE_URL="${EMBEDDINGS_BASE_URL%/}"

printf 'Checking chat models: %s/models\n' "${CHAT_BASE_URL}" >&2
curl -fsS "${CHAT_BASE_URL}/models" >/dev/null

printf 'Checking chat completion: %s/chat/completions\n' "${CHAT_BASE_URL}" >&2
curl -fsS "${CHAT_BASE_URL}/chat/completions" \
	-H "Content-Type: application/json" \
	-d "{
    \"model\": \"${CHAT_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Return exactly OK.\"}],
    \"max_tokens\": 16,
    \"temperature\": 0,
    \"chat_template_kwargs\": {\"enable_thinking\": false},
    \"stream\": false
  }" >/dev/null

printf 'Checking embeddings: %s/embeddings\n' "${EMBEDDINGS_BASE_URL}" >&2
curl -fsS "${EMBEDDINGS_BASE_URL}/embeddings" \
	-H "Content-Type: application/json" \
	-d "{
    \"model\": \"${EMBEDDING_MODEL}\",
    \"input\": \"test text\"
  }" >/dev/null

printf 'OK: chat and embeddings endpoints responded\n'
