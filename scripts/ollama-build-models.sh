#!/bin/bash
# Setup Ollama models with 256k context

set -euo pipefail

GIT_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_FILE="$GIT_ROOT/configs/models.yaml"
BENCH_FILE="$GIT_ROOT/configs/bench.yaml"
CONFIG_DIR="$GIT_ROOT/configs"

if ! command -v yq >/dev/null 2>&1; then
  echo "ERROR: yq is required but not installed."
  exit 1
fi

mapfile -t OLLAMA_MODELS < <(
  yq -r '
    .models[]
    | to_entries[]
    | select(.value.runtime.server == "ollama")
    | [
        .key,
        .value.model.name,
        (.value.build_name // .key)
      ]
    | @tsv
  ' "$CONFIG_FILE"
)

OLLAMA_NUM_CTX="$(yq -r '.defaults.runtimes.ollama.request_options.num_ctx // 262144' "$BENCH_FILE")"

if [[ ${#OLLAMA_MODELS[@]} -eq 0 ]]; then
  echo "ERROR: no models with runtime: ollama in $CONFIG_FILE"
  exit 1
fi

declare -a MODELFILES=()
declare -a CREATED_MODELS=()

echo "Creating Modelfiles from $CONFIG_FILE..."
for row in "${OLLAMA_MODELS[@]}"; do
  IFS=$'\t' read -r key source_model build_name <<< "$row"
  modelfile="$CONFIG_DIR/Modelfile.$key"
  cat > "$modelfile" << EOF
FROM $source_model
PARAMETER num_ctx $OLLAMA_NUM_CTX
EOF
  MODELFILES+=("$modelfile")
done

echo "Creating models..."
for row in "${OLLAMA_MODELS[@]}"; do
  IFS=$'\t' read -r key _source_model build_name <<< "$row"
  target_model="${build_name}-256k"
  ollama create "$target_model" -f "$CONFIG_DIR/Modelfile.$key"
  CREATED_MODELS+=("$target_model")
done

echo "✅ Done! Models with 256k context:"
for model in "${CREATED_MODELS[@]}"; do
  echo "  - $model"
done

# Cleanup
rm "${MODELFILES[@]}"

echo ""
echo "Update opencode.json to use these models with -256k suffix"
