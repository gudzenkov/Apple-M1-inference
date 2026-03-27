#!/bin/bash
# Setup Ollama models with 256k context

set -euo pipefail

GIT_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_FILE="$GIT_ROOT/configs/models.yaml"
CONFIG_DIR="$GIT_ROOT/configs"

if ! command -v yq >/dev/null 2>&1; then
  echo "ERROR: yq is required but not installed."
  exit 1
fi

mapfile -t OLLAMA_MODELS < <(
  yq -r '
    .models[]
    | to_entries[]
    | select(.value.runtime == "ollama")
    | [
        .key,
        .value.model,
        (.value.build_name // .key),
        ((.value.num_ctx // 262144) | tostring)
      ]
    | @tsv
  ' "$CONFIG_FILE"
)

if [[ ${#OLLAMA_MODELS[@]} -eq 0 ]]; then
  echo "ERROR: no models with runtime: ollama in $CONFIG_FILE"
  exit 1
fi

declare -a MODELFILES=()
declare -a CREATED_MODELS=()

echo "Creating Modelfiles from $CONFIG_FILE..."
for row in "${OLLAMA_MODELS[@]}"; do
  IFS=$'\t' read -r key source_model build_name num_ctx <<< "$row"
  modelfile="$CONFIG_DIR/Modelfile.$key"
  cat > "$modelfile" << EOF
FROM $source_model
PARAMETER num_ctx $num_ctx
EOF
  MODELFILES+=("$modelfile")
done

echo "Creating models..."
for row in "${OLLAMA_MODELS[@]}"; do
  IFS=$'\t' read -r key _source_model build_name _num_ctx <<< "$row"
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
