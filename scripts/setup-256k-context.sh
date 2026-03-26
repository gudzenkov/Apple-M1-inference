#!/bin/bash
# Setup Ollama models with 256k context

set -e

echo "Creating Modelfiles with 256k context..."

# qwen3.5:9b
cat > /tmp/Modelfile.qwen3.5-9b << 'EOF'
FROM qwen3.5:9b
PARAMETER num_ctx 262144
EOF

# SimonPu/Qwen3-Coder:30B
cat > /tmp/Modelfile.qwen3-coder-30b << 'EOF'
FROM SimonPu/Qwen3-Coder:30B-Instruct_Q4_K_XL
PARAMETER num_ctx 262144
EOF

# sinhang/opus
cat > /tmp/Modelfile.opus-27b << 'EOF'
FROM sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M
PARAMETER num_ctx 262144
EOF

# ukjin/deepseek
cat > /tmp/Modelfile.deepseek-30b << 'EOF'
FROM ukjin/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill
PARAMETER num_ctx 262144
EOF

echo "Creating models with 256k context..."
ollama create qwen3.5-9b-256k -f /tmp/Modelfile.qwen3.5-9b
ollama create qwen3-coder-30b-256k -f /tmp/Modelfile.qwen3-coder-30b
ollama create opus-27b-256k -f /tmp/Modelfile.opus-27b
ollama create deepseek-30b-256k -f /tmp/Modelfile.deepseek-30b

echo "✅ Done! Models with 256k context:"
echo "  - qwen3.5-9b-256k"
echo "  - qwen3-coder-30b-256k"
echo "  - opus-27b-256k"
echo "  - deepseek-30b-256k"

# Cleanup
rm /tmp/Modelfile.*

echo ""
echo "Update opencode.json to use these models with -256k suffix"
