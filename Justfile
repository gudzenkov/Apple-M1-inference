set shell := ["bash", "-cu"]
set dotenv-load := true

model := env_var_or_default("HUGGINGFACE_MODEL", "optiq")
host := env_var_or_default("HOST", "127.0.0.1")
mlx_port := env_var_or_default("MLX_PORT", "8000")
optiq_port := env_var_or_default("OPTIQ_PORT", "8080")
vllm_port := env_var_or_default("VLLM_PORT", "8000")
embedding_port := env_var_or_default("QWEN_EMBEDDING_PORT", "8001")

help:
    @echo "Just commands:"
    @echo "  Optional env (.env is auto-loaded):"
    @echo "    HUGGINGFACE_MODEL=<hf-repo-id-or-alias>"
    @echo "    aliases: optiq/optiq-9b, opus/opus-27b, claw/claw-27b"
    @echo "  just mlx <start|stop|status> [model]"
    @echo "  just optiq <start|stop|status> [model]"
    @echo "  just vllm-metal"
    @echo "  just qwen-embeddings"
    @echo "  just probe-openai"
    @echo "  just stop-all"
    @echo "  just test <mlx|optiq|all> [model]"
    @echo "  just bench [args...]"
    @echo "  uv run mlx-cli --list-models"
    @echo "  uv run mlx-cli -m optiq -p \"Say hi\" --max-tokens 32 --json"

mlx action model=model:
    @case "{{action}}" in \
      start) HUGGINGFACE_MODEL="{{model}}" uv run mlx-openai-server start ;; \
      stop) uv run mlx-openai-server stop ;; \
      status) uv run mlx-openai-server status ;; \
      *) echo "Invalid action '{{action}}'. Use start|stop|status." >&2; exit 2 ;; \
    esac

optiq action model=model:
    @case "{{action}}" in \
      start) HUGGINGFACE_MODEL="{{model}}" uv run mlx-openai-optiq-server start ;; \
      stop) uv run mlx-openai-optiq-server stop ;; \
      status) uv run mlx-openai-optiq-server status ;; \
      *) echo "Invalid action '{{action}}'. Use start|stop|status." >&2; exit 2 ;; \
    esac

vllm-metal:
    @scripts/start-vllm-metal-qwen36.sh

qwen-embeddings:
    @scripts/start-qwen3-embedding-llama.cpp.sh

probe-openai:
    @OPENAI_BASE_URL="http://{{host}}:{{vllm_port}}/v1" OPENAI_EMBEDDINGS_BASE_URL="http://{{host}}:{{embedding_port}}/v1" scripts/probe-openai-local.sh

stop-all:
    @just mlx stop || true
    @just optiq stop || true

test target model=model:
    @case "{{target}}" in \
      mlx) \
        uv run mlx-cli --server mlx --base-url "http://{{host}}:{{mlx_port}}" --list-models --json; \
        uv run mlx-cli --server mlx --base-url "http://{{host}}:{{mlx_port}}" -m "{{model}}" -p "Say hi in 5 words." --max-tokens 32 --json ;; \
      optiq) \
        uv run mlx-cli --server optiq --base-url "http://{{host}}:{{optiq_port}}" --list-models --json; \
        uv run mlx-cli --server optiq --base-url "http://{{host}}:{{optiq_port}}" -m "{{model}}" -p "Say hi in 5 words." --max-tokens 32 --json ;; \
      all) \
        just test mlx "{{model}}"; \
        just test optiq "{{model}}" ;; \
      *) echo "Invalid target '{{target}}'. Use mlx|optiq|all." >&2; exit 2 ;; \
    esac

bench *args:
    @uv run benchmark {{args}}
