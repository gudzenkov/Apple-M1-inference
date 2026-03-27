set shell := ["bash", "-cu"]

model := env_var("HUGGINGFACE_MODEL")
host := env_var("HOST")
mlx_port := env_var("MLX_PORT")
optiq_port := env_var("OPTIQ_PORT")

help:
    @echo "Just commands:"
    @echo "  export HUGGINGFACE_MODEL=<hf-repo-id>"
    @echo "    aliases: optiq/optiq-9b, opus/opus-27b, claw/claw-27b, coder/coder-30b"
    @echo "  export HOST=<host>"
    @echo "  export MLX_PORT=<port>"
    @echo "  export OPTIQ_PORT=<port>"
    @echo "  just mlx <start|stop|status> [model]"
    @echo "  just optiq <start|stop|status> [model]"
    @echo "  just stop-all"
    @echo "  just test <mlx|optiq|all> [model]"
    @echo "  just bench [args...]      # runs dataset=all"
    @echo "  just bench-dataset <quick|long|all> [args...]"

mlx action model=model:
    @case "{{action}}" in \
      start) HUGGINGFACE_MODEL="{{model}}" .venv/bin/python src/mlx-openai-server/mlx-openai-server.py start ;; \
      stop) .venv/bin/python src/mlx-openai-server/mlx-openai-server.py stop ;; \
      status) .venv/bin/python src/mlx-openai-server/mlx-openai-server.py status ;; \
      *) echo "Invalid action '{{action}}'. Use start|stop|status." >&2; exit 2 ;; \
    esac

optiq action model=model:
    @case "{{action}}" in \
      start) HUGGINGFACE_MODEL="{{model}}" .venv/bin/python src/mlx-openai-optiq-server/mlx-openai-optiq-server.py start ;; \
      stop) .venv/bin/python src/mlx-openai-optiq-server/mlx-openai-optiq-server.py stop ;; \
      status) .venv/bin/python src/mlx-openai-optiq-server/mlx-openai-optiq-server.py status ;; \
      *) echo "Invalid action '{{action}}'. Use start|stop|status." >&2; exit 2 ;; \
    esac

stop-all:
    @just mlx stop || true
    @just optiq stop || true

test target model=model:
    @case "{{target}}" in \
      mlx) \
        curl -fsS "http://{{host}}:{{mlx_port}}/v1/models" | python3 -m json.tool; \
        curl -fsS "http://{{host}}:{{mlx_port}}/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{"model":"{{model}}","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":32}' | python3 -m json.tool ;; \
      optiq) \
        curl -fsS "http://{{host}}:{{optiq_port}}/v1/models" | python3 -m json.tool; \
        curl -fsS "http://{{host}}:{{optiq_port}}/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{"model":"{{model}}","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":32}' | python3 -m json.tool ;; \
      all) \
        just test mlx "{{model}}"; \
        just test optiq "{{model}}" ;; \
      *) echo "Invalid target '{{target}}'. Use mlx|optiq|all." >&2; exit 2 ;; \
    esac

bench *args:
    @.venv/bin/python scripts/benchmark.py --dataset all {{args}}

bench-dataset dataset *args:
    @.venv/bin/python scripts/benchmark.py --dataset "{{dataset}}" {{args}}
