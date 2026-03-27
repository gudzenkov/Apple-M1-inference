from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.shared.models import (
    get_default_model_id,
    get_models_for_runtime,
    resolve_model_reference,
)

DEFAULT_MLX_MODEL = get_default_model_id(runtime="mlx")
DEFAULT_MLX_OPTIQ_MODEL = get_default_model_id(runtime="mlx-optiq")
DEFAULT_OLLAMA_MODEL = get_default_model_id(runtime="ollama")
MLX_MODELS = get_models_for_runtime("mlx")
MLX_OPTIQ_MODELS = get_models_for_runtime("mlx-optiq")
OLLAMA_MODELS = get_models_for_runtime("ollama")

CONFIGS: Dict[str, Dict[str, Any]] = {
    "mlx": {
        "managed_server": True,
        "model_runtime": "mlx",
        "chat_url": "http://127.0.0.1:8000/v1/chat/completions",
        "health_url": "http://127.0.0.1:8000/v1/models",
        "port": 8000,
        "default_model": DEFAULT_MLX_MODEL,
        "models": MLX_MODELS,
        "start_cmd": ["uv", "run", "mlx-openai-server", "serve"],
        "log_file": "logs/mlx-server.log",
        "process_hint": "mlx_openai_server",
    },
    "mlx-optiq": {
        "managed_server": True,
        "model_runtime": "mlx-optiq",
        "chat_url": "http://127.0.0.1:8080/v1/chat/completions",
        "health_url": "http://127.0.0.1:8080/v1/models",
        "port": 8080,
        "default_model": DEFAULT_MLX_OPTIQ_MODEL,
        "models": MLX_OPTIQ_MODELS,
        "start_cmd": ["uv", "run", "mlx-openai-optiq-server", "serve"],
        "log_file": "logs/mlx-optiq-server.log",
        "process_hint": "mlx_openai_optiq_server",
    },
    "ollama": {
        "managed_server": False,
        "model_runtime": "ollama",
        "chat_url": "http://127.0.0.1:11434/v1/chat/completions",
        "health_url": "http://127.0.0.1:11434/v1/models",
        "port": 11434,
        "default_model": DEFAULT_OLLAMA_MODEL,
        "models": OLLAMA_MODELS,
        "process_hint": "ollama",
    },
}


def resolve_runtimes(runtime_arg: str) -> List[str]:
    if runtime_arg == "both":
        return ["mlx", "mlx-optiq"]
    if runtime_arg == "all":
        return ["mlx", "mlx-optiq", "ollama"]
    return [runtime_arg]


def select_models(config: Dict[str, Any], specific_model: Optional[str], all_models: bool) -> List[str]:
    model_runtime = config.get("model_runtime")
    if specific_model:
        if model_runtime:
            return [resolve_model_reference(specific_model, runtime=model_runtime)]
        return [specific_model]
    if all_models:
        return list(dict.fromkeys(config["models"]))
    if model_runtime:
        return [resolve_model_reference(config["default_model"], runtime=model_runtime)]
    return [config["default_model"]]
