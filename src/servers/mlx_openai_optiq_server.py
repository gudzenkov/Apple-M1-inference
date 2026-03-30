from pathlib import Path
import sys
from typing import Any, List

import uvicorn

from src.shared.mlx_openai_app import build_mlx_openai_app
from src.shared.mlx_server import (
    ServerControl,
    handle_management_command,
    resolve_host_port,
    resolve_runtime,
)


def _make_turbo_cache(model: Any, *, seed_base: int = 42, bits: int = 4) -> List[Any]:
    from optiq.core.turbo_kv_cache import TurboQuantKVCache

    cache = model.make_cache()
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "self_attn"):
            cache[i] = TurboQuantKVCache(
                head_dim=layer.self_attn.head_dim,
                bits=bits,
                seed=seed_base + i,
            )
    return cache


def build_app(model_id: str):
    from optiq.core.turbo_kv_cache import patch_attention

    patch_attention()
    return build_mlx_openai_app(
        model_id=model_id,
        runtime="mlx-optiq",
        server_name="mlx-openai-optiq-server",
        cache_factory=_make_turbo_cache,
        ephemeral_cache_without_cache_id=True,
    )


def main() -> int:
    host, port = resolve_host_port(
        default_port=8080,
        port_env_name="OPTIQ_PORT",
    )

    control = ServerControl(
        server_name="mlx-openai-optiq-server",
        script_path=Path(__file__).resolve(),
        pid_file=Path("logs/mlx-optiq-server.pid"),
        log_file=Path("logs/mlx-optiq-server.log"),
        host=host,
        port=port,
    )

    management_result = handle_management_command(control, sys.argv[1:])
    if management_result is not None:
        return management_result

    model_id, _, _ = resolve_runtime(
        default_port=8080,
        port_env_name="OPTIQ_PORT",
        model_runtime="mlx-optiq",
    )

    app = build_app(model_id)
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
