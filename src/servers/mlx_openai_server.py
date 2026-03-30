from pathlib import Path
import sys

import uvicorn

from src.shared.mlx_openai_app import build_mlx_openai_app
from src.shared.mlx_server import (
    ServerControl,
    handle_management_command,
    resolve_host_port,
    resolve_runtime,
)


def build_app(model_id: str):
    return build_mlx_openai_app(
        model_id=model_id,
        runtime="mlx",
        server_name="mlx-openai-server",
    )


def main() -> int:
    host, port = resolve_host_port(
        default_port=8000,
        port_env_name="MLX_PORT",
    )

    control = ServerControl(
        server_name="mlx-openai-server",
        script_path=Path(__file__).resolve(),
        pid_file=Path("logs/mlx-server.pid"),
        log_file=Path("logs/mlx-server.log"),
        host=host,
        port=port,
    )

    management_result = handle_management_command(control, sys.argv[1:])
    if management_result is not None:
        return management_result

    model_id, _, _ = resolve_runtime(
        default_port=8000,
        port_env_name="MLX_PORT",
        model_runtime="mlx",
    )

    app = build_app(model_id)
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
