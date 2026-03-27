from __future__ import annotations

from pathlib import Path
import sys
import time
import uuid
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mlx_server_shared import (  # noqa: E402
    ServerControl,
    handle_management_command,
    install_shutdown_endpoint,
    resolve_runtime,
)


def build_app(model_id: str) -> FastAPI:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_id)

    app = FastAPI()
    install_shutdown_endpoint(app, "mlx-openai-server")

    class Message(BaseModel):
        role: str
        content: Any

    class ChatRequest(BaseModel):
        model: Optional[str] = model_id
        messages: List[Message]
        temperature: Optional[float] = 0.2
        max_tokens: Optional[int] = 1024
        stream: Optional[bool] = False

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{
                "id": model_id,
                "object": "model",
                "owned_by": "local",
            }],
        }

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict[str, Any]:
        prompt = tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages],
            add_generation_prompt=True,
            tokenize=False,
        )

        sampler = make_sampler(temp=req.temperature or 0.0)

        text = ""
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=req.max_tokens,
            sampler=sampler,
        ):
            if hasattr(chunk, "text"):
                text += chunk.text
            else:
                text += str(chunk)

        now = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }],
            "usage": {},
        }

    return app


def main() -> int:
    model_id, host, port = resolve_runtime(default_port=8000, port_env_name="MLX_PORT")

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

    app = build_app(model_id)
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
