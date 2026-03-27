from pathlib import Path
import sys
import time
import uuid
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.shared.mlx_server import (
    ServerControl,
    handle_management_command,
    install_shutdown_endpoint,
    resolve_runtime,
)


def build_app(model_id: str) -> FastAPI:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from optiq.core.turbo_kv_cache import TurboQuantKVCache, patch_attention

    model, tokenizer = load(model_id)
    patch_attention()

    def make_turbo_cache(seed_base: int = 42, bits: int = 4):
        cache = model.make_cache()
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "self_attn"):
                cache[i] = TurboQuantKVCache(
                    head_dim=layer.self_attn.head_dim,
                    bits=bits,
                    seed=seed_base + i,
                )
        return cache

    app = FastAPI()
    install_shutdown_endpoint(app, "mlx-openai-optiq-server")

    def safe_token_count(text: str) -> int:
        if not text:
            return 0
        try:
            if hasattr(tokenizer, "encode"):
                return int(len(tokenizer.encode(text)))
        except Exception:  # noqa: BLE001
            pass
        return max(1, len(text.split()))

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
        started_at = time.perf_counter()
        prompt = tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages],
            add_generation_prompt=True,
            tokenize=False,
        )

        cache = make_turbo_cache(bits=4)
        sampler = make_sampler(temp=req.temperature or 0.0)

        text = ""
        prompt_tokens = 0
        completion_tokens = 0
        prompt_tps = 0.0
        generation_tps = 0.0
        peak_memory_gb = 0.0
        first_token_at: Optional[float] = None
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=req.max_tokens,
            sampler=sampler,
            prompt_cache=cache,
        ):
            if hasattr(chunk, "text"):
                text += chunk.text
            else:
                text += str(chunk)
            if first_token_at is None:
                first_token_at = time.perf_counter()
            prompt_tokens = max(prompt_tokens, int(getattr(chunk, "prompt_tokens", 0) or 0))
            completion_tokens = max(completion_tokens, int(getattr(chunk, "generation_tokens", 0) or 0))
            prompt_tps = float(getattr(chunk, "prompt_tps", prompt_tps) or 0.0)
            generation_tps = float(getattr(chunk, "generation_tps", generation_tps) or 0.0)
            peak_memory_gb = max(peak_memory_gb, float(getattr(chunk, "peak_memory", 0.0) or 0.0))

        total_time = time.perf_counter() - started_at
        if prompt_tokens <= 0:
            prompt_tokens = safe_token_count(prompt)
        if completion_tokens <= 0:
            completion_tokens = safe_token_count(text)
        total_tokens = prompt_tokens + completion_tokens
        ttft_sec = (
            (first_token_at - started_at)
            if first_token_at is not None
            else total_time
        )

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
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "perf": {
                "ttft_sec": round(ttft_sec, 4),
                "total_time_sec": round(total_time, 4),
                "prompt_tps": round(prompt_tps, 4),
                "generation_tps": round(generation_tps, 4),
                "peak_memory_gb": round(peak_memory_gb, 4),
            },
        }

    return app


def main() -> int:
    model_id, host, port = resolve_runtime(
        default_port=8080,
        port_env_name="OPTIQ_PORT",
        model_runtime="mlx-optiq",
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

    app = build_app(model_id)
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
