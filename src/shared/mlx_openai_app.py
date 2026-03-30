import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.shared.mlx_server import install_shutdown_endpoint
from src.shared.models import get_model_entry


CacheFactory = Callable[[Any], List[Any]]


def build_mlx_openai_app(
    *,
    model_id: str,
    runtime: str,
    server_name: str,
    cache_factory: Optional[CacheFactory] = None,
    ephemeral_cache_without_cache_id: bool = False,
) -> FastAPI:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_id)

    try:
        model_entry = get_model_entry(model_id, runtime=runtime)
    except Exception:  # noqa: BLE001
        model_entry = None

    capabilities = dict(model_entry.get("capabilities", {})) if isinstance(model_entry, dict) else {}
    reasoning_supported = bool(capabilities.get("reasoning_supported", False))
    reasoning_format_raw = capabilities.get("reasoning_format")
    reasoning_format = (
        str(reasoning_format_raw).strip().lower()
        if isinstance(reasoning_format_raw, str) and reasoning_format_raw.strip()
        else None
    )

    app = FastAPI()
    install_shutdown_endpoint(app, server_name)

    prompt_cache_store: Dict[str, Dict[str, Any]] = {}
    prompt_cache_lock = threading.Lock()

    def make_runtime_cache() -> List[Any]:
        if cache_factory is not None:
            return cache_factory(model)
        return model.make_cache()

    def safe_token_count(text: str) -> int:
        if not text:
            return 0
        try:
            if hasattr(tokenizer, "encode"):
                return int(len(tokenizer.encode(text)))
        except Exception:  # noqa: BLE001
            pass
        return max(1, len(text.split()))

    def encode_prompt(prompt: str) -> mx.array:
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        token_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return mx.array(token_ids)

    def cache_size(prompt_cache: Optional[List[Any]]) -> int:
        if not prompt_cache:
            return 0
        first = prompt_cache[0]
        size_attr = getattr(first, "size", None)
        if callable(size_attr):
            return int(size_attr())
        if isinstance(size_attr, int):
            return int(size_attr)
        return 0

    def trim_cache_to_size(prompt_cache: Optional[List[Any]], target_size: int) -> None:
        if not prompt_cache:
            return
        current_size = cache_size(prompt_cache)
        trim_tokens = current_size - max(0, target_size)
        if trim_tokens <= 0:
            return
        for layer_cache in prompt_cache:
            trim_fn = getattr(layer_cache, "trim", None)
            if callable(trim_fn):
                trim_fn(trim_tokens)

    def prefill_cache(prompt: str) -> Dict[str, Any]:
        prompt_cache = make_runtime_cache()
        prompt_tokens = encode_prompt(prompt)
        sampler = make_sampler(temp=0.0)
        for _ in generate_step(
            prompt_tokens,
            model,
            max_tokens=0,
            sampler=sampler,
            prompt_cache=prompt_cache,
        ):
            pass
        prefix_size = cache_size(prompt_cache)
        return {
            "cache": prompt_cache,
            "prefix_size": prefix_size,
            "prompt_tokens": int(prompt_tokens.size),
            "created_at": int(time.time()),
        }

    class Message(BaseModel):
        role: str
        content: Any

    class ChatRequest(BaseModel):
        model: Optional[str] = model_id
        messages: List[Message] = Field(default_factory=list)
        raw_prompt: Optional[str] = None
        cache_id: Optional[str] = None
        reasoning: Optional[Any] = None
        reasoning_effort: Optional[str] = None
        temperature: Optional[float] = 0.2
        max_tokens: Optional[int] = 1024
        stream: Optional[bool] = False

    class CachePrefillRequest(BaseModel):
        cache_id: str
        raw_prompt: str

    class CacheClearRequest(BaseModel):
        cache_id: str

    def requested_reasoning_mode(req: ChatRequest) -> str:
        mode = "off"
        if isinstance(req.reasoning, str):
            value = req.reasoning.strip().lower()
            if value in {"off", "on", "auto"}:
                mode = value
        elif isinstance(req.reasoning, dict):
            raw_effort = req.reasoning.get("effort")
            effort = str(raw_effort).strip().lower() if isinstance(raw_effort, str) else ""
            if effort in {"none", "off"}:
                mode = "off"
            elif effort:
                mode = "on"

        if isinstance(req.reasoning_effort, str) and req.reasoning_effort.strip():
            effort = req.reasoning_effort.strip().lower()
            mode = "off" if effort in {"none", "off"} else "on"

        return mode

    def normalize_reasoning_mode(requested_mode: str) -> str:
        if requested_mode == "auto":
            return "on" if reasoning_supported else "off"
        return requested_mode

    def reasoning_template_kwargs(reasoning_mode: str) -> dict[str, Any]:
        if reasoning_mode == "on" and not reasoning_supported:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Reasoning requested but not supported for this model/profile. "
                    "Use reasoning=off or a reasoning-capable model."
                ),
            )
        if reasoning_format == "qwen":
            return {"enable_thinking": reasoning_mode == "on"}
        if reasoning_mode == "on" and reasoning_format is None:
            raise HTTPException(
                status_code=400,
                detail="Reasoning requested but no reasoning template format is configured.",
            )
        return {}

    def resolve_prompt(req: ChatRequest, reasoning_mode: str) -> str:
        if req.raw_prompt and req.raw_prompt.strip():
            return req.raw_prompt
        if not req.messages:
            raise HTTPException(status_code=400, detail="Either raw_prompt or messages must be provided")
        messages = [m.model_dump() for m in req.messages]
        kwargs = reasoning_template_kwargs(reasoning_mode)
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
        except TypeError as exc:
            if kwargs:
                raise HTTPException(
                    status_code=400,
                    detail=f"Reasoning template kwargs are not supported by tokenizer: {exc}",
                ) from exc
            raise

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

    @app.post("/v1/cache/prefill")
    def cache_prefill(req: CachePrefillRequest) -> dict[str, Any]:
        cache_id = req.cache_id.strip()
        prompt = req.raw_prompt.strip()
        if not cache_id:
            raise HTTPException(status_code=400, detail="cache_id must be non-empty")
        if not prompt:
            raise HTTPException(status_code=400, detail="raw_prompt must be non-empty")
        cache_entry = prefill_cache(prompt)
        with prompt_cache_lock:
            prompt_cache_store[cache_id] = cache_entry
        return {
            "success": True,
            "cache_id": cache_id,
            "prefix_size": cache_entry["prefix_size"],
            "prompt_tokens": cache_entry["prompt_tokens"],
            "created_at": cache_entry["created_at"],
        }

    @app.post("/v1/cache/clear")
    def cache_clear(req: CacheClearRequest) -> dict[str, Any]:
        cache_id = req.cache_id.strip()
        if not cache_id:
            raise HTTPException(status_code=400, detail="cache_id must be non-empty")
        with prompt_cache_lock:
            removed = prompt_cache_store.pop(cache_id, None) is not None
        return {
            "success": True,
            "cache_id": cache_id,
            "cleared": removed,
        }

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict[str, Any]:
        started_at = time.perf_counter()
        requested_mode = requested_reasoning_mode(req)
        reasoning_mode = normalize_reasoning_mode(requested_mode)
        prompt = resolve_prompt(req, reasoning_mode)

        sampler = make_sampler(temp=req.temperature or 0.0)
        prompt_cache: Optional[List[Any]] = None
        prompt_cache_prefix_size = 0
        if req.cache_id:
            cache_id = req.cache_id.strip()
            with prompt_cache_lock:
                cache_entry = prompt_cache_store.get(cache_id)
            if cache_entry is None:
                raise HTTPException(status_code=404, detail=f"Prompt cache not found: {cache_id}")
            prompt_cache = cache_entry["cache"]
            prompt_cache_prefix_size = int(cache_entry.get("prefix_size", 0) or 0)
            trim_cache_to_size(prompt_cache, prompt_cache_prefix_size)
        elif ephemeral_cache_without_cache_id:
            prompt_cache = make_runtime_cache()

        text = ""
        prompt_tokens = 0
        completion_tokens = 0
        prompt_tps = 0.0
        generation_tps = 0.0
        peak_memory_gb = 0.0
        first_token_at: Optional[float] = None
        stream_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": req.max_tokens,
            "sampler": sampler,
        }
        if prompt_cache is not None:
            stream_kwargs["prompt_cache"] = prompt_cache

        try:
            for chunk in stream_generate(
                model,
                tokenizer,
                **stream_kwargs,
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
        finally:
            if req.cache_id and prompt_cache is not None:
                trim_cache_to_size(prompt_cache, prompt_cache_prefix_size)

        total_time = time.perf_counter() - started_at
        if prompt_tokens <= 0:
            prompt_tokens = safe_token_count(prompt)
        if completion_tokens <= 0:
            completion_tokens = safe_token_count(text)
        total_tokens = prompt_tokens + completion_tokens
        ttft_sec = (first_token_at - started_at) if first_token_at is not None else total_time

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
            "reasoning": {
                "requested": requested_mode,
                "effective": reasoning_mode,
                "supported": reasoning_supported,
                "format": reasoning_format,
            },
        }

    return app
