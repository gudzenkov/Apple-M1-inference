from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
from mlx_lm import load, stream_generate
from optiq.core.turbo_kv_cache import TurboQuantKVCache
import uvicorn
import time
import uuid
import os

MODEL_ID = os.getenv("HUGGINGFACE_MODEL", "mlx-community/Qwen3.5-9B-OptiQ-4bit")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

model, tokenizer = load(MODEL_ID)

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

class Message(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[Message]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "owned_by": "local"
        }]
    }

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        add_generation_prompt=True,
        tokenize=False,
    )

    cache = make_turbo_cache(bits=4)

    text = ""
    for chunk in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=req.max_tokens,
        temp=req.temperature,
        prompt_cache=cache,
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
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {}
    }

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
