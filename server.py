"""OpenAI-compatible FastAPI server for Qwen3-4B with Attention KV Cache (AKV)."""

import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from akv import (
    AKVEngine,
    ChatCompletionRequest,
    settings,
)
from akv.schemas import CacheStatusResponse, ModelCard, ModelListResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("akv.server")

# Global engine instance
engine: AKVEngine = AKVEngine(
    model_path=settings.model_path,
    n_ctx=settings.n_ctx,
    model_id=settings.model_id,
    n_gpu_layers=settings.n_gpu_layers,
    flash_attn=settings.flash_attn,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"AKV Server ready on http://{settings.host}:{settings.port}")
    yield
    logger.info("AKV Server shutting down...")


app = FastAPI(
    title="Qwen3-4B AKV OpenAI-Compatible Server",
    description="Stateful LLM server with selective Attention KV cache pruning and RoPE repositioning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ui")
def ui():
    return FileResponse("index.html")


@app.get("/")
def root():
    return {
        "status": "online",
        "model": settings.model_id,
        "n_ctx": settings.n_ctx,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.model_id}


@app.get("/v1/models", response_model=ModelListResponse)
def list_models():
    return ModelListResponse(
        data=[ModelCard(id=settings.model_id, owned_by="local-akv")]
    )


@app.get("/v1/models/{model_id}", response_model=ModelCard)
def get_model(model_id: str):
    if model_id != settings.model_id:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelCard(id=settings.model_id, owned_by="local-akv")


@app.post("/v1/chat/completions")
def create_chat_completion(req: ChatCompletionRequest) -> Dict[str, Any]:
    """OpenAI-compatible chat completion endpoint supporting AKV cache manipulation."""
    try:
        return engine.process_chat_completion(
            messages=req.messages,
            tools=req.tools,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.get_max_tokens(),
            enable_thinking=req.enable_thinking,
            drop_message=req.drop_message,
            reposition=req.reposition,
            auto_drop_keep_last_n_tools=req.auto_drop_keep_last_n_tools,
        )
    except ValueError as ve:
        logger.warning(f"Validation/Context error in chat completion: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Inference error:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/v1/cache/reset")
def reset_kv_cache():
    """Explicitly resets the entire KV cache."""
    engine.reset_cache()
    return {"status": "ok", "message": "KV cache cleared successfully"}


@app.get("/v1/cache/status", response_model=CacheStatusResponse)
def get_kv_cache_status():
    """Returns the current diagnostics and tracked message states of the KV cache."""
    return engine.get_cache_status()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
    )
