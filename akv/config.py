"""Configuration settings for the AKV (Attention KV Cache) inference server."""

import os
from dataclasses import dataclass

N_CTX = int(os.environ.get("N_CTX", "4096"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "28"))


@dataclass(frozen=True)
class Settings:
    # Model configuration
    model_path: str = os.environ.get(
        "QWEN_MODEL_PATH",
        r"E:\Models\qwen3_akv\models\Qwen3-4B-Q4_K_M.gguf",
    )
    model_id: str = os.environ.get("QWEN_MODEL_ID", "qwen3-4b")
    n_ctx: int = N_CTX
    n_gpu_layers: int = N_GPU_LAYERS
    flash_attn: bool = False

    # Server configuration
    host: str = os.environ.get("HOST", "127.0.0.1")
    port: int = int(os.environ.get("PORT", "8000"))

    # Generation defaults
    default_temperature: float = 0.7
    default_top_p: float = 0.8
    default_top_k: int = 20
    default_max_tokens: int = 512


settings = Settings()

