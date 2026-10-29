"""Low-level KV cache manipulation primitives for llama.cpp.

Provides cross-version compatibility for:
- Selective KV cache sequence eviction (`kv_cache_rm`)
- RoPE position shifting / compaction (`kv_cache_shift`)
- Full KV cache reset (`kv_cache_clear`)
"""

import logging
from typing import Any
import llama_cpp
from llama_cpp import Llama

logger = logging.getLogger("akv.kv_cache")


def kv_cache_rm(llm: Llama, seq_id: int, p0: int, p1: int) -> None:
    """Evicts KV cache entries in position range [p0, p1) for sequence seq_id.
    
    If seq_id is -1, applies to all sequences.
    If p1 is -1, evicts from p0 through the end of the context.
    """
    ctx = llm._ctx.ctx

    if hasattr(llm._ctx, "kv_cache_seq_rm"):
        llm._ctx.kv_cache_seq_rm(seq_id, p0, p1)
    elif hasattr(llama_cpp, "llama_get_memory") and hasattr(llama_cpp, "llama_memory_seq_rm"):
        mem = llama_cpp.llama_get_memory(ctx)
        llama_cpp.llama_memory_seq_rm(mem, seq_id, p0, p1)
    elif hasattr(llama_cpp, "llama_kv_self_seq_rm"):
        llama_cpp.llama_kv_self_seq_rm(ctx, seq_id, p0, p1)
    elif hasattr(llama_cpp, "llama_kv_cache_seq_rm"):
        llama_cpp.llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)
    else:
        raise RuntimeError("No compatible KV cache removal method found in llama_cpp installation.")


def kv_cache_shift(llm: Llama, seq_id: int, p0: int, p1: int, delta: int) -> None:
    """Rotates RoPE embeddings in-place by `delta` for positions in [p0, p1).
    
    A negative delta shifts positions backwards (left) to close gaps left by
    evicted messages.
    """
    ctx = llm._ctx.ctx

    if hasattr(llm._ctx, "kv_cache_seq_add"):
        llm._ctx.kv_cache_seq_add(seq_id, p0, p1, delta)
    elif hasattr(llama_cpp, "llama_get_memory") and hasattr(llama_cpp, "llama_memory_seq_add"):
        mem = llama_cpp.llama_get_memory(ctx)
        llama_cpp.llama_memory_seq_add(mem, seq_id, p0, p1, delta)
    elif hasattr(llama_cpp, "llama_kv_self_seq_add"):
        llama_cpp.llama_kv_self_seq_add(ctx, seq_id, p0, p1, delta)
    elif hasattr(llama_cpp, "llama_kv_cache_seq_add"):
        llama_cpp.llama_kv_cache_seq_add(ctx, seq_id, p0, p1, delta)
    else:
        raise RuntimeError("No compatible KV cache shift method found in llama_cpp installation.")


def kv_cache_clear(llm: Llama) -> None:
    """Completely clears all sequences from the KV cache and resets model state."""
    try:
        kv_cache_rm(llm, -1, 0, -1)
    except Exception as e:
        logger.warning(f"Error during kv_cache_rm clear: {e}")

    if hasattr(llm, "reset"):
        try:
            llm.reset()
        except Exception as e:
            logger.warning(f"Error during llm.reset(): {e}")
