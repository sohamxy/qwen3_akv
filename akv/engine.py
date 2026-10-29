"""Stateful AKV (Attention KV Cache) Inference Engine."""

from dataclasses import dataclass
import logging
import os
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import llama_cpp
from llama_cpp import Llama

from akv.config import settings
from akv.chatml import format_message_to_chatml, parse_tool_calls_from_reply
from akv.kv_cache import kv_cache_clear, kv_cache_rm, kv_cache_shift
from akv.sampling import sample_next_token

logger = logging.getLogger("akv.engine")

N_CTX = int(os.environ.get("N_CTX", "4096"))


@dataclass
class CachedMessage:
    index: int
    role: str
    chatml: str
    p0: int
    p1: int
    active: bool = True
    token_count: int = 0


class AKVEngine:
    """Manages model loading, stateful KV caching, selective drop, and position repositioning."""

    def __init__(
        self,
        model_path: str = settings.model_path,
        n_ctx: int = N_CTX,
        model_id: str = settings.model_id,
        n_gpu_layers: int = int(os.environ.get("N_GPU_LAYERS", "28")),
        flash_attn: bool = False,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        logger.info(f"Loading {model_path} (n_ctx={n_ctx}, n_gpu_layers={int(os.environ.get('N_GPU_LAYERS', '28'))}, flash_attn=False)...")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "28")),
                n_threads=6,
                n_threads_batch=8,
                n_batch=512,
                n_ubatch=128,
                flash_attn=False,
                type_k=1,
                type_v=8,
                verbose=False,
            )
        except ValueError:
            # llama.cpp requires flash_attn=True when type_v is quantized (type_v=8 / Q8_0).
            # When flash_attn is disabled to prevent Vulkan shader micro-stutters on Radeon 780M,
            # fallback to type_v=1 (GGML_TYPE_F16) so the context can be created successfully.
            logger.warning(
                "Llama context initialization failed with flash_attn=False and type_v=8 "
                "(quantized V-cache requires flash_attn in llama.cpp). Falling back to type_v=1 (FP16)."
            )
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "28")),
                n_threads=6,
                n_threads_batch=8,
                n_batch=512,
                n_ubatch=128,
                flash_attn=False,
                type_k=1,
                type_v=1,
                verbose=False,
            )
        self.ctx = self.llm._ctx.ctx
        self.n_vocab = self.llm._n_vocab
        self.n_ctx = n_ctx
        self.model_id = model_id
        self.seq_id = 0
        self.logical_pos = 0
        self.last_batch_len = 1
        self.lock = Lock()

        self.cached_messages: List[CachedMessage] = []

        # Resolve stop token IDs
        self.stop_token_ids: Set[int] = {self.llm.token_eos()}
        for token_str in (b"<|im_end|>", b"<|endoftext|>"):
            toks = self.llm.tokenize(token_str, add_bos=False, special=True)
            if toks:
                self.stop_token_ids.add(toks[-1])

        logger.info(f"Model loaded successfully! Vocab size: {self.n_vocab}, Stop IDs: {self.stop_token_ids}")

    def reset_cache(self) -> None:
        """Fully clears the KV cache and resets all tracking structures."""
        with self.lock:
            kv_cache_clear(self.llm)
            self.logical_pos = 0
            self.last_batch_len = 1
            self.cached_messages.clear()
            logger.info("[AKV] Full KV cache reset completed.")

    def _eval_tokens(self, tokens: List[int]) -> None:
        """Evaluates tokens at `self.logical_pos` using llama-cpp-python's internal batch."""
        if not tokens:
            return

        n_step = min(self.llm.n_batch, 256)
        for start in range(0, len(tokens), n_step):
            chunk = tokens[start : start + n_step]
            n_cur = len(chunk)

            if self.logical_pos + n_cur > self.n_ctx:
                raise ValueError(
                    f"Context window exceeded: logical_pos ({self.logical_pos}) + tokens ({n_cur}) > n_ctx ({self.n_ctx})"
                )

            self.llm._batch.set_batch(
                batch=chunk,
                n_past=self.logical_pos,
                logits_all=False,
            )
            self.llm._ctx.decode(self.llm._batch)

            self.logical_pos += n_cur
            self.last_batch_len = n_cur

            if start + n_step < len(tokens):
                time.sleep(0.001)

    def _get_last_token_logits(self) -> np.ndarray:
        """Retrieves logits for the last evaluated token from llama context."""
        logits_ptr = llama_cpp.llama_get_logits_ith(self.ctx, self.last_batch_len - 1)
        if not logits_ptr:
            logits_ptr = llama_cpp.llama_get_logits(self.ctx)
        return np.ctypeslib.as_array(logits_ptr, shape=(self.n_vocab,)).copy()

    def drop_message_kv(self, msg_idx: int) -> int:
        """Evicts the KV cache for a given message index without altering other positions."""
        if 0 <= msg_idx < len(self.cached_messages):
            entry = self.cached_messages[msg_idx]
            if entry.active:
                kv_cache_rm(self.llm, self.seq_id, entry.p0, entry.p1)
                entry.active = False
                freed = entry.p1 - entry.p0
                logger.info(
                    f"  [AKV DROP] Message #{msg_idx} ({entry.role}) evicted "
                    f"from pos [{entry.p0}, {entry.p1}) (-{freed} tokens)"
                )
                return freed
        return 0

    def reposition_kv(self) -> int:
        """Compacts KV positions via RoPE rotation to eliminate gaps from dropped messages."""
        old_pos = self.logical_pos
        new_pos = 0

        for entry in self.cached_messages:
            if not entry.active:
                continue

            p0, p1 = entry.p0, entry.p1
            length = p1 - p0
            if p0 > new_pos:
                delta = new_pos - p0
                kv_cache_shift(self.llm, self.seq_id, p0, p1, delta)
                entry.p0 = new_pos
                entry.p1 = new_pos + length

            new_pos = entry.p1

        self.logical_pos = new_pos
        saved = old_pos - new_pos
        if saved > 0:
            logger.info(f"  [AKV REPOS] Compacted KV cache: {old_pos} -> {new_pos} (-{saved} gap tokens)")
        return saved

    def get_cache_status(self) -> Dict[str, Any]:
        """Returns diagnostic metrics and message states for the KV cache."""
        with self.lock:
            active_tokens = sum(e.p1 - e.p0 for e in self.cached_messages if e.active)
            dropped_tokens = sum(e.p1 - e.p0 for e in self.cached_messages if not e.active)
            return {
                "logical_pos": self.logical_pos,
                "n_ctx": self.n_ctx,
                "active_kv_tokens": active_tokens,
                "dropped_kv_tokens": dropped_tokens,
                "total_messages_tracked": len(self.cached_messages),
                "active_messages": sum(1 for e in self.cached_messages if e.active),
                "messages": [
                    {
                        "index": e.index,
                        "role": e.role,
                        "p0": e.p0,
                        "p1": e.p1,
                        "active": e.active,
                        "token_count": e.token_count,
                    }
                    for e in self.cached_messages
                ],
            }

    def process_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        drop_message: Optional[Dict[Union[str, int], List[int]]] = None,
        reposition: Optional[List[Union[str, int]]] = None,
        auto_drop_keep_last_n_tools: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes stateful chat completion with selective KV caching and manipulation."""
        with self.lock:
            # Check if there is an explicit system message; if not and tools are present, prepend one
            has_system = any(m.get("role") == "system" for m in messages)
            eff_messages = list(messages)
            if not has_system and tools:
                eff_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

            # Format messages into ChatML blocks
            has_system_tools = False
            formatted_list: List[str] = []
            for m in eff_messages:
                is_first_sys = (m.get("role") == "system" and not has_system_tools)
                if is_first_sys:
                    has_system_tools = True
                formatted_list.append(format_message_to_chatml(m, tools, is_first_sys))

            # 1. Verify prefix continuity with cached messages
            prefix_match_len = 0
            for i in range(min(len(formatted_list), len(self.cached_messages))):
                if self.cached_messages[i].chatml == formatted_list[i]:
                    prefix_match_len = i + 1
                else:
                    break

            if prefix_match_len < len(self.cached_messages):
                if prefix_match_len == 0:
                    self.reset_cache()
                else:
                    active_kept = [e for e in self.cached_messages[:prefix_match_len] if e.active]
                    cut_p0 = active_kept[-1].p1 if active_kept else 0
                    kv_cache_rm(self.llm, self.seq_id, cut_p0, -1)
                    self.logical_pos = cut_p0
                    self.cached_messages = self.cached_messages[:prefix_match_len]
                    logger.info(f"[AKV ROLLBACK] Rolled back cache to index {prefix_match_len}, cut_p0={cut_p0}")

            # Normalize drop_message and reposition parameters
            drop_map: Dict[str, List[int]] = {}
            if drop_message:
                for k, v in drop_message.items():
                    drop_map[str(k)] = v

            repos_set: Set[int] = set()
            if reposition:
                for item in reposition:
                    try:
                        repos_set.add(int(item))
                    except (ValueError, TypeError):
                        pass

            # 2. AKV APPEND: Prefill newly arrived messages
            newly_prefilled_tokens = 0
            for idx in range(prefix_match_len, len(formatted_list)):
                raw_chatml = formatted_list[idx]
                tokens = self.llm.tokenize(raw_chatml.encode("utf-8"), add_bos=(idx == 0), special=True)
                p0 = self.logical_pos
                self._eval_tokens(tokens)
                p1 = self.logical_pos
                newly_prefilled_tokens += len(tokens)

                self.cached_messages.append(
                    CachedMessage(
                        index=idx,
                        role=eff_messages[idx].get("role", "user"),
                        chatml=raw_chatml,
                        p0=p0,
                        p1=p1,
                        active=True,
                        token_count=len(tokens),
                    )
                )

                # 3. Execute AKV drop_message & reposition triggered at message index `idx`
                if str(idx) in drop_map:
                    for target_idx in drop_map[str(idx)]:
                        self.drop_message_kv(target_idx)

                if idx in repos_set:
                    self.reposition_kv()

            # 4. Automatic Rolling Tool-Drop Policy
            if auto_drop_keep_last_n_tools is not None:
                active_tool_indices = [
                    e.index for e in self.cached_messages if e.role == "tool" and e.active
                ]
                if len(active_tool_indices) > auto_drop_keep_last_n_tools:
                    to_drop = active_tool_indices[:-auto_drop_keep_last_n_tools]
                    freed_any = sum(self.drop_message_kv(i) for i in to_drop)
                    if freed_any > 0:
                        self.reposition_kv()

            # 5. Generate Assistant Response
            think_prefix = "" if enable_thinking else "<think>\n\n</think>\n\n"
            gen_header = f"<|im_start|>assistant\n{think_prefix}"
            header_tokens = self.llm.tokenize(gen_header.encode("utf-8"), add_bos=False, special=True)

            asst_p0 = self.logical_pos
            self._eval_tokens(header_tokens)

            gen_tokens: List[int] = []
            hit_stop = False
            for _ in range(max_tokens):
                logits = self._get_last_token_logits()
                tok = sample_next_token(
                    logits=logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                if tok in self.stop_token_ids:
                    hit_stop = True
                    break

                gen_tokens.append(tok)
                self._eval_tokens([tok])
                time.sleep(0.002)

            footer_tokens = self.llm.tokenize(b"<|im_end|>\n", add_bos=False, special=True)
            self._eval_tokens(footer_tokens)
            asst_p1 = self.logical_pos

            raw_reply = self.llm.detokenize(gen_tokens).decode("utf-8", errors="replace").strip()
            clean_content, parsed_tool_calls = parse_tool_calls_from_reply(raw_reply)

            # Record assistant turn into cached_messages
            asst_msg_dict: Dict[str, Any] = {"role": "assistant", "content": clean_content}
            if parsed_tool_calls:
                asst_msg_dict["tool_calls"] = parsed_tool_calls
            canonical_asst_chatml = format_message_to_chatml(asst_msg_dict)

            self.cached_messages.append(
                CachedMessage(
                    index=len(self.cached_messages),
                    role="assistant",
                    chatml=canonical_asst_chatml,
                    p0=asst_p0,
                    p1=asst_p1,
                    active=True,
                    token_count=asst_p1 - asst_p0,
                )
            )

            # Calculate token counts
            active_kv_tokens = sum(e.p1 - e.p0 for e in self.cached_messages if e.active)
            dropped_kv_tokens = sum(e.p1 - e.p0 for e in self.cached_messages if not e.active)

            logger.info(
                f"[Turn Summary] Prefilled: {newly_prefilled_tokens} tok | "
                f"Generated: {len(gen_tokens)} tok | "
                f"Active KV: {active_kv_tokens} tok | "
                f"Evicted KV: {dropped_kv_tokens} tok"
            )

            finish_reason = "tool_calls" if parsed_tool_calls else ("stop" if hit_stop else "length")

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": clean_content,
                            "tool_calls": parsed_tool_calls,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": newly_prefilled_tokens,
                    "completion_tokens": len(gen_tokens),
                    "total_tokens": newly_prefilled_tokens + len(gen_tokens),
                    "active_kv_tokens": active_kv_tokens,
                    "dropped_kv_tokens": dropped_kv_tokens,
                },
            }
