"""Pydantic schemas for OpenAI-compatible API requests and responses."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with AKV custom controls."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    model: str = "qwen3-4b"
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: Optional[int] = Field(default=512, alias="max_completion_tokens")
    stop: Optional[Union[str, List[str]]] = None

    # AKV Cache Management Parameters
    drop_message: Optional[Dict[Union[str, int], List[int]]] = None
    reposition: Optional[List[Union[str, int]]] = None
    auto_drop_keep_last_n_tools: Optional[int] = None
    enable_thinking: bool = False

    def get_max_tokens(self) -> int:
        return self.max_tokens if self.max_tokens is not None else 512


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    active_kv_tokens: int
    dropped_kv_tokens: int


class ChatChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local-akv"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class CacheStatusResponse(BaseModel):
    logical_pos: int
    active_kv_tokens: int
    dropped_kv_tokens: int
    total_messages_tracked: int
    active_messages: int
    messages: List[Dict[str, Any]]
