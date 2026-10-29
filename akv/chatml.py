"""Qwen3 ChatML formatting, tool calling XML prompts, and reply parsing."""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


def extract_text_content(content: Any) -> str:
    """Extracts raw string text from diverse message content types (str, list of blocks, None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif "text" in part:
                    parts.append(str(part["text"]))
        return "".join(parts)
    return str(content)


def build_tools_instruction(tools: List[Dict[str, Any]]) -> str:
    """Builds the Qwen XML tool calling system prompt block."""
    tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return (
        f"\n\n# Tools\n\n"
        f"You may call one or more functions to assist with the user query.\n\n"
        f"You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tools_str}\n</tools>\n\n"
        f"For each function call, return a json object with function name and arguments "
        f"within <tool_call></tool_call> XML tags:\n"
        f"<tool_call>\n"
        f"{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n"
        f"</tool_call>"
    )


def format_message_to_chatml(
    msg: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
    is_first_system: bool = False,
) -> str:
    """Formats an OpenAI-style message dictionary into a Qwen ChatML string block."""
    role = msg.get("role", "user")
    content = extract_text_content(msg.get("content"))

    if role == "system":
        if is_first_system and tools:
            content = f"{content}{build_tools_instruction(tools)}"
        return f"<|im_start|>system\n{content}<|im_end|>\n"

    elif role == "user":
        return f"<|im_start|>user\n{content}<|im_end|>\n"

    elif role == "assistant":
        tool_calls = msg.get("tool_calls") or []
        tc_blocks = ""
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                tc = tc.model_dump()
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "")
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tc_json = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
            tc_blocks += f"\n<tool_call>\n{tc_json}\n</tool_call>"

        body = f"{content}{tc_blocks}".strip()
        return f"<|im_start|>assistant\n{body}<|im_end|>\n"

    elif role == "tool":
        return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"

    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def parse_tool_calls_from_reply(text: str) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """Parses tool calls from raw assistant reply text and strips them from content.
    
    Returns:
        (clean_content, tool_calls_list)
    """
    # Strip empty <think></think> blocks if present
    cleaned = re.sub(r"<think>\s*</think>", "", text).strip()

    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return cleaned or None, None

    tool_calls: List[Dict[str, Any]] = []
    for m in matches:
        raw_call = m.group(1).strip()
        # Strip potential markdown code fences: ```json ... ```
        if raw_call.startswith("```"):
            raw_call = re.sub(r"^```(?:json)?\s*", "", raw_call)
            raw_call = re.sub(r"\s*```$", "", raw_call).strip()

        try:
            data = json.loads(raw_call)
            # Handle potential array of tool calls inside single tag
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                args = item.get("arguments") or item.get("parameters") or {}
                args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str,
                    },
                })
        except json.JSONDecodeError:
            continue

    clean_content = pattern.sub("", cleaned).strip() or None
    return clean_content, (tool_calls if tool_calls else None)
