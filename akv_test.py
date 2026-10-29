from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_record",
            "description": "Read a warehouse record by ID.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
    }
]

messages = [
    {"role": "system", "content": "You are a helpful assistant. Summarize each record briefly before calling the next tool."},
    {"role": "user", "content": "First read record A, then read record B, and compare their prices."},
]

# --- Turn 1: Model calls read_record("A") ---
resp1 = client.chat.completions.create(
    model="qwen3-4b",
    messages=messages,
    tools=tools,
)
msg1 = resp1.choices[0].message
messages.append(msg1.model_dump(exclude_none=True))

# Append bulky Tool Result A (Message index 3)
messages.append({
    "role": "tool",
    "tool_call_id": msg1.tool_calls[0].id,
    "content": "Record A: Price is $120. " + ("AUDIT_LOG_PASS_OK " * 80),
})

# --- Turn 2: Model reads A, summarizes it in its own KV, and calls read_record("B") ---
resp2 = client.chat.completions.create(
    model="qwen3-4b",
    messages=messages,
    tools=tools,
)
msg2 = resp2.choices[0].message
messages.append(msg2.model_dump(exclude_none=True))

# Append bulky Tool Result B (Message index 5)
messages.append({
    "role": "tool",
    "tool_call_id": msg2.tool_calls[0].id,
    "content": "Record B: Price is $95. " + ("AUDIT_LOG_PASS_OK " * 80),
})

# --- Turn 3: Pass AKV `drop_message` & `reposition` via OpenAI SDK `extra_body` ---
# After processing Message 5 (Record B), drop Message 3 (Record A's bulky KV) and compact!
resp3 = client.chat.completions.create(
    model="qwen3-4b",
    messages=messages,
    tools=tools,
    extra_body={
        "drop_message": {"5": [3]},  # Drop Record A's tool KV after processing Record B
        "reposition": [5],           # Shift RoPE positions to close the gap
    },
)

print("\nFinal Answer:\n", resp3.choices[0].message.content)
print("\nServer KV Usage Stats:\n", resp3.usage)
