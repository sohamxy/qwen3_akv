from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="qwen3-4b",
    messages=[
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": "How do fishes swim?"},
    ],
    temperature=0.7,
    max_tokens=256,
    extra_body={
        "enable_thinking": False  # Set to True if you want Qwen3's <think> reasoning block
    },
)

print(response.choices[0].message.content)