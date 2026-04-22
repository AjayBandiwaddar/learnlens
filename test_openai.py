import openai, os
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=50,
    messages=[{"role": "user", "content": "Reply only with JSON: {\"relevance\": 0.8, \"coherence\": 0.9, \"uncertainty\": 0.7}"}]
)
print(resp.choices[0].message.content)