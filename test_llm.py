import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
client = OpenAI(api_key=api_key, base_url=base_url)

prompt = "You are a triage assistant. Return STRICT JSON with keys urgency, specialty, rationale. Test case: patient with chest pain."
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    temperature=0.0,
    messages=[
        {"role": "system", "content": "You are a careful clinical triage assistant."},
        {"role": "user", "content": prompt},
    ],
)
print("Raw response:")
print(response)
print("\nChoices:")
for c in response.choices:
    print(c.message)
    print("\nContent:")
    print(repr(c.message.content))