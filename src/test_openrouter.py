from dotenv import load_dotenv
import os
import openai  # old 0.28.1 client

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY not set in .env")

# Configure OpenRouter
openai.api_key = api_key
openai.api_base = base_url

resp = openai.ChatCompletion.create(
    model="openrouter/free",  # or another OpenRouter model slug you want to test
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
)

print(resp.choices[0].message["content"])