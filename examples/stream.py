"""
Minimal example: a single streaming request.

Sends one prompt to Gemini (through its OpenAI-compatible endpoint),
prints the reply as it streams in, then shows time-to-first-token (TTFT),
token usage and why generation stopped.

Run:  python examples/stream.py
"""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-3.5-flash-lite"

# Try lowering this to 20 to see the reply cut off and finish_reason become "length"
# MAX_TOKENS = 30
MAX_TOKENS = 300

start = time.perf_counter()
ttft, usage, finish = None, None, None

stream = client.chat.completions.create(
    model=MODEL,
    max_tokens=MAX_TOKENS,
    messages=[{"role": "user", "content": "Explain tokens in 3 sentences."}],
    stream=True,
    stream_options={"include_usage": True},  # usage arrives in a final chunk
)

for chunk in stream:
    if chunk.usage:
        usage = chunk.usage
    if chunk.choices:
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish = choice.finish_reason
        if choice.delta.content:
            if ttft is None:
                ttft = time.perf_counter() - start
            print(choice.delta.content, end="", flush=True)

total = time.perf_counter() - start
print("\n")
print(f"TTFT:          {ttft:.2f}s" if ttft is not None else "TTFT:          n/a")
print(f"Total time:    {total:.2f}s")
if usage:
    print(f"Input tokens:  {usage.prompt_tokens}")
    print(f"Output tokens: {usage.completion_tokens}")
print(f"Finish reason: {finish}")