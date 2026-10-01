"""
LLM Chat CLI: interactive multi-model chat driven by config.yaml.

Models and generation settings (temperature, top_p, max_tokens, stop)
live in config.yaml as presets. Switch presets and tweak settings live
from the chat. History is kept across switches, even between providers.
Every request's tokens, cost and latency are tracked (see /usage), and
old messages are trimmed to keep the history under a token budget.

Run:  python chat.py      Type /help for commands.
"""
import os
import random
import time

import yaml
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from context import TOKENIZER, count_messages, trim_history
from usage import UsageTracker, fmt_cost

load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

SETTINGS = CONFIG.get("settings", {})
RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError)

# Generation settings that can be sent to the API, with validation rules
GEN_SETTINGS = {
    "temperature": (float, 0.0, 2.0),
    "top_p": (float, 0.0, 1.0),
    "max_tokens": (int, 1, 100_000),
}

HELP = """Commands:
  /presets                 list presets from config.yaml
  /use <preset>            switch preset (history is kept)
  /set <setting> <value>   change temperature, top_p or max_tokens ("none" to unset)
  /stop <text> | /stop off set or clear a stop sequence
  /config                  show the active settings
  /usage                   show tokens, cost and latency for this session
  /context                 show history size against the token budget
  /budget <tokens>         change the history token budget
  /reset                   clear conversation history
  /help                    show this help
  /exit                    quit"""


# ---------------- clients and presets ----------------

_clients = {}


def get_client(provider):
    """Create one client per provider, reusing it afterwards."""
    if provider not in _clients:
        p = CONFIG["providers"][provider]
        key = os.getenv(p["api_key_env"])
        if not key:
            raise ValueError(f"{p['api_key_env']} is not set in .env")
        _clients[provider] = OpenAI(
            api_key=key,
            base_url=p["base_url"],
            timeout=SETTINGS.get("timeout", 60),
            max_retries=0,  # we handle retries ourselves
        )
    return _clients[provider]


def load_preset(name):
    """Return a copy of a preset, so /set changes don't touch the config."""
    if name not in CONFIG["presets"]:
        raise KeyError(f"No preset named '{name}'. Try /presets.")
    preset = dict(CONFIG["presets"][name])
    get_client(preset["provider"])  # fail now if the key is missing
    return preset


def build_request(preset, messages):
    """Only send settings that are actually set; the rest use provider defaults."""
    kwargs = {
        "model": preset["model"],
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for key in (*GEN_SETTINGS, "stop"):
        if preset.get(key) is not None:
            kwargs[key] = preset[key]
    return kwargs


# ---------------- API calls ----------------

def create_with_retry(preset, messages):
    """Start a streaming request, retrying temporary failures with exponential backoff."""
    client = get_client(preset["provider"])
    attempts = SETTINGS.get("max_attempts", 4)
    base_delay = 1.0

    for attempt in range(1, attempts + 1):
        try:
            return client.chat.completions.create(**build_request(preset, messages))
        except RateLimitError as e:
            # 429 can mean "slow down" (retry) or "out of quota" (never retry)
            if "quota" in str(e).lower() or attempt == attempts:
                raise
            reason = "rate limited"
        except RETRYABLE as e:
            if attempt == attempts:
                raise
            reason = type(e).__name__

        delay = base_delay * 2 ** (attempt - 1) + random.uniform(0, 0.5)
        print(f"\n  ({reason}, retry {attempt}/{attempts - 1} in {delay:.1f}s)", end="", flush=True)
        time.sleep(delay)


def stream_reply(preset, messages):
    """Stream one reply, print it live, and return (text, ttft, duration, usage, finish_reason)."""
    start = time.perf_counter()
    ttft, parts, usage, finish = None, [], None, None

    for chunk in create_with_retry(preset, messages):
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            if choice.delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                parts.append(choice.delta.content)
                print(choice.delta.content, end="", flush=True)

    print()
    duration = time.perf_counter() - start
    return "".join(parts), ttft, duration, usage, finish


# ---------------- context budget ----------------

def context_budget(state):
    """
    Tokens the history may use: our own cap, but never more than the model's
    window minus room for the reply.
    """
    preset = state["preset"]
    limits = []
    if state.get("max_context"):
        limits.append(state["max_context"])
    if preset.get("context_window"):
        limits.append(preset["context_window"] - (preset.get("max_tokens") or 1024))
    return min(limits) if limits else None


# ---------------- commands ----------------

def describe(name, preset):
    shown = [f"{k}={preset[k]}" for k in (*GEN_SETTINGS, "stop") if preset.get(k) is not None]
    return f"{name}: {preset['provider']} / {preset['model']}  " + (", ".join(shown) or "provider defaults")


def cmd_set(preset, args):
    if len(args) != 2 or args[0] not in GEN_SETTINGS:
        print(f"Usage: /set <{'|'.join(GEN_SETTINGS)}> <value|none>")
        return
    key, raw = args
    if raw.lower() == "none":
        preset.pop(key, None)
        print(f"{key} unset (provider default).")
        return
    kind, low, high = GEN_SETTINGS[key]
    try:
        value = kind(raw)
    except ValueError:
        print(f"{key} must be a{'n integer' if kind is int else ' number'}.")
        return
    if not low <= value <= high:
        print(f"{key} must be between {low} and {high}.")
        return
    preset[key] = value
    print(f"{key} = {value}")


def handle_command(line, state):
    """Run a /command. Returns False when the user wants to quit."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit":
        return False
    elif cmd == "/help":
        print(HELP)
    elif cmd == "/reset":
        del state["history"][1:]  # keep only the system message
        print("History cleared.")
    elif cmd == "/presets":
        for name, p in CONFIG["presets"].items():
            marker = "*" if name == state["name"] else " "
            print(f" {marker} {describe(name, p)}")
    elif cmd == "/use":
        try:
            state["preset"] = load_preset(rest.strip())
            state["name"] = rest.strip()
            print(f"Now using {describe(state['name'], state['preset'])}")
        except (KeyError, ValueError) as e:
            print(e.args[0])
    elif cmd == "/set":
        cmd_set(state["preset"], rest.split())
    elif cmd == "/stop":
        if not rest or rest.lower() == "off":
            state["preset"].pop("stop", None)
            print("Stop sequence cleared.")
        else:
            state["preset"]["stop"] = [rest]
            print(f"Generation will stop before: {rest!r}")
    elif cmd == "/config":
        print(describe(state["name"], state["preset"]))
    elif cmd == "/usage":
        print(state["tracker"].summary())
    elif cmd == "/context":
        history = state["history"]
        used = count_messages(history)
        budget = context_budget(state)
        print(f"Messages:  {len(history) - 1} (+ system prompt)")
        print(f"Estimated: ~{used:,} tokens" + (f" of {budget:,} budget ({used / budget:.0%})" if budget else ""))
        print(f"Counting:  {TOKENIZER}")
    elif cmd == "/budget":
        try:
            value = int(rest)
            if value < 100:
                raise ValueError
            state["max_context"] = value
            print(f"History budget set to {value:,} tokens (effective: {context_budget(state):,}).")
        except ValueError:
            print("Usage: /budget <tokens>  (100 or more)")
    else:
        print(f"Unknown command {cmd}. Type /help.")
    return True


# ---------------- main loop ----------------

def main():
    name = CONFIG.get("default_preset", next(iter(CONFIG["presets"])))
    state = {
        "name": name,
        "preset": load_preset(name),
        "history": [{"role": "system", "content": SETTINGS.get("system_prompt", "You are a helpful assistant.")}],
        "tracker": UsageTracker(CONFIG.get("pricing"), SETTINGS.get("usage_log")),
        "max_context": SETTINGS.get("max_context_tokens"),
    }
    print(f"Using {describe(name, state['preset'])}")
    print("Type /help for commands.")

    while True:
        try:
            user = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user:
            continue
        if user.startswith("/"):
            if not handle_command(user, state):
                break
            continue

        history = state["history"]
        message = {"role": "user", "content": user}
        budget = context_budget(state)

        # A single message bigger than the whole budget can't be sent
        if budget and count_messages([history[0], message]) > budget:
            print(f"[Message too long: ~{count_messages([message]):,} tokens, budget is {budget:,}. "
                  "Shorten it or raise /budget.]")
            continue

        history.append(message)
        if budget:
            dropped = trim_history(history, budget)
            if dropped:
                print(f"[context: dropped {dropped} oldest messages to stay under ~{budget:,} tokens]")

        print("AI: ", end="")
        try:
            reply, ttft, duration, usage, finish = stream_reply(state["preset"], history)
        except APIError as e:
            history.pop()  # drop the unanswered message so history stays valid
            print(f"\n[Request failed: {type(e).__name__}. Try again, or /use another preset.]")
            continue

        history.append({"role": "assistant", "content": reply})

        preset = state["preset"]
        stats = [preset["model"]]
        if ttft is not None:
            stats.append(f"TTFT {ttft:.2f}s")
        if usage:
            cost = state["tracker"].record(
                preset["provider"], preset["model"],
                usage.prompt_tokens, usage.completion_tokens, ttft, duration,
            )
            stats.append(f"in {usage.prompt_tokens}")
            stats.append(f"out {usage.completion_tokens}")
            stats.append(fmt_cost(cost))
        if finish and finish != "stop":
            stats.append(f"finish: {finish}")
        print(f"[{' | '.join(stats)}]")

    if state["tracker"].has_data:
        print("\nSession usage:")
        print(state["tracker"].summary())
    print("Bye!")


if __name__ == "__main__":
    main()