# LLM Chat CLI

A command-line chatbot that talks to multiple LLMs through one interface, with streaming replies, configurable model presets, usage and cost tracking, automatic retries, and context-window management.

## Project structure

```
llm-chat-cli/
├── chat.py            # CLI loop, commands, providers, streaming, retries
├── usage.py           # token, cost and latency tracking + JSONL log
├── context.py         # token counting and history trimming
├── config.yaml        # providers, model presets, pricing, settings
├── examples/
│   └── stream.py      # minimal single streaming request
├── requirements.txt
├── .env.example       # which API keys are needed
└── .gitignore
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/mandalika-sravani/llm-chat-cli.git
cd llm-chat-cli
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and add your keys:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the keys you have. A free Gemini key from [Google AI Studio](https://aistudio.google.com) is enough to run every Gemini preset. The OpenAI preset needs an OpenAI key with API credits.

```
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
```

Then start the chat:

```bash
python chat.py
```
