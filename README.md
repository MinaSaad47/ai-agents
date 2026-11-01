# Simple LangGraph Agent (OpenRouter)

A minimal tool-calling AI agent built with [LangGraph](https://langchain-ai.github.io/langgraph/)
and **free** models from [OpenRouter](https://openrouter.ai) — $0 per token.

- Default model: `google/gemma-4-31b-it:free`
- Fallbacks (used automatically if the default is rate-limited):
  `nvidia/nemotron-3-ultra-550b-a55b:free`, `google/gemma-4-26b-a4b-it:free`

> Free models have rate limits (requests per minute/day). If you hit them, wait a bit or
> switch to a cheap paid model like `openai/gpt-4o-mini`.

```
START → agent ──(tool call?)──► tools
          ▲                       │
          └───────────────────────┘
          │ (no tool call)
          ▼
         END
```

The agent has two example tools: a safe `calculator` and `current_time`. It also remembers the
conversation within a session (via LangGraph's `MemorySaver`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your OpenRouter key in .env
```

## Run

```bash
python agent.py
```

```
You: What is 17 * 23 + 5?
Agent (google/gemma-4-31b-it:free): 17 * 23 + 5 = 396.
```

## Change the model

Set `OPENROUTER_MODEL` (and optionally `OPENROUTER_FALLBACK_MODELS`) in `.env` to any
OpenRouter model that supports tool calling. Browse free ones at
<https://openrouter.ai/models?max_price=0&supported_parameters=tools>.

## Add your own tool

Write a function with the `@tool` decorator and a docstring in `agent.py`, then add it to `TOOLS`.
