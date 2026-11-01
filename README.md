# Simple LangGraph Agent (OpenRouter)

A minimal tool-calling AI agent built with [LangGraph](https://langchain-ai.github.io/langgraph/)
and a cheap model from [OpenRouter](https://openrouter.ai) (default: `openai/gpt-4o-mini`,
~$0.15 / 1M input tokens, $0.60 / 1M output tokens).

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
Agent: 17 * 23 + 5 = 396.
```

## Change the model

Set `OPENROUTER_MODEL` in `.env` to any OpenRouter model that supports tool calling, e.g.
`google/gemini-2.0-flash-001` or `deepseek/deepseek-chat`.

## Add your own tool

Write a function with the `@tool` decorator and a docstring in `agent.py`, then add it to `TOOLS`.
