"""A simple tool-using AI agent built with LangGraph and free OpenRouter models."""

import ast
import operator
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# --- Tools -------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(3 + 4) * 2 ** 3'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


@tool
def current_time() -> str:
    """Return the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


TOOLS = [calculator, current_time]

# --- Model (OpenRouter is OpenAI-compatible) ---------------------------------

# Free (":free") models that support tool calling. Free models are rate-limited,
# so OpenRouter automatically falls back to the next model in the list.
DEFAULT_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_FALLBACKS = "nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-26b-a4b-it:free"

MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
FALLBACKS = [
    m.strip()
    for m in os.getenv("OPENROUTER_FALLBACK_MODELS", DEFAULT_FALLBACKS).split(",")
    if m.strip()
]

llm = ChatOpenAI(
    model=MODEL,
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_retries=3,
    # OpenRouter-specific: try these models in order if the primary one fails.
    extra_body={"models": [MODEL, *FALLBACKS]},
).bind_tools(TOOLS)

SYSTEM_PROMPT = SystemMessage(
    "You are a helpful assistant. Use the tools when they help answer accurately."
)

# --- Graph -------------------------------------------------------------------


def call_model(state: MessagesState):
    response = llm.invoke([SYSTEM_PROMPT, *state["messages"]])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
# If the model asked for a tool, run it; otherwise finish.
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=MemorySaver())  # memory keeps chat history


def main():
    config = {"configurable": {"thread_id": "cli"}}
    print(f"Agent ready (model: {MODEL}). Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        try:
            result = graph.invoke({"messages": [HumanMessage(user_input)]}, config)
        except Exception as exc:  # noqa: BLE001 - e.g. free-tier rate limit (429)
            print(f"Error: {exc}\nFree models are rate-limited; wait a bit and retry.")
            continue
        last = result["messages"][-1]
        used = last.response_metadata.get("model_name", MODEL)
        print(f"Agent ({used}): {last.content}")


if __name__ == "__main__":
    main()
