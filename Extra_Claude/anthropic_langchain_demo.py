"""
Comprehensive demo of ChatAnthropic (langchain_anthropic)

Covers: basic invoke, streaming, async, tool calling (+ strict mode), multimodal
image input, extended thinking, effort, structured output, prompt caching,
token usage/response metadata, and client-side tools (bash, text editor, memory).

Setup:
    pip install -U langchain-anthropic python-dotenv pydantic --break-system-packages
    .env file with: ANTHROPIC_API_KEY=your-key-here

Note: Some sections use beta/advanced features (task budgets, context management,
MCP, web search) that require specific model support or beta headers. Those are
shown as commented reference snippets rather than run live, since they depend on
your account's access level.
"""

import asyncio
import base64
import json
import subprocess
from typing import Literal, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool

load_dotenv()

# Use a current, active model throughout (see model-deprecations page before hardcoding).
MODEL_NAME = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# 1. Basic invoke
# ---------------------------------------------------------------------------
def basic_invoke():
    print("\n=== 1. Basic invoke ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=200)
    result = model.invoke("What is the capital of India?")
    print("Content:", result.content)              # Anthropic AIMessage content (string here)
    print("Usage:", result.usage_metadata)          # normalized token counts across providers
    print("Response metadata:", result.response_metadata)  # raw Anthropic fields (id, stop_reason, ...)


# ---------------------------------------------------------------------------
# 2. Streaming
# ---------------------------------------------------------------------------
def streaming_demo():
    print("\n=== 2. Streaming ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=100)
    full = None
    for chunk in model.stream("Write a two-line poem about rivers."):
        print(chunk.content, end="", flush=True)
        full = chunk if full is None else full + chunk  # AIMessageChunk supports += to reassemble
    print("\n-- Reassembled usage:", full.usage_metadata)


# ---------------------------------------------------------------------------
# 3. Async invoke
# ---------------------------------------------------------------------------
async def async_demo():
    print("\n=== 3. Async invoke ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=100)
    result = await model.ainvoke("Name one river in Africa.")
    print(result.content)


# ---------------------------------------------------------------------------
# 4. Tool calling (standard + strict mode)
# ---------------------------------------------------------------------------
class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    location: str = Field(description="The city and state, e.g. San Francisco, CA")


def tool_calling_demo():
    print("\n=== 4. Tool calling ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=300)

    # Standard tool calling
    model_with_tools = model.bind_tools([GetWeather])
    ai_msg = model_with_tools.invoke("What's the weather in Pune?")
    print("Tool calls (standard):", ai_msg.tool_calls)

    # Strict tool calling: guarantees schema-compliant args, no type mismatches
    # or missing fields. Requires langchain-anthropic>=1.1.0 and a supporting model.
    model_with_strict_tools = model.bind_tools([GetWeather], strict=True)
    ai_msg_strict = model_with_strict_tools.invoke("What's the weather in Mumbai?")
    print("Tool calls (strict):", ai_msg_strict.tool_calls)


# ---------------------------------------------------------------------------
# 5. Structured output
# ---------------------------------------------------------------------------
class Joke(BaseModel):
    """Joke to tell the user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


def structured_output_demo():
    print("\n=== 5. Structured output ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=300)
    structured_model = model.with_structured_output(Joke)  # guarantees schema-adherent response
    joke = structured_model.invoke("Tell me a joke about programmers")
    print(joke)


# ---------------------------------------------------------------------------
# 6. Multimodal image input
# ---------------------------------------------------------------------------
def image_input_demo():
    print("\n=== 6. Image input ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=200)
    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )
    # Anthropic accepts a plain "url" field directly (no need to base64-encode
    # unless you want to send inline bytes instead of a hosted link).
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the weather in this image in one sentence."},
            {"type": "image", "url": image_url},
        ]
    )
    result = model.invoke([message])
    print(result.content)


# ---------------------------------------------------------------------------
# 7. Extended thinking (reasoning)
# ---------------------------------------------------------------------------
def extended_thinking_demo():
    print("\n=== 7. Extended thinking ===")
    # Sonnet/earlier models need an explicit token budget for thinking.
    # (Opus 4.6+ supports adaptive thinking with no budget needed.)
    model = ChatAnthropic(
        model=MODEL_NAME,
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    response = model.invoke("What is the cube root of 50.653?")
    print(json.dumps(response.content_blocks, indent=2, default=str))


# ---------------------------------------------------------------------------
# 8. Effort (token-spend control) — only on models that support it
# ---------------------------------------------------------------------------
def effort_demo():
    print("\n=== 8. Effort ===")
    # 'effort' controls how many tokens Claude spends responding.
    # Generally available on Opus 4.6/4.5 class models — using a lighter model
    # here for cost, so this may be a no-op / unsupported depending on your access.
    try:
        model = ChatAnthropic(model=MODEL_NAME, max_tokens=300, effort="low")
        response = model.invoke("Briefly compare REST and GraphQL.")
        print(response.content)
    except Exception as e:
        print("Effort not supported on this model/account:", e)


# ---------------------------------------------------------------------------
# 9. Prompt caching (automatic)
# ---------------------------------------------------------------------------
def prompt_caching_demo():
    print("\n=== 9. Prompt caching ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=200)

    # Pull a real chunk of text to make caching worth demonstrating.
    readme = httpx.get(
        "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
    ).text[:3000]

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a technology expert."},
                {"type": "text", "text": readme},
            ],
        },
        {"role": "user", "content": "What is LangChain, in one sentence?"},
    ]

    # First call: cache miss (cache_creation > 0)
    response_1 = model.invoke(messages, cache_control={"type": "ephemeral"})
    print("First call cache usage:", response_1.usage_metadata["input_token_details"])

    # Second call with the same prefix: cache hit (cache_read > 0, cheaper/faster)
    response_2 = model.invoke(messages, cache_control={"type": "ephemeral"})
    print("Second call cache usage:", response_2.usage_metadata["input_token_details"])


# ---------------------------------------------------------------------------
# 10. Citations
# ---------------------------------------------------------------------------
def citations_demo():
    print("\n=== 10. Citations ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=300)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "The grass is green. The sky is blue.",
                    },
                    "title": "My Document",
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": "What color is the grass and sky?"},
            ],
        }
    ]
    response = model.invoke(messages)
    print(json.dumps(response.content, indent=2, default=str))


# ---------------------------------------------------------------------------
# 11. Client-side tool: bash (runs locally — use with caution, no sandboxing here)
# ---------------------------------------------------------------------------
@tool
def bash(command: str) -> str:
    """Execute a bash/shell command and return its output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {e}"


def bash_tool_demo():
    print("\n=== 11. Bash tool (client-side, unsandboxed demo only) ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=300)
    model_with_bash = model.bind_tools([bash])

    messages = [HumanMessage("What does 'echo hello' print? Run it to check.")]
    response = model_with_bash.invoke(messages)

    # Agent loop: keep executing tool calls until the model stops requesting them
    while response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            result = bash.invoke(tool_call)  # returns a ToolMessage automatically
            tool_messages.append(result)
        messages = [*messages, response, *tool_messages]
        response = model_with_bash.invoke(messages)

    print(response.content)


# ---------------------------------------------------------------------------
# 12. Client-side tool: simple in-memory "memory" tool
# ---------------------------------------------------------------------------
memory_store: dict[str, str] = {"/memories/interests": "User enjoys Python and hiking"}


@tool
def memory(command: Literal["view", "create"], path: str, content: Optional[str] = None) -> str:
    """View or create entries in a simple persistent memory store."""
    if command == "view":
        if path == "/memories":
            return "\n".join(memory_store.keys()) or "No memories stored"
        return memory_store.get(path, f"No memory at {path}")
    elif command == "create":
        memory_store[path] = content or ""
        return f"Created memory at {path}"
    return f"Unsupported command: {command}"


def memory_tool_demo():
    print("\n=== 12. Memory tool (simplified, in-memory only) ===")
    model = ChatAnthropic(model=MODEL_NAME, max_tokens=300)
    model_with_memory = model.bind_tools([memory])

    messages = [HumanMessage("What are my interests, based on stored memory?")]
    response = model_with_memory.invoke(messages)

    while response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            result = memory.invoke(tool_call)
            tool_messages.append(result)
        messages = [*messages, response, *tool_messages]
        response = model_with_memory.invoke(messages)

    print(response.content)


# ---------------------------------------------------------------------------
# 13. Token counting (before sending a request)
# ---------------------------------------------------------------------------
def token_counting_demo():
    print("\n=== 13. Token counting ===")
    model = ChatAnthropic(model=MODEL_NAME)
    num_tokens = model.get_num_tokens_from_messages(
        [{"role": "user", "content": "How many tokens is this sentence?"}]
    )
    print("Estimated tokens:", num_tokens)


# ---------------------------------------------------------------------------
# Reference-only snippets (not run): features needing beta headers, external
# infra, or higher-tier model/account access.
# ---------------------------------------------------------------------------
"""
# Server-side web search (needs web search enabled on your account):
from anthropic.types import WebSearchTool20260209Param
search_tool = WebSearchTool20260209Param(name="web_search", type="web_search_20260209", max_uses=3)
model_with_search = model.bind_tools([search_tool])
model_with_search.invoke("How do I update a web app to TypeScript 5.5?")

# Remote MCP connector:
model = ChatAnthropic(model="claude-sonnet-4-6", mcp_servers=[
    {"type": "url", "url": "https://docs.langchain.com/mcp", "name": "LangChain Docs"}
])

# Context management / automatic compaction (needs beta headers):
model = ChatAnthropic(
    model="claude-sonnet-4-6",
    betas=["context-management-2025-06-27"],
    context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
)

# Task budgets (Opus 4.7+, beta):
model = ChatAnthropic(
    model="claude-opus-4-8",
    output_config={"effort": "high", "task_budget": {"type": "tokens", "total": 128_000}},
)
"""


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    basic_invoke()
    streaming_demo()
    asyncio.run(async_demo())
    tool_calling_demo()
    structured_output_demo()
    image_input_demo()
    extended_thinking_demo()
    effort_demo()
    prompt_caching_demo()
    citations_demo()
    bash_tool_demo()
    memory_tool_demo()
    token_counting_demo()
