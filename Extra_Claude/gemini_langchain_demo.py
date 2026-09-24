"""
Comprehensive demo of ChatGoogleGenerativeAI (langchain_google_genai)

Covers: basic invoke, streaming, async, thinking/reasoning, tool calling,
structured output, image input, safety settings, token usage, response_metadata.

Setup:
    pip install langchain-google-genai python-dotenv pydantic --break-system-packages
    .env file with: GOOGLE_API_KEY=your-key-here
"""

import asyncio
import base64
from typing import Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_core.messages import HumanMessage

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Basic model setup (with thinking/reasoning control + safety settings)
# ---------------------------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    thinking_level="low",  # 'minimal' | 'low' | 'medium' | 'high' (Gemini 3+)
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)


def basic_invoke():
    print("\n=== 1. Basic invoke ===")
    result = model.invoke("What is the capital of India?")
    print("Text:", result.text)                  # provider-agnostic text extraction
    print("Usage:", result.usage_metadata)        # token counts
    print("Finish reason:", result.response_metadata.get("finish_reason"))


# ---------------------------------------------------------------------------
# 2. Streaming
# ---------------------------------------------------------------------------
def streaming_demo():
    print("\n=== 2. Streaming ===")
    full = None
    for chunk in model.stream("Write a two-line poem about the ocean."):
        print(chunk.content, end="", flush=True)
        full = chunk if full is None else full + chunk  # reassemble full message
    print("\n-- Reassembled usage:", full.usage_metadata)


# ---------------------------------------------------------------------------
# 3. Async invoke
# ---------------------------------------------------------------------------
async def async_demo():
    print("\n=== 3. Async invoke ===")
    result = await model.ainvoke("Name one river in Africa.")
    print(result.text)


# ---------------------------------------------------------------------------
# 4. Tool calling
# ---------------------------------------------------------------------------
class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def tool_calling_demo():
    print("\n=== 4. Tool calling ===")
    model_with_tools = model.bind_tools([GetWeather])
    ai_msg = model_with_tools.invoke("What's the weather like in Pune?")
    print("Tool calls:", ai_msg.tool_calls)


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
    structured_model = model.with_structured_output(Joke)  # default method='json_schema'
    joke = structured_model.invoke("Tell me a joke about programmers")
    print(joke)


# ---------------------------------------------------------------------------
# 6. Image input (multimodal)
# ---------------------------------------------------------------------------
def image_input_demo():
    print("\n=== 6. Image input ===")
    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the weather in this image in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ]
    )
    result = model.invoke([message])
    print(result.text)


# ---------------------------------------------------------------------------
# 7. Built-in Google Search tool (grounding)
# ---------------------------------------------------------------------------
def google_search_demo():
    print("\n=== 7. Google Search grounding ===")
    model_with_search = model.bind_tools([{"google_search": {}}])
    result = model_with_search.invoke("What is today's date?")
    print(result.content_blocks)


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
    google_search_demo()
