from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# model = ChatOpenAI(model='gpt-5-nano', temperature=1.5, reasoning_effort='minimal', max_completion_tokens=50) 
model = ChatOpenAI(model='gpt-5-nano-2025-08-07',
                    temperature=1.5,
                    reasoning_effort='minimal', max_completion_tokens=200)

# result = model.invoke("What is the capital of india")

# print(result) 

result = model.invoke("generate a very short poem on basketball")

print(result)

#content='The capital of India is New Delhi.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 8, 'prompt_tokens': 13, 'total_tokens': 21, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-nano-2025-04-14', 'system_fingerprint': 'fp_f2644ff0ca', 'id': 'chatcmpl-ER1kYYzH4PasyHsrGLz2W8H5JMzDc', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0cae3-47f5-7911-8379-4e639f9b3738-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 13, 'output_tokens': 8, 'total_tokens': 21, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

# This is the full AIMessage object langchain returns, printed because you're doing print(result) instead of print(result.content). Breaking down the fields:

# content='The capital of India is New Delhi.' — the actual answer text you want
# additional_kwargs={'refusal': None} — extra OpenAI-specific fields; refusal is None because the model didn't refuse to answer
# response_metadata — details about the API call itself:
# token_usage — 13 prompt tokens in, 8 completion tokens out, 21 total
# model_name: 'gpt-4.1-nano-2025-04-14' — the exact model snapshot that served the request
# finish_reason: 'stop' — the model finished naturally (not cut off by a length limit or anything)
# system_fingerprint — an OpenAI identifier for the backend configuration that generated this response, useful for reproducibility debugging
# service_tier: 'default' — which OpenAI compute tier handled the request
# id='lc_run--...' — langchain's internal run ID for tracing, not from OpenAI
# tool_calls=[] and invalid_tool_calls=[] — empty since you didn't bind any tools to this call
# usage_metadata — langchain's normalized (provider-agnostic) version of token usage, same numbers as above, restated in a consistent schema across different LLM providers 

# That id (lc_run--01a0cae3-47f5-7911-8379-4e639f9b3738-0) is LangChain's internal run identifier, generated per invocation. It's mainly useful for tracing and debugging rather than anything you'd show to end users. A few practical uses:

# 1. Correlate with LangSmith traces
# If you have LangSmith tracing enabled (LANGCHAIN_TRACING_V2=true), this run ID maps to a specific trace in your LangSmith dashboard. You can look up that exact call, see the full input, output, latency, and cost in the UI.

# 2. Access it in code

# python
# result = llm.invoke("What is the capital of India")
# print(result.id)  # 'lc_run--01a0cae3-...'

# 3. Use it for logging/debugging in production
# If you're building an app and logging every LLM call, storing this ID alongside your own request ID lets you cross-reference "which LangChain run produced this specific response" later, especially useful when debugging a chain with multiple LLM calls.

# 4. Feedback loops with LangSmith
# If you want to programmatically attach feedback (like a thumbs up/down or a correctness score) to a specific run, you pass this run ID to LangSmith's client:

# python
# from langsmith import Client

# client = Client()
# client.create_feedback(
#     run_id=result.id,
#     key="correctness",
#     score=1
# )

# Note: this ID is different from OpenAI's own response ID (id='chatcmpl-ER1kYYzH4PasyHsrGLz2W8H5JMzDc' inside response_metadata), which is what you'd use if you needed to reference the call directly with OpenAI, for instance in their usage dashboard or support requests.


