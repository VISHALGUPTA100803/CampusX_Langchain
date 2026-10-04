from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07', reasoning_effort='minimal')

class Review(TypedDict):
    keyTheme: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review as pos for positive or neg for negative"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

# The square brackets specify the type and metadata; they don’t make Annotated a list.

structured_model = model.with_structured_output(Review, strict=True)

prompt = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""

result = structured_model.invoke(prompt)

print(result)



# with_structured_output
# Big picture: there usually isnt a natural-language "prompt" added at all

# This is the key thing to understand first, and it directly explains why json_mode's docstring says "you must include instructions for formatting the output into the desired schema into the model call", that line is a hint that for the other two methods (json_schema, function_calling), no extra text gets added to your prompt at all. Instead, the schema is sent as a separate, structured field in the API request, parallel to your messages, not injected into them as text.

# Your actual invoke(prompt) call still sends exactly the prompt you wrote. What changes is what else gets bundled into the HTTP request to OpenAI's /v1/chat/completions (or /v1/responses) endpoint alongside it.

# Method 1: "json_schema" (your case, the default)

# Internally, with_structured_output converts your Pydantic/TypedDict class into a JSON Schema object (via convert_to_openai_tool, mentioned explicitly in the docstring), then attaches it to the request as a response_format parameter:

# json
# {
#   "model": "gpt-5-nano-2025-08-07",
#   "messages": [
#     {"role": "user", "content": "I recently upgraded to the Samsung Galaxy S24 Ultra..."}
#   ],
#   "response_format": {
#     "type": "json_schema",
#     "json_schema": {
#       "name": "Review",
#       "strict": true,
#       "schema": {
#         "type": "object",
#         "properties": {
#           "keyTheme": {"type": "array", "items": {"type": "string"}, "description": "Write down all the key themes..."},
#           "summary": {"type": "string", "description": "A brief summary of the review"},
#           "sentiment": {"type": "string", "enum": ["pos", "neg"], "description": "Return sentiment..."},
#           "pros": {"type": ["array", "null"], "items": {"type": "string"}},
#           "cons": {"type": ["array", "null"], "items": {"type": "string"}},
#           "name": {"type": ["string", "null"]}
#         },
#         "required": ["keyTheme", "summary", "sentiment", "pros", "cons", "name"],
#         "additionalProperties": false
#       }
#     }
#   }
# }

# This is literally the mechanism from your previous question, your Annotated[..., "description"] strings become the "description" fields inside this schema, that's how the model learns "pos means positive." And "additionalProperties": false is exactly the flag that gets omitted/not enforced when strict isn't explicitly True, directly explaining the earlier bug.

# Your original messages list is sent completely unmodified, no injected formatting instructions, OpenAI's inference servers handle constraining the output to match response_format at the token-generation level itself (this is a real architectural feature on OpenAI's side, using constrained decoding/grammar-based sampling, not just "the model tries harder").

# Method 2: "function_calling"

# Same schema conversion, but instead of response_format, it gets sent as a fake tool definition:

# json
# {
#   "model": "...",
#   "messages": [...],
#   "tools": [
#     {
#       "type": "function",
#       "function": {
#         "name": "Review",
#         "description": "...",
#         "parameters": { /* same JSON schema as above */ }
#       }
#     }
#   ],
#   "tool_choice": {"type": "function", "function": {"name": "Review"}}
# }

# tool_choice forces the model to call this specific "tool" rather than respond in plain text. The model's response comes back as a tool call (additional_kwargs.tool_calls, exactly the structure you saw in the AnswerWithJustification example output in the docstring), and LangChain then parses the tool call's arguments JSON string into your schema object, rather than parsing message content directly.

# Method 3: "json_mode"

# This is the one that genuinely does put the burden on your prompt text, per the docstring's explicit warning. The request just sets:

# json
# {"response_format": {"type": "json_object"}}

# No schema is sent to OpenAI at all in this mode, that flag only guarantees "the output will be syntactically valid JSON," nothing about which keys or structure. This is why the docstring's json_mode example manually writes "Make sure to return a JSON blob with keys 'answer' and 'justification'" directly into the prompt text itself, the schema constraint has to live in your words, since there's no structural enforcement backing it up at all.

# After the response comes back: parsing step

# Once OpenAI's response arrives, LangChain runs a parser (which differs by method, extracting from content for json_schema/json_mode, or from tool_calls[0].function.arguments for function_calling), turning the raw JSON string into either:

# a Pydantic instance (if your schema was a Pydantic class, validated along the way)
# a plain dict (if TypedDict/raw JSON schema, unvalidated, per the docstring's explicit note)
# Summary table of what actually gets sent where
# Method	What's added to the request	Where schema lives	Enforcement
# json_schema	response_format field	Separate structured field	Server-side constrained decoding (strongest, especially with strict=True)
# function_calling	tools + tool_choice field	Inside a fake tool definition	Server-side, via tool-calling constraints
# json_mode	response_format: {"type": "json_object"} only	Nowhere structural, must be in your prompt text	Weakest, only guarantees valid JSON syntax, not shape

# Bottom line: for the method you're using (json_schema, the default), your actual prompt text is untouched, the schema travels to OpenAI as a completely separate, structured part of the API payload, and OpenAI's inference engine uses that schema to constrain token generation directly at the model level, this is fundamentally different from "the model reads your schema as an instruction," it's closer to a hard decoding constraint, which is also exactly why strict=True (fully enabling that constraint) versus None/False (only partially engaging it) makes such a concrete, measurable difference in whether extra fields can slip through.