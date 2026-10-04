from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

json_schema = {
     "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"], 
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

# The square brackets specify the type and metadata; they don’t make Annotated a list.

structured_model = model.with_structured_output(json_schema, strict=True)

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

result = structured_model.invoke(prompt) # Pydantic object

print(result["cons"])

print(result)

# Literal become enum in json 

# # JSON schema passed → dictionary
# result["cons"]

# # Pydantic class passed → Review object
# result.cons

# Updated comparison table, adding this third option
# Stage	Pydantic	TypedDict	Raw JSON Schema
# Schema generation	Pydantic's native generator	LangChain manually builds it from annotations	You write it yourself, no generation step
# strict default	None (same as others, but schema itself is strict-compatible)	None/off by default per docstring	None/off by default per docstring, same as TypedDict
# Post-response validation	Real: model_validate(), can raise ValidationError	None	None
# Return type	Review instance	dict	dict
# Authoring experience	Python classes, IDE autocomplete, type checking	Python classes, lighter weight	Raw dict literal, no type checking, no IDE help, full manual control over JSON Schema features