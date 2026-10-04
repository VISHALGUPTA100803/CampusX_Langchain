from typing import  Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07', reasoning_effort='minimal')

class Review(BaseModel):

    keyTheme: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")  
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review as pos for positive or neg for negative") 
    pros: Optional[list[str]] = Field(default = None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default = None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default = None, description="Write the name of the reviewer")

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

result = structured_model.invoke(prompt) # Pydantic object

#print(result.cons)

print(result)


# Why: Optional[X] and "has a default" are two separate things in Pydantic

# This is a common point of confusion. Optional[list[str]] only means the value, if provided, can be None, it does NOT automatically mean "this field is allowed to be missing entirely." Without a default, Pydantic still treats it as a required field, just one whose allowed value type happens to include None.

# python
# from pydantic import BaseModel, Field
# from typing import Optional

# class Review(BaseModel):
#     pros: Optional[list[str]] = Field(description="...")  # no default

# Review(pros=["great camera"])   # ✅ works
# Review(pros=None)               # ✅ works, None is a valid value for Optional
# Review()                        # ❌ ValidationError: pros is required!

# That last case is the surprising one: even though the type is Optional, Pydantic still demands something be passed, None is an acceptable value, but omitting the field entirely is not acceptable without a default.

# With default=None, it genuinely becomes optional (can be omitted)
# python
# class Review(BaseModel):
#     pros: Optional[list[str]] = Field(default=None, description="...")

# Review(pros=["great camera"])   # ✅ works
# Review(pros=None)               # ✅ works
# Review()                        # ✅ now works too, pros defaults to None