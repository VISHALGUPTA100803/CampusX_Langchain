from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description="Name of the person")
    age: int = Field(gt=18,description="Age of the person'")
    city: str = Field(description="Name of the city the person belongs to") 

parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.invoke({'place': "sri lankan"})
# print(prompt.to_string())
# result = model.invoke(prompt)
# print(result)
# final_result = parser.parse(result.content)
# print(final_result)
# print(final_result.age)

chain = template | model | parser

result = chain.invoke({'place': "sri lankan"})

print(result)


# INTERNAL PROMPT
# Generate the name, age and city of a fictional sri lankan person 
#  The output should be formatted as a JSON instance that conforms to the JSON schema below.

# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

# Here is the output schema:
# ```
# {"properties": {"name": {"description": "Name of the person", "title": "Name", "type": "string"}, "age": {"description": "Age of the person'", "title": "Age", "type": "integer"}, "city": {"description": "Name of the city the person belongs to", "title": "City", "type": "string"}}, "required": ["name", "age", "city"]}
# ```