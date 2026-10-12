from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableMap
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano-2025-08-07")

prompt1 = PromptTemplate(
    template="Generate a joke on {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke: {text} in {length} lines",
    input_variables=["text", "length"]
)

length = 3  # Set the number of lines manually

map_inputs = RunnableMap({
    "text": lambda output: output,
    "length": lambda output: length
})

chain = RunnableSequence(
    prompt1,
    model,
    parser,
    map_inputs,
    prompt2,
    model,
    parser
)

print(chain.invoke({"topic": "AI"}))


# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence, RunnableMap
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI(model="gpt-5-nano-2025-08-07")

# prompt1 = PromptTemplate(
#     template=(
#         "Generate a joke about {topic}. "
#         "Return only a JSON object with these keys: "
#         "'generated_joke': the joke text, "
#         "'explanation_lines': the number 3."
#     ),
#     input_variables=["topic"]
# )

# # Converts the JSON response into a Python dictionary
# parser1 = JsonOutputParser()

# map_inputs = RunnableMap({
#     "text": lambda data: data["generated_joke"],
#     "length": lambda data: data["explanation_lines"]
# })

# prompt2 = PromptTemplate(
#     template="Explain the following joke: {text} in {length} lines.",
#     input_variables=["text", "length"]
# )

# parser2 = StrOutputParser()

# chain = RunnableSequence(
#     prompt1,
#     model,
#     parser1,
#     map_inputs,
#     prompt2,
#     model,
#     parser2
# )

# print(chain.invoke({"topic": "AI"}))