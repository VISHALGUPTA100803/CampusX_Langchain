from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

# prompt1 = PromptTemplate(
#     template="generate joke on a {topic}",
#     input_variables=["topic"]
# )

parser = StrOutputParser()

# prompt2 = PromptTemplate(
#     template="give explanation of the following joke - {text} in {length} lines",
#     input_variables=["text","length"]
# )

# length = 3

# prepare_prompt2 = RunnableLambda(
#     lambda x : {"text": x, "length": length}
# )

# chain = RunnableSequence(prompt1, model, parser, prepare_prompt2, prompt2, model,parser)

# print(chain.invoke({"topic": "AI"}))

prompt1 = PromptTemplate(
    template=(
        "Generate a joke about {topic}. "
        "Return only a JSON object with two keys: "
        "'generated_joke' containing the joke, and "
        "'explanation_lines' containing the number 3."
    ),
    input_variables=["topic"]
)

# Converts the model's JSON response into a Python dictionary.
parser1 = JsonOutputParser()

prompt2 = PromptTemplate(
    template="Explain this joke: {text} in {length} lines.",
    input_variables=["text", "length"]
)

map_keys = RunnableLambda(
    lambda previous_output : {
        "text": previous_output["generated_joke"],
        "length": previous_output["explanation_lines"] 
    }
)

chain = RunnableSequence(
    prompt1,
    model,
    parser1,
    map_keys,
    prompt2,
    model,
    parser
)

print(chain.invoke({"topic": "computers"}))