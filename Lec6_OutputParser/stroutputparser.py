from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template = "Write a 5 line summary on the following text. \n {text}",
    input_variables=["text"]
)


chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'planet earth'})

print(result)

# template2 is invoked automatically when you invoke the whole chain:
# chain = template1 | model | parser | template2 | model | parser
# result = chain.invoke({'topic': 'planet earth'})
# The | connects the steps: each step’s output becomes the next step’s input.
# Your data flows like this:
# {'topic': 'planet earth'}
#         ↓ template1
# "Write a detailed report on planet earth"
#         ↓ model
# AIMessage containing the report
#         ↓ parser
# Report as a plain string
#         ↓ template2
# "Write a 5 line summary on the following text ... [report]"
#         ↓ model
# AIMessage containing the summary
#         ↓ parser
# Summary as a plain string
# But how does the report string get assigned to {text}?
# Because template2 has only one input variable, text, LangChain automatically assigns an incoming string to that variable. So these are equivalent:
# template2.invoke(report)
# template2.invoke({'text': report})
# This is LangChain’s single-variable prompt behavior.
# Your chain therefore behaves roughly like these explicit calls:
# prompt1 = template1.invoke({'topic': 'planet earth'})
# report = parser.invoke(model.invoke(prompt1))

# prompt2 = template2.invoke({'text': report})
# result = parser.invoke(model.invoke(prompt2))
# You supply topic; the first model generates the text that template2 receives.



# If template2 has multiple variables, you must pass it a dictionary containing all those variables.
# For example, suppose it needs text and language:
# template2 = PromptTemplate(
#     template="Write a 5 line summary in {language} of:\n{text}",
#     input_variables=["text", "language"]
# )
# Add a step that turns the report string into the required dictionary:
# chain = (
#     template1
#     | model
#     | parser
#     | (lambda report: {"text": report, "language": "Hindi"})
#     | template2
#     | model
#     | parser
# )

# result = chain.invoke({"topic": "planet earth"})
# The lambda receives the previous parser’s output—the report—and produces:
# {
#     "text": "The generated report about planet earth...",
#     "language": "Hindi"
# }
# template2 then uses those dictionary keys to fill {text} and {language}.