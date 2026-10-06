from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model1 = ChatOpenAI(model='gpt-5-nano-2025-08-07')

model2 = ChatAnthropic(model="claude-haiku-4-5-20251001")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"]
)

chain = prompt1 | model1 | parser | prompt2 | model2 | parser 

# result = chain.invoke({"topic": "black hole"})

# print(result)

chain.get_graph().print_ascii()