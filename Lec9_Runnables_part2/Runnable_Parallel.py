from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

prompt1 = PromptTemplate(
    template="generate a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="generate a linkedin post on {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1,model,parser),
    "linkedin": RunnableSequence(prompt2,model,parser)
}
)

result = parallel_chain.invoke({"topic": "AI"})

print(result)
print(result['tweet'])
print(result['linkedin'])