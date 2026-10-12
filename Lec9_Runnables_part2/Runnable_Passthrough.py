from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

prompt1 = PromptTemplate(
    template="generate joke on a {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="give explanation of the following joke - {text}",
    input_variables=["text"]
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel( {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(prompt2,model,parser)
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic': 'AI'}))