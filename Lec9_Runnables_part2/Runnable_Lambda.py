from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

from dotenv import load_dotenv

load_dotenv()

def word_length(text):
    return len(text.split())

model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

prompt1 = PromptTemplate(
    template="generate only one joke on a {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1,model,parser)

# parallel_chain =RunnableParallel ({
#     "joke": RunnablePassthrough(),
#     "word_length": RunnableLambda(word_length)
# })

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_length": RunnableLambda(lambda x : len(x.split()))
})


final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic': 'AI'}))