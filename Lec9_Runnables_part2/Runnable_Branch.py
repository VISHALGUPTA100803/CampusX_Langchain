from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch

from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

prompt1 = PromptTemplate(
    template="write detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="summarize the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

# content_gen_chain = RunnableSequence(prompt1,model,parser)

# summary_chain = RunnableBranch(
#     (lambda x : len(x.split()) > 500, RunnableSequence(prompt2,model,parser)), # (condition, runnable) Tuple
#     RunnablePassthrough() # default
# )

content_gen_chain = prompt1 | model | parser

summary_chain = RunnableBranch(
    (lambda x : len(x.split()) > 500, prompt2 | model | parser), # (condition, runnable) Tuple
    RunnablePassthrough() # default
)

# def needs_summary(text):
#     return len(text.split()) > 500

# summary_chain = RunnableBranch(
#     (needs_summary, RunnableSequence(prompt2, model, parser)),
#     RunnablePassthrough()
# )

#final_chain = RunnableSequence(content_gen_chain, summary_chain)

final_chain = content_gen_chain | summary_chain

print(final_chain.invoke({'topic': 'planet earth'}))

# The tuple is (condition, action), but both positions can contain runnables.
# A runnable is simply something LangChain can execute. It doesn’t have to be a prompt or model chain.
# RunnableLambda(lambda x: len(x.split()) > 500)

# The default has no condition. It is the final argument and runs when none of the conditions match. These are different ways to write it:
# # 1. Return the original input unchanged
# summary_chain = RunnableBranch(
#     (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
#     RunnablePassthrough()
# )
# # 2. Same behavior using a plain lambda
# summary_chain = RunnableBranch(
#     (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
#     lambda x: x
# )
# # 3. Same behavior using an explicit RunnableLambda
# summary_chain = RunnableBranch(
#     (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
#     RunnableLambda(lambda x: x)
# )
# # 4. Return a custom message instead
# summary_chain = RunnableBranch(
#     (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
#     lambda x: "The report is already short enough."
# )
# For your case, RunnablePassthrough() clearly expresses “return the original report.”