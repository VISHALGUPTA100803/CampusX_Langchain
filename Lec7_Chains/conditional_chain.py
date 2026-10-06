from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

parser1 = StrOutputParser()


class Feedback(BaseModel):

    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)  

prompt1 = PromptTemplate(
    template= "Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write one short, ready-to-send reply to this customer's positive feedback in 1-2 sentences. Output only the message—no options, tips, headings, placeholders, or questions. \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write one short, ready-to-send reply to this customer's negative feedback in 1-2 sentences. Output only the message—no options, tips, headings, placeholders, or questions. \n {feedback}",
    input_variables=["feedback"]
)

conditional_chain = RunnableBranch(
    (lambda x : x.sentiment== "positive", prompt2 | model | parser1), # (condition, Runnable)
    (lambda x: x.sentiment == "negative", prompt3 | model | parser1),
    RunnableLambda(lambda x : "could not find sentiment")
)

chain = classifier_chain | conditional_chain 

result = chain.invoke({"feedback": "this smartphone works well"})

print(result)

print("hello")


# chain.get_graph().print_ascii()
# classifier_chain.get_graph().print_ascii()
# conditional_chain.get_graph().print_ascii()

# result = classifier_chain.invoke({"feedback": "this is amazing phone"})
# print(prompt1)
# print(result)
# print(result.sentiment)
# classifier_chain.get_graph().print_ascii()
