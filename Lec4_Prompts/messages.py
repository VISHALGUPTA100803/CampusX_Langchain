from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07', reasoning_effort='minimal')

messages = [
    SystemMessage(content="You are a Helpful Assistant"),
    HumanMessage(content="Tell about planet earth in 3 lines")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)

