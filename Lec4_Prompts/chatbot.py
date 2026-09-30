from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07', reasoning_effort='minimal')

chat_history = [
    SystemMessage(content="You are Helpful AI assistant")
]


while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if(user_input=="exit"):
        break 

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI :",result.content) 

print(chat_history)
