from langchain_core.prompts import ChatPromptTemplate 

chat_template = ChatPromptTemplate([ # two tuples inside a list
    ("system", "you are a helpful {domain} expert"),
    ("human", "tell about planet {planet_name}")
])

prompt = chat_template.invoke({
    "domain": "astrophysicist",
    "planet_name": "jupiter"
})

print(prompt)