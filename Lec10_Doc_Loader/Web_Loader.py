import os

os.environ["USER_AGENT"] = "MyLangChainLearningApp/1.0"

from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.flipkart.com/zebronics-zeb-thunder-true-60hr-backup-bt-v6-0-gaming-mode-enc-aux-msd-dual-pairing-bluetooth/p/itm241139e524726?pid=ACCFBHDEHZFK6M2Y&lid=LSTACCFBHDEHZFK6M2YEW7GQQ&hl_lid=&marketplace=FLIPKART&fm=eyJ3dHAiOiJyZWNvIiwicHJwdCI6InBwIiwibWlkIjoicHJvZHVjdFJlY29tbWVuZGF0aW9uL3NpbWlsYXIifQ%3D%3D&pageUID=1792005513913'

loader = WebBaseLoader(url)

docs = loader.load()


from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

parser = StrOutputParser()

prompt = PromptTemplate(
    template="answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"]
)

chain = prompt | model | parser 

result = chain.invoke({"question": "what is the product price ?", "text": docs[0].page_content})

# print(len(docs))
# print(docs)

print(result)

