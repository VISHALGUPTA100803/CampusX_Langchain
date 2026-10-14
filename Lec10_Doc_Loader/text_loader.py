from langchain_community.document_loaders import TextLoader 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

parser = StrOutputParser()

loader = TextLoader("cricket.txt", encoding='utf-8')

docs = loader.load()

template = PromptTemplate(
    template="Write a summary for the following poem - \n {poem}",
    input_variables=["poem"]
)

chain = template | model | parser 

result =  chain.invoke({"poem": docs[0].page_content})

print(result)

# print(type(loader)) #<class 'langchain_community.document_loaders.text.TextLoader'>
# print("\n ######")
# print(type(docs)) #<class 'list'>
# print("\n ######")
# print(loader)  #<langchain_community.document_loaders.text.TextLoader object at 0x00000241C8727770>
# print("\n ######")
# print(docs) 
# print("\n ######")
# print(len(docs))
# print("\n ######")
# print(docs[0])
# print("\n ######")
# print(docs[0].page_content)
# print("\n ######")
# print(docs[0].metadata )

# print(type(docs[0])) # <class 'langchain_core.documents.base.Document'>
# print(type(docs[0].page_content)) # <class 'str'>
# print(type(docs[0].metadata)) # <class 'dict'>

