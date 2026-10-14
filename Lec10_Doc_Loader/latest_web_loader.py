import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Use the same product URL from your existing file
url = (
    "https://www.flipkart.com/"
    "zebronics-zeb-thunder-true-60hr-backup-bt-v6-0-gaming-mode-enc-aux-msd-dual-pairing-bluetooth/"
    "p/itm241139e524726?pid=ACCFBHDEHZFK6M2Y"
)

response = requests.get(
    url,
    headers={"User-Agent": "MyLangChainLearningApp/1.0"},
    timeout=30
)
response.raise_for_status()

soup = BeautifulSoup(response.content, "html.parser")

for tag in soup(["script", "style"]):
    tag.decompose()

docs = [
    Document(
        page_content=soup.get_text(separator="\n", strip=True),
        metadata={"source": url}
    )
]

model = ChatOpenAI(model="gpt-5-nano-2025-08-07")

parser = StrOutputParser()

prompt = PromptTemplate(
    template=(
        "Answer the following question:\n{question}\n"
        "Using only the following text:\n{text}\n"
        "If the answer is missing, say it was not found."
    ),
    input_variables=["question", "text"]
)

chain = prompt | model | parser

result = chain.invoke({
    "question": "What is the product price?",
    "text": docs[0].page_content
})

print(result)


# 1. Parse the HTML
# soup = BeautifulSoup(response.content, "html.parser")
# - response.content contains the webpage’s downloaded HTML as bytes.
# - BeautifulSoup(...) parses that HTML into an object whose tags you can search and modify.
# - "html.parser" selects Python’s built-in HTML parser. 

# 2. Remove scripts and styles
# for tag in soup(["script", "style"]):
#     tag.decompose()
# soup(["script", "style"]) finds all <script> and <style> tags.
# The loop calls decompose() on each matching tag, removing the tag and its contents from the parsed HTML.
# We remove them because JavaScript code and CSS styling aren’t useful text for answering questions about the product.

# 3. Extract text and create a document
# docs = [
#     Document(
#         page_content=soup.get_text(separator="\n", strip=True),
#         metadata={"source": url}
#     )
# ]
# page_content
# soup.get_text(separator="\n", strip=True)
# This extracts text without HTML tags:
# - separator="\n" puts a newline between extracted text pieces.
# - strip=True removes leading and trailing whitespace from each piece.

# metadata
# metadata={"source": url}
# This stores the webpage URL alongside the text so you know where the document came from.
# Why the square brackets?
# docs = [Document(...)]
# The brackets create a list containing one Document, matching the structure you used with loaders: