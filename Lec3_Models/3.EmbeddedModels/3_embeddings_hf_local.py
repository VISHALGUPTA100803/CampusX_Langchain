import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'  


from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()



embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Delhi is the capital of India"

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_query(text)
vector2 = embedding.embed_documents(documents)

print(vector)
print(vector2)