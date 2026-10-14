from pathlib import Path
from langchain_core.documents import Document

file_path = "cricket.txt"

docs = [
    Document(
        page_content=Path(file_path).read_text(encoding="utf-8"),
        metadata={"source": file_path}
    )
]


print(type(docs)) # <class 'list'>
print("\n ######")
 
print("\n ######")
print(docs) 
print("\n ######")
print(len(docs))
print("\n ######")
print(docs[0])
print("\n ######")
print(docs[0].page_content)
print("\n ######")
print(docs[0].metadata )

print(type(docs[0]))
print(type(docs[0].page_content))
print(type(docs[0].metadata))