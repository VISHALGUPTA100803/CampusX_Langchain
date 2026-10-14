from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='Books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs = loader.load()

# print(docs[0])
# print(f"length {len(docs)} type {type(docs)}")

#Lazy Load

docs = loader.lazy_load()

print(type(docs)) # <class 'generator'>

# for document in docs:
#     print(document.metadata) 

#print(type(document))