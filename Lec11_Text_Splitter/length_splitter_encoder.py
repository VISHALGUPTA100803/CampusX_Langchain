from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load() 

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)

texts = text_splitter.split_documents(docs)

print(texts[0].page_content)