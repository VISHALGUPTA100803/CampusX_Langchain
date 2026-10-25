from langchain_text_splitters import CharacterTextSplitter 

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load() 

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    separator=''
)

text = """
  Y Combinator encourages founders to focus on growth rate rather than absolute numbers. It prevents them from being discouraged early on, when the absolute numbers are still low. It also helps them decide what to focus on: you can use growth rate as a compass to tell you how to evolve the company. But the main advantage is that by focusing on growth rate you tend to get something that grows exponentially.

YC doesn't explicitly tell founders that with growth rate "you get out what you put in," but it's not far from the truth. And if growth rate were proportional to performance, then the reward for performance p over time t would be proportional to pt.

Even after decades of thinking about this, I find that sentence startling.
"""

result1 = splitter.split_text(text)
result = splitter.split_documents(docs)

print(type(result1))

print(result1)
print(type(result))

print(result[0])