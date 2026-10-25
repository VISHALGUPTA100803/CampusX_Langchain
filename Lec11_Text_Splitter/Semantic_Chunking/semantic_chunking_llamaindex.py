# Install: python -m pip install llama-index-core llama-index-embeddings-openai python-dotenv

from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

splitter = SemanticSplitterNodeParser(
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    buffer_size=1,
    breakpoint_percentile_threshold=95,
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

nodes = splitter.get_nodes_from_documents([Document(text=sample)])
docs = [node.get_content() for node in nodes]

print(len(docs))
print(docs)
