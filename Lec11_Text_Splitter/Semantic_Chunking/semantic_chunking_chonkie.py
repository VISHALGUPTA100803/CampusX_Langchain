from chonkie import SemanticChunker
from chonkie.refinery import OverlapRefinery
from langchain_core.documents import Document

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""
text = sample.strip()

# ---------- settings ----------
EMBEDDING_MODEL = "minishlab/potion-base-32M"   # local model, no API key needed
THRESHOLD = 0.7        # lower = bigger chunks, higher = smaller chunks
CHUNK_SIZE = 512       # maximum tokens per chunk
MIN_SENTENCES = 2      # avoid tiny chunks
SKIP_WINDOW = 0        # set to 1 to merge similar non consecutive groups
OVERLAP = 50           # overlap size added after chunking

# optional: use OpenAI embeddings instead (needs OPENAI_API_KEY)
# from chonkie.embeddings import OpenAIEmbeddings
# EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")


def build_chunker(threshold):
    return SemanticChunker(
        embedding_model=EMBEDDING_MODEL,
        threshold=threshold,
        chunk_size=CHUNK_SIZE,
        similarity_window=3,
        min_sentences_per_chunk=MIN_SENTENCES,
        skip_window=SKIP_WINDOW,
    )


# ---------- 1. tune the threshold ----------
for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    result = build_chunker(t).chunk(text)
    avg = sum(c.token_count for c in result) / len(result)
    print(f"threshold {t}: {len(result)} chunks, average {avg:.0f} tokens")

# ---------- 2. final chunking with the chosen threshold ----------
chunker = build_chunker(THRESHOLD)
chunks = chunker.chunk(text)
chunks = OverlapRefinery(
    tokenizer=chunker.tokenizer,
    context_size=OVERLAP,
).refine(chunks)

# ---------- 3. convert to LangChain documents ----------
lc_docs = [
    Document(
        page_content=c.text,
        metadata={
            "start_index": c.start_index,
            "end_index": c.end_index,
            "token_count": c.token_count,
        },
    )
    for c in chunks
]

# ---------- 4. inspect ----------
print(f"\nFinal chunks: {len(lc_docs)}")
for i, d in enumerate(lc_docs, 1):
    print(f"\n--- chunk {i} ({d.metadata['token_count']} tokens) ---")
    print(d.page_content)
