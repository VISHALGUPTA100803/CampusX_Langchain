import re
from math import sqrt
from statistics import mean, pstdev

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


# Custom semantic splitter; does not require langchain_experimental.
class SemanticChunker:
    def __init__(self, embeddings, breakpoint_threshold_amount=1, buffer_size=1):
        self.embeddings = embeddings
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.buffer_size = buffer_size

    @staticmethod
    def cosine_distance(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sqrt(sum(x * x for x in a))
        norm_b = sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            raise ValueError("Cannot compare zero-length embedding vectors.")

        return 1 - max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    def split_text(self, text):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]

        if len(sentences) <= 1:
            return sentences

        # Embed each sentence together with its neighboring sentences.
        windows = [
            " ".join(
                sentences[
                    max(0, i - self.buffer_size):
                    min(len(sentences), i + self.buffer_size + 1)
                ]
            )
            for i in range(len(sentences))
        ]

        # API reference:
        # https://reference.langchain.com/python/langchain-openai/embeddings/base/OpenAIEmbeddings/embed_documents
        vectors = self.embeddings.embed_documents(windows)

        distances = [
            self.cosine_distance(vectors[i], vectors[i + 1])
            for i in range(len(vectors) - 1)
        ]

        # Standard-deviation breakpoint threshold.
        threshold = (
            mean(distances)
            + self.breakpoint_threshold_amount * pstdev(distances)
        )

        chunks = []
        start = 0

        for i, distance in enumerate(distances):
            if distance > threshold:
                chunks.append(" ".join(sentences[start:i + 1]))
                start = i + 1

        chunks.append(" ".join(sentences[start:]))
        return chunks


splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_amount=1,
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = splitter.split_text(sample)

print(len(docs))
print(docs)