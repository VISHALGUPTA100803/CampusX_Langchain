import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

sentence = "Chunking unfamiliar vocabulary requires splitting rare words into multiple subword tokens."
print(len(enc.encode(sentence)))                    # whole sentence count

pieces = ["Chunking", " unfamiliar", " vocabulary", " requires", " splitting",
          " rare", " words", " into", " multiple", " subword", " tokens."]
print([len(enc.encode(p)) for p in pieces])         # per piece counts
print(sum(len(enc.encode(p)) for p in pieces))      # sum of parts