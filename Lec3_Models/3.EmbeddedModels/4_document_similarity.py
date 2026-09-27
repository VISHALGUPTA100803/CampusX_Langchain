import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'  


from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from dotenv import load_dotenv
load_dotenv()



embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0] # send query embedding as 2d list 

index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]


print(index,score)

print(query)
print(documents[index])
print("similarity score is:", score) 

# Dry run of the line
# python
# index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]

# Let's trace it step by step using your actual scores array. Say (simplified from your output):

# python
# scores = [0.0577, 0.1841, 0.1705, 0.1474, 0.6184]

# (index 0 through 4, corresponding to Kohli, Dhoni, Tendulkar, Rohit, Bumrah)

# Step 1: enumerate(scores)

# enumerate() walks through scores and pairs each value with its position (index), starting from 0. This produces an enumerate object (a lazy iterator), not yet a visible list:

# (0, 0.0577)
# (1, 0.1841)
# (2, 0.1705)
# (3, 0.1474)
# (4, 0.6184)

# Step 2: list(...)

# Forces that lazy iterator into an actual, concrete list you can sort:

# python
# [(0, 0.0577), (1, 0.1841), (2, 0.1705), (3, 0.1474), (4, 0.6184)]

# Step 3: sorted(..., key=lambda x: x[1])

# sorted() reorders this list. The key tells it what to sort by, lambda x: x[1] means "for each tuple x, use its second element (x[1], the score) as the sort criterion", ignore the index (x[0]) when deciding order.

# By default sorted() goes ascending (smallest first). So it reorders by score, low to high:

# python
# [(0, 0.0577), (3, 0.1474), (2, 0.1705), (1, 0.1841), (4, 0.6184)]

# Notice the tuples themselves stay intact (index still paired with its own score), only their position in the list changes.

# Step 4: [-1]

# Grabs the last element of that sorted list, since it's sorted ascending, the last element has the highest score:

# python
# (4, 0.6184)

# This is exactly the "find the max" trick, sort ascending, take the last one.

# Step 5: index, score = (4, 0.6184)

# Python unpacks the 2-element tuple directly into two variables:

# python
# index = 4
# score = 0.6184

# So index tells you which document won (index 4 → the Bumrah sentence), and score tells you how similar it was to the query.

# Why score prints without the np.float64(...) wrapper

# This is a genuinely good catch, and it comes down to repr() vs str() again, the exact distinction from your earlier question about str(result).

# When you printed the whole sorted list, Python's list-printing calls repr() on each element inside it to build the display string. And repr() of a NumPy scalar is designed to be unambiguous/reconstructible, so it shows the type explicitly:

# python
# repr(np.float64(0.6183838197088036))
# # 'np.float64(0.6183838197088036)'

# That's why inside the list, you see np.float64(0.618...) for every entry, the list's own repr recursively calls repr on its contents.

# But once you print(score) on the bare, unpacked scalar (not inside a list/tuple anymore), print() calls str() on it instead, and str() on a NumPy scalar is designed for clean, human-readable display:

# python
# str(np.float64(0.6183838197088036))
# # '0.6183838197088036'

# No wrapper, just the number.

# Side-by-side to make it concrete:

# python
# x = np.float64(0.6184)

# print(x)          # 0.6184           (calls str())
# print(repr(x))    # np.float64(0.6184)   (explicit repr)
# print([x])        # [np.float64(0.6184)]  (list repr calls repr() on x internally)
# print(x, [x])     # 0.6184 [np.float64(0.6184)]  (same value, different display depending on context)

# The underlying rule, same one from your earlier str(result) question: it's still the exact same np.float64 object and the 
# exact same numeric value the whole time. What changes is only which display method Python invokes, str() for a directly-printed 
# standalone value, repr() when that value is shown as part of a larger structure like a list or tuple. The number itself never 
# changed, only how it chose to present itself to you. 


# i am still not getting the reper thing

# No problem, let's slow this way down with something even simpler than NumPy, using plain strings, since that removes the extra confusion of "what even is a float64."

# First, forget NumPy entirely. Just a plain string.
# python
# x = "hello"

# print(x)        # hello
# print([x])      # ['hello']
# Notice: printing x alone shows hello, no quotes. Printing [x] (the same string, but inside a list) shows 'hello', with quotes.

# Why the quotes appear only in the second case: every Python object has two different ways it can turn itself into text:

# str() → "the nice, readable version for humans"
# repr() → "the version that looks like how you'd type it in code"
# For a string, str("hello") gives hello (just the raw text), but repr("hello") gives 'hello' (with quotes, because that's literally how you'd write that string as Python code).

# Here's the actual rule that explains both lines above:

# print(x) calls str(x) on whatever you give it directly → hello
# When Python needs to display a list, it can't just dump raw text for each item (that would be ambiguous, is that item a string? a number? unclear without some marker). So Python's list-printing always calls repr() on each item inside it → 'hello', quotes included, so you can visually tell "this is a string" at a glance.
# Now apply that exact same rule to your NumPy score
# python
# x = np.float64(0.618)

# print(x)        # 0.618              ← str(x) was used
# print([x])      # [np.float64(0.618)]  ← repr(x) was used
# Same rule as the string example, just with a different object type:

# str(np.float64(0.618)) = "0.618" (clean, no type label, since a human already knows it's a number just by looking)
# repr(np.float64(0.618)) = "np.float64(0.618)" (unambiguous, code-like, explicitly says "this is specifically a NumPy float64, not a plain Python float")
# Connecting it back to your actual line of code
# python
# scores_list = [(0, np.float64(0.0577)), (4, np.float64(0.618))]   # a list of tuples

# print(scores_list)
# # [(0, np.float64(0.0577)), (4, np.float64(0.618))]   ← printing the WHOLE LIST → repr() used on contents

# index, score = scores_list[-1]   # unpacks the tuple → score is now standing alone

# print(score)
# # 0.618   ← printing score ALONE, not inside any list/tuple → str() used
# The one-sentence version: it's not that the number changes, it's that print() always uses str() on whatever you hand it directly, but the moment that same value is sitting inside a list or tuple that you're printing, Python instead uses repr() for everything inside that container, so it can show you unambiguously what type each item is. Same value, same object, two different display rules depending on whether it's alone or inside a container.



