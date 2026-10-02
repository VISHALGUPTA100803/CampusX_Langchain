from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.messages import messages_from_dict
import json

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])


with open("chat_history.txt") as f:
    raw_history = json.load(f)  

chat_history = messages_from_dict(raw_history)
print(chat_history)

prompt = chat_template.invoke({
    "chat_history": chat_history,
                               "query": "where is my refund ?"
                               })

print(prompt)

# A tuple is Python's immutable, ordered collection, think of it as a list's stricter sibling.

# Basic syntax
# python
# t = (1, 2, 3)
# t2 = ("Delhi", "Mumbai", "Chennai")
# t3 = 1, 2, 3          # parentheses are optional, commas are what matter
# single = (5,)         # a single-element tuple NEEDS a trailing comma
# not_a_tuple = (5)     # this is just the int 5, not a tuple!

# That trailing-comma rule trips people up constantly, (5) is just 5 in parentheses (normal math-style grouping), while (5,) is a one-item tuple. The comma, not the parentheses, is what actually makes it a tuple.

# The defining feature: immutability
# python
# t = (1, 2, 3)
# t[0] = 99        # TypeError: 'tuple' object does not support item assignment

# Once created, you cannot change, add, or remove elements. Compare to a list, where this works fine:

# python
# lst = [1, 2, 3]
# lst[0] = 99      # works: [99, 2, 3]
# Why use a tuple instead of a list, if it's "less capable"?

# 1. Semantic signal: "this data shouldn't change"
# If you're returning coordinates (x, y) or a date (year, month, day), using a tuple communicates to anyone reading your code "these values are a fixed, complete unit", not a growable collection.

# 2. You've actually been using this exact pattern all through this conversation, without necessarily naming it:

# python
# index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# enumerate(scores) produces (index, value) tuples, and index, score = (4, 0.618) is tuple unpacking, assigning each element to its own variable in one line. This is one of the most common real-world uses of tuples.

# 3. Dictionary keys: lists can't be dict keys (they're mutable, so their hash would be unstable), but tuples can:

# python
# locations = {
#     (28.6, 77.2): "Delhi",
#     (19.0, 72.8): "Mumbai",
# }

# 4. Slightly faster and more memory-efficient than lists, since Python can optimize for the fact that they never change.

# Things you CAN do with a tuple (read-only operations work fine)
# python
# t = (10, 20, 30, 40)

# print(t[0])        # 10 — indexing works
# print(t[1:3])      # (20, 30) — slicing works
# print(len(t))      # 4
# print(30 in t)     # True
# for x in t:        # iteration works
#     print(x)

# a, b, c, d = t     # unpacking works


# with open("chat_history.txt") as f:
#     raw_history = json.load(f)  

# chat_history = messages_from_dict(raw_history)
# print(chat_history)

# Previously you did:
# messages_to_dict() → json.dump()


# to save LangChain messages.
# Now you're doing:
# json.load() → messages_from_dict()


# to load them back.
# The complete flow is:
# chat_history.txt
#        ↓
#    json.load()
#        ↓
# Python list of dictionaries
#        ↓
# messages_from_dict()
#        ↓
# LangChain Message objects

# Let's go line by line.

# 2. What is currently inside the file?
# From your previous program, you did:
# json.dump(messages_to_dict(history), f)


# So chat_history.txt contains JSON representing your messages.
# Simplifying the contents, imagine it contains:
# [
#     {
#         "type": "human",
#         "data": {
#             "content": "I want to request a refund for my order #12345."
#         }
#     },
#     {
#         "type": "ai",
#         "data": {
#             "content": "Your refund request for order #12345 has been initiated."
#         }
#     }
# ]

# Important distinction:
# At this moment, this is text sitting on your hard drive.
# It is not yet a Python list.
# It is not yet a HumanMessage.
# It is not yet an AIMessage.
# It's simply JSON text stored in a file.
# 3. json.load(f)
# Now Python executes:
# raw_history = json.load(f)


# This is very important.
# json.load() reads JSON from an opened file and converts it into normal Python objects.
# So:
# JSON file
#     ↓
# json.load()
#     ↓
# Python objects

# For example, JSON:
# {
#     "name": "Vishal",
#     "age": 25
# }

# would become a Python dictionary:
# {    "name": "Vishal",    "age": 25}


# In your case, the JSON array becomes a Python list.
# 4. Dry run of json.load(f)
# Before this line:
# raw_history = json.load(f)


# you have:
# chat_history.txt

# [
#    {"type": "human", ...},
#    {"type": "ai", ...}
# ]

# Python reads it:
# json.load(f)


# and creates something conceptually like:
# [    {        "type": "human",        "data": {            "content": "I want to request a refund for my order #12345."        }    },    {        "type": "ai",        "data": {            "content": "Your refund request for order #12345 has been initiated."        }    }]


# This Python list is assigned to:
# raw_history


# So now:
# raw_history
#      │
#      ▼
# Python list
#      │
#      ├── dictionary
#      │      ├── "type": "human"
#      │      └── "data": {...}
#      │
#      └── dictionary
#             ├── "type": "ai"
#             └── "data": {...}

# 5. What type is raw_history?
# If you wrote:
# print(type(raw_history))


# you'd get:
# <class 'list'>

# And:
# print(type(raw_history[0]))


# would give:
# <class 'dict'>

# So this is extremely important:
# raw_history


# is currently:
# list of dictionaries

# It is not yet:
# list of LangChain messages

# 6. Why can't we directly use raw_history as chat history?
# Because LangChain generally works with message objects such as:
# HumanMessage(...)AIMessage(...)SystemMessage(...)


# But currently you have ordinary dictionaries:
# {    "type": "human",    "data": {...}}


# So we need to reconstruct the LangChain objects.
# That's where this comes in:
# messages_from_dict(raw_history)


# 7. messages_from_dict(raw_history)
# Your next line is:
# chat_history = messages_from_dict(raw_history)


# This is essentially the opposite of:
# messages_to_dict()


# Previously:
# HumanMessage
# AIMessage
#      ↓
# messages_to_dict()
#      ↓
# dictionaries

# Now:
# dictionaries
#      ↓
# messages_from_dict()
#      ↓
# HumanMessage
# AIMessage

# 8. Dry run of messages_from_dict()
# Before:
# raw_history


# conceptually contains:
# [    {        "type": "human",        "data": {            "content": "I want to request a refund..."        }    },    {        "type": "ai",        "data": {            "content": "Your refund request..."        }    }]


# Now:
# messages_from_dict(raw_history)


# looks at the first dictionary:
# {    "type": "human",    ...}


# LangChain sees:
# type = "human"

# so it reconstructs a:
# HumanMessage(...)


# Then it sees:
# {    "type": "ai",    ...}


# and reconstructs:
# AIMessage(...)


# So the transformation is conceptually:
# raw_history

# [
#     {
#         "type": "human",
#         "data": {
#             "content": "I want a refund..."
#         }
#     },

#     {
#         "type": "ai",
#         "data": {
#             "content": "Your refund..."
#         }
#     }
# ]

#              ↓

#     messages_from_dict()

#              ↓

# [
#     HumanMessage(
#         content="I want a refund..."
#     ),

#     AIMessage(
#         content="Your refund..."
#     )
# ]

# That resulting list is assigned to:
# chat_history