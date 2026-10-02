# convert_history.py — run this ONCE to fix the existing file
from langchain_core.messages import HumanMessage, AIMessage, messages_to_dict
import json

history = [
    HumanMessage(content="I want to request a refund for my order #12345."),
    AIMessage(content="Your refund request for order #12345 has been initiated. It will be processed in 3-5 business days."),
]

with open("chat_history.txt", "w") as f:
    json.dump(messages_to_dict(history), f) 


# messages_to_dict
# This is different.
# It's a function that converts LangChain message objects into ordinary Python dictionaries.
# For example, conceptually:
# HumanMessage(content="Hello")


# becomes something similar to:
# {    "type": "human",    "data": {        "content": "Hello",        ...    }}


# Why do we need this conversion? Because json.dump() knows how to serialize normal Python types such as:
# dict
# list
# str
# int
# float
# bool
# None

# but it doesn't automatically know how to serialize arbitrary LangChain objects like HumanMessage.

# 7. The most important line
# Now:
# json.dump(messages_to_dict(history), f)


# Read this from the inside out.
# Python first evaluates:
# messages_to_dict(history)


# Then takes the result and does:
# json.dump(result, f)


# So mentally break it into:
# converted_history = messages_to_dict(history)json.dump(converted_history, f)


# Let's dry-run both steps.
# 8. messages_to_dict(history)
# Currently:
# history = [    HumanMessage(...),    AIMessage(...)]


# These are LangChain objects.
# messages_to_dict() converts them into normal dictionaries that preserve the message type and data.
# Conceptually, the transformation is:
# BEFORE

# [
#     HumanMessage(...),
#     AIMessage(...)
# ]

#         ↓

# messages_to_dict()

#         ↓

# AFTER

# [
#     {
#         "type": "human",
#         "data": {...}
#     },

#     {
#         "type": "ai",
#         "data": {...}
#     }
# ]

# A simplified representation would look like:
# [    {        "type": "human",        "data": {            "content": "I want to request a refund for my order #12345."        }    },    {        "type": "ai",        "data": {            "content": "Your refund request for order #12345 has been initiated. It will be processed in 3-5 business days."        }    }]


# The actual dictionaries can contain additional LangChain message fields depending on your installed LangChain version, but this is the important structure to understand.
# So now we have gone from:
# LangChain objects

# to:
# normal Python list + dictionaries + strings

# And those can be serialized to JSON.
# 9. Why can't we just do json.dump(history, f)?
# Suppose you tried:
# json.dump(history, f)


# Python would see:
# [    HumanMessage(...),    AIMessage(...)]


# and eventually encounter the HumanMessage object.
# The standard JSON encoder doesn't inherently know:
# How should I represent this custom LangChain HumanMessage object in JSON?

# So you can get an error similar to:
# TypeError: Object of type HumanMessage is not JSON serializable

# That's why this conversion is important:
# messages_to_dict(history)


# It turns:
# HumanMessage object
# AIMessage object

# into JSON-friendly Python dictionaries.