Yes, exactly. For LangChain message objects, the basic pattern is:
# Save to DBdata = messages_to_dict(chat_history)# store data in DB


Then when fetching:
# Fetch from DBdata = db_resultchat_history = messages_from_dict(data)


So:
SAVE:
HumanMessage/AIMessage
→ messages_to_dict()
→ DB

LOAD:
DB
→ messages_from_dict()
→ HumanMessage/AIMessage

At scale, the idea is the same, but you'd usually store the serialized messages in a database (often JSON/JSONB or structured columns) rather than a .txt file.
I'll keep the next explanations shorter.