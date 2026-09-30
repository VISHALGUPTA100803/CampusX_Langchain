from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.load import load
import json
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07', reasoning_effort='minimal')


st.header('Reasearch Tool')


paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

with open("template.json", encoding="utf-8") as file:
    template = load(json.load(file), allowed_objects=[PromptTemplate])
 



#print(prompt)
#print(prompt.to_messages())
if st.button('Summarize'):

    chain = template | model

    result = chain.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})


    print(result)
    st.write(result.content)


# from langchain_core.load import load
# template = load(json.load(file), allowed_objects=[PromptTemplate])
# First line: from langchain_core.load import load

# from: tells Python where to find something.
# langchain_core.load: the module inside the langchain_core package that provides serialization/loading utilities.
# import: tells Python which name to bring into this file.
# load: the function you’ll call below. It takes serialized LangChain data and reconstructs the corresponding LangChain object.
# After this line, you can call load(...) directly. In your installed LangChain version, it may display a beta warning because the API could change.

# Second line: template = load(json.load(file), allowed_objects=[PromptTemplate])

# template: the variable that will hold the reconstructed prompt object.
# =: assigns the result on the right to template.
# load(...): asks LangChain to reconstruct an object from serialized data.
# json.load(file): reads JSON text from the open file and converts it into Python data, usually a dictionary. Here, file must be the file object created by a surrounding with open(...) as file: block.
# allowed_objects=: a named argument that tells LangChain which object classes it is permitted to reconstruct from the serialized data.
# [PromptTemplate]: a Python list containing the PromptTemplate class. It allows LangChain to reconstruct this prompt type.
# So the whole line reads as: “Read the JSON from file, let LangChain reconstruct the serialized data as an allowed PromptTemplate, and store the result in template.”

# This assumes the JSON file was generated from the prompt’s LangChain serialization, such as template.to_json(). load() is intended for trusted serialized data, so only load files you trust.

# load(..., allowed_objects=[PromptTemplate]) rebuilds the prompt object.

# The dictionary contains serialized prompt data, including the prompt text, variable names, and format. load() reads that structure and reconstructs a PromptTemplate. The allowlist says PromptTemplate is an allowed type to rebuild from this data.

# After this line, template is no longer a dictionary; it is a PromptTemplate object.

# template.invoke({...}) fills in the prompt variables.

# JSON file
#   -> Python dictionary
#   -> PromptTemplate object
#   -> formatted prompt with the selected values
#   -> model response
#   -> displayed text 


# 3. What does | mean in LangChain?
# This is one of the most important things to understand.
# chain = template | model


# The | operator means:
# Take the output of the thing on the left and pass it as input to the thing on the right.

# So:
# template | model


# means:
# INPUT
#   ↓
# template
#   ↓
# model
#   ↓
# OUTPUT

# It's similar conceptually to a Unix pipe:
# command1 | command2

# where the output of command1 becomes the input of command2.
# In LangChain, this composition style is part of the LangChain Expression Language (LCEL).
# For example:
# chain = template | model


# can be mentally understood as:
# input   ↓template.invoke(input)   ↓formatted prompt   ↓model.invoke(formatted_prompt)   ↓AI response


# The actual implementation is handled by LangChain, but this is the right mental model.
# 4. What is chain then?
# After:
# chain = template | model


# chain represents the entire pipeline.
# It is no longer just a prompt and it is not just a model.
# Think of it as a function:
# chain(input) → output


# Except LangChain uses:
# chain.invoke(input)


# So you can think of:
# chain = template | model


# as roughly creating:
# def chain(inputs):    formatted_prompt = template.invoke(inputs)    response = model.invoke(formatted_prompt)    return response


# This isn't the literal LangChain source code, but conceptually it's very close.