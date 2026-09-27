from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import os
from transformers import GenerationConfig
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_HOME'] = 'D:/huggingface_cache'

gen_config = GenerationConfig(temperature=0.5, do_sample=True)
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    # pipeline_kwargs=dict(
    #     temperature = 0.5
    # )
    pipeline_kwargs=dict(generation_config=gen_config)

)

model = ChatHuggingFace(llm=llm)

result = model.invoke("tell about planet earth")

print(result)