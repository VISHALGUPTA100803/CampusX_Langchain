from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("tell unique things about planet earth")

print(result)

# content="Planet Earth is unique in many ways, and here are some of the most fascinating aspects that set it apart:\n\n1. **Water-Rich Surface**: Earth is the only known planet in our solar system with liquid water on its surface. About 71% of the Earth's surface is covered by oceans, which are essential for life as we know it.\n\n2. **Biodiversity**: Earth supports a vast and diverse range of life forms, from microscopic bacteria to massive blue whales. The planet's ecosystems, from rainforests to deserts to the deep sea, are home to millions of species.\n\n3. **Plate Tect" additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 128, 'prompt_tokens': 14, 'total_tokens': 142}, 'model_name': 'Qwen/Qwen2.5-72B-Instruct', 'system_fingerprint': None, 'finish_reason': 'length', 'logprobs': None} id='lc_run--01a0d756-6e3b-7563-a806-1327819ca604-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 14, 'output_tokens': 128, 'total_tokens': 142}