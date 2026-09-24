from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

result = model.invoke('what is the capital of india')

print(result)

#print(result.text)



#Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
#content=[{'type': 'text', 'text': 'The capital of India is **New Delhi**.', 'extras': {'signature': 'EnEKbwFpFH0Tvwq6oryxo+HLUKLoRXT5peeuavH7VELVfu5W5yM4gdYYP91zQ0Visuiz4TFnxHzHpPs62grfqpCcJntWh67Kz+udEQVp4aN8E3rOmNjYcdpikt7NaKtkWqYnnF14SQNehP9RegXLPSn4uA=='}}] additional_kwargs={} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3.1-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--01a0d487-23b0-7bd0-a3e1-97e264ec03df-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 7, 'output_tokens': 9, 'total_tokens': 16, 'input_token_details': {'cache_read': 0}}