from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

result = model.invoke("what is the capital of india")

print(result)

# content="The capital of India is **New Delhi**.\n\nNew Delhi is located in the northern part of India and serves as the seat of the Indian government. It's part of the larger Delhi metropolitan area and was designed by British architect Edwin Lutyens in the early 20th century." additional_kwargs={} response_metadata={'id': 'msg_011CfNgX4TTCoaibUNTSip6A', 'container': None, 'model': 'claude-haiku-4-5-20251001', 'stop_details': None, 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0}, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'inference_geo': 'not_available', 'input_tokens': 13, 'output_tokens': 61, 'output_tokens_details': None, 'server_tool_use': None, 'service_tier': 'standard'}, 'diagnostics': None, 'model_name': 'claude-haiku-4-5-20251001', 'model_provider': 'anthropic'} id='lc_run--01a0d486-4fac-7b00-a90b-6e1d7f33a15a-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 13, 'output_tokens': 61, 'total_tokens': 74, 'input_token_details': {'cache_read': 0, 'cache_creation': 0, 'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 0}}
