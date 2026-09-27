"""
Tries several candidate models against Hugging Face's Inference Providers
(conversational task) and prints the FULL raw error for each failure —
repr(e), str(e), and the raw HTTP response body if available — so you can
see exactly what's in there before deciding what to filter/extract.

Setup:
    pip install langchain-huggingface python-dotenv --break-system-packages

.env file (custom variable name, NOT the default HUGGINGFACEHUB_API_TOKEN):
    HF_KEY=your-hf-token-here
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.environ.get("HF_KEY")

if not hf_token:
    raise ValueError("HF_KEY not found in environment. Check your .env file.")

CANDIDATE_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3.5-mini-instruct",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "tiiuae/falcon-7b-instruct",
]

PROMPT = "Tell one unique fact about planet Earth in one sentence."


def try_model(repo_id: str):
    """Attempt to invoke a model. Prints full raw error details on failure."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="conversational",
            max_new_tokens=64,
            huggingfacehub_api_token=hf_token,
        )
        model = ChatHuggingFace(llm=llm)
        result = model.invoke(PROMPT)
        print(f"  ✅ WORKS — response: {result.content}\n")
        return True

    except Exception as e:
        print("  ❌ FAILED")
        print("  ---- repr(e) ----")
        print(" ", repr(e))
        print("  ---- str(e) ----")
        print(" ", str(e))

        # Some huggingface_hub errors carry the raw HTTP response object.
        # This is the actual JSON body the server sent back, unfiltered.
        if hasattr(e, "response") and e.response is not None:
            print("  ---- e.response.status_code ----")
            print(" ", e.response.status_code)
            print("  ---- e.response.text ----")
            print(" ", e.response.text)
        else:
            print("  (no .response attribute on this exception)")

        print()  # blank line for readability between models
        return False


if __name__ == "__main__":
    print(f"Testing {len(CANDIDATE_MODELS)} models...\n")

    working_models = []

    for repo_id in CANDIDATE_MODELS:
        print(f"Trying: {repo_id}")
        if try_model(repo_id):
            working_models.append(repo_id)

    print("=" * 60)
    if working_models:
        print(f"Working models ({len(working_models)}):")
        for m in working_models:
            print(f"  - {m}")
    else:
        print("None of the candidate models worked.")