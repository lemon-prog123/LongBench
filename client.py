from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = ""
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
base_url="http://localhost:3003/v1",
api_key="122",
)

chat_response = client.chat.completions.create(
    model="/mnt/qwen2.5-72B/qwen/Qwen2___5-72B-Instruct",
    messages=[
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "I am Jack,hello"},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response)
  
