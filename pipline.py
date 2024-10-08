import transformers
import torch

model_id = "/mnt/qwen2.5-72B/LLM-Research/Meta-Llama-3___1-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="cuda:0"
)

print(pipeline("Hey how are you doing today?"))