import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig

model_name = "deepseek-ai/DeepSeek-V2-Lite"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.candidates = 6

model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

model.generation_config = GenerationConfig.from_pretrained(model_name)

model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
