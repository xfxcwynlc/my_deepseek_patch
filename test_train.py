from torch.optim import AdamW  # <- Use PyTorch's AdamWimport torch

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig

model_name = "deepseek-ai/DeepSeek-V2-Lite"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.candidates = 6

model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

# Set model to training mode
model.train()

# Example text
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)

# Prepare labels: use same as input_ids (causal LM)
inputs["labels"] = inputs["input_ids"].clone()

# Forward pass (returns loss)
outputs = model(**inputs)
loss = outputs.loss
print(f"Training loss: {loss.item()}")

# Backward pass and optimization
loss.backward()
optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer.step()
optimizer.zero_grad()

