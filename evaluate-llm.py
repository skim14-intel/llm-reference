import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
import os

max_output_token = 1024

# Set auth token variable from hugging face 
#auth_token = "YOUR HUGGING FACE AUTH TOKEN HERE"
my_hf_dir = os.environ.get('HF_HOME')

if my_hf_dir:
    print(f" Use hf from {my_hf_dir}")
else:
    print("Please set the HF_HOME")
    quit()

# Load the Llama3 model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

start_time = time.time()
# Set the device to run the model on
device = "xpu"
model.to(device)

# Generate text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
end_time = time.time()
initialize_xpu_time = end_time - start_time

# Measure performance

# Generate text for the first token
start_time = time.time()
output = model.generate(input_ids, max_new_tokens=1)
end_time = time.time()
first_token_generation_time = end_time - start_time

# Generate text for the remaining tokens
start_time = time.time()
output = model.generate(input_ids, max_new_tokens=max_output_token)
end_time = time.time()
throughput_time = (end_time - start_time) / max_output_token

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Initialize XPU:", initialize_xpu_time)
print("First Token Generation Time:", first_token_generation_time)
print("Throughput Time:", throughput_time)
print("Generated Text:", generated_text)