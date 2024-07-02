import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

class TextGenerator:
    def __init__(self, model_name, device='cpu'):
        self.max_output_tokens = 1024
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        my_hf_dir = os.environ.get('HF_HOME')
        if not my_hf_dir:
            raise EnvironmentError("Please set the HF_HOME environment variable.")
        print(f"Using hf from {my_hf_dir}")

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)

    def generate_text(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        # Generate text for the first token
        start_time = time.time()
        output = self.model.generate(input_ids, max_new_tokens=1)
        first_token_generation_time = time.time() - start_time

        # Generate text for the remaining tokens
        start_time = time.time()
        output = self.model.generate(input_ids, max_new_tokens=self.max_output_tokens)
        throughput_time = (time.time() - start_time) / self.max_output_tokens

        # Decode and print the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return {
            "initialize_xpu_time": None,  # This should be measured separately if needed
            "first_token_generation_time": first_token_generation_time,
            "throughput_time": throughput_time,
            "generated_text": generated_text
        }

# Usage
model_name = "meta-llama/Llama-2-7b-chat-hf"  # or "meta-llama/Meta-Llama-3-8B-Instruct"
device = "xpu"
text_generator = TextGenerator(model_name, device)

input_text = "Once upon a time"
results = text_generator.generate_text(input_text)

print("First Token Generation Time:", results["first_token_generation_time"])
print("Throughput Time:", results["throughput_time"])
print("Generated Text:", results["generated_text"])
