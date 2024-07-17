import os
import time
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

class TextGenerator:
    def __init__(self, model_name, device='cpu'):
        self.max_output_tokens = 256
        self.input_token = "64"
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.batch_size = 1
        self.initialize_model()

    def initialize_model(self):
        my_hf_dir = os.environ.get('HF_HOME')
        if not my_hf_dir:
            #raise EnvironmentError("Please set the HF_HOME environment variable.")
            my_hf_dir = "/home/model"
        print(f"Using hf from {my_hf_dir}")

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set the padding token if not already defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device, dtype=torch.bfloat16)

#    @profile
    def generate_text(self, input_texts):
        # Tokenize a batch of input texts
        encoding = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Generate text for the first token
        start_time = time.time()
        output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1)
        first_token_generation_time = time.time() - start_time

        # Generate text for the remaining tokens
        start_time = time.time()
        output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=self.max_output_tokens)
        throughput_time = (time.time() - start_time) / (self.max_output_tokens * len(input_texts))
        throughput_token = (1 * self.batch_size)/ throughput_time

        # Decode and print the generated text for each input in the batch
        generated_texts = [self.tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in output]

        return {
            "initialize_xpu_time": None,  # This should be measured separately if needed
            "first_token_generation_time": first_token_generation_time,
            "throughput_token": throughput_token,
            "generated_texts": generated_texts
        }
    
#@profile
def run_llm():
# Usage
    #model_name = "meta-llama/Llama-2-7b-chat-hf"  # or 
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "xpu"
    text_generator = TextGenerator(model_name, device)

    #input_text = ["Once upon a time"] * text_generator.batch_size
    current_path = os.path.dirname(__file__)
    with open(str(current_path) + "/prompt.json") as f:
    #with open("/home/seankim/workspace/rag-reference" + "/prompt.json") as f:
        prompt_pool = json.load(f)
    prompt = prompt_pool[text_generator.input_token]

    input_text = [prompt]*text_generator.batch_size

    results = text_generator.generate_text(input_text)

    print("First Token Generation Time:", results["first_token_generation_time"])
    print("Throughput Token/Sec:", results["throughput_token"])
    for i, text in enumerate(results["generated_texts"]):
        print(f"Generated Text {i+1}:", text)
        print("------------------------------------------------------")

if __name__ == "__main__":
    run_llm()