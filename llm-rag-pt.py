''' 
1. Please login huggingface before run this script
2. For Intel GPU, you need to set oneAPI variables.
3. Set your huggingface home dir as HF_HOME
'''
# Import transformer classes for generation
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes 
import torch
#import intel_extension_for_pytorch as ipex
import os

#print(ipex.xpu.get_device_name(0))

# Define variable to hold llama2 weights naming 
name = "meta-llama/Llama-2-7b-chat-hf"
# Set auth token variable from hugging face 
#auth_token = "YOUR HUGGING FACE AUTH TOKEN HERE"
my_hf_dir = os.environ.get('HF_HOME')

if my_hf_dir:
    print(f" Use hf from {my_hf_dir}")
else:
    print("Please set the HF_HOME")
    quit()

my_db_file = os.getcwd()+ "/data/isd-user-guide.pdf"

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(name, 
    cache_dir=my_hf_dir)

# Create model
model = AutoModelForCausalLM.from_pretrained(name, 
    cache_dir=my_hf_dir,torch_dtype=torch.float32, 
    rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=False) 

# Use IPEX
# Move to Intel GPU
model = model.eval().to("xpu")

# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Create a system prompt 
system_prompt = """[INST] <>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.<>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Complete the query prompt
query_wrapper_prompt.format(query_str='hello')

# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM
# Create a HF LLM using the llama index wrapper 
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Create and dl embeddings instance  
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)

# Import deps to load documents 
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path

# Download PDF Loader 
PyMuPDFReader = download_loader("PyMuPDFReader")
# Create PDF Loader
loader = PyMuPDFReader()
# Load documents 
documents = loader.load(file_path=Path(my_db_file), metadata=True)

# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)

# Setup index query engine using LLM 
query_engine = index.as_query_engine()

# Test out a query in natural
#response = query_engine.query("what was the FY2022 return on equity?")

my_prompt = "null"

while my_prompt:

    my_prompt = input("How can I help you ? \n")
    if my_prompt == 'quit':
        quit()
    else:
        response = query_engine.query(my_prompt)
        print(f"In ISD - {response}")
