#!/bin/bash

# This version of IPEX shall be run w/ 2024.0
python -m pip install langchain==0.0.218
python -m pip install llama-index==0.7.21
python -m pip install llama_hub==0.0.19

python -m pip install transformers
python -m pip install sentence_transformers
python -m pip install sentencepiece==0.2.0

python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/


