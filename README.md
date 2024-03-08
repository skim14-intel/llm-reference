# rag-reference

This project is based on "https://github.com/nicknochnack/Llama2RAG/tree/main"

#How-to

1. Create your own conda env with python-3.10
2. Run the setup.sh
```bash
bash ./setup.sh
```
4. Initiate oneAPI environments to use XPU(GPU)
  ```bash
source /opt/intel/oneapi/setvars.sh
```
6. Set your hugggingface home directory
```bash
export HF_HOME=/home/{user}/.cache/huggingface/hub/
```
8. Run the "llm-rag.py" which is using "/data/isd-user-guide.pdf" as vector DB
```bash
python llm-rag.py
```
