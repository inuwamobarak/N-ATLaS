# N-ATLaS-LLM Demo

This project demonstrates how to use the **N-ATLaS-LLM**, a multilingual large language model fine-tuned for African languages (Hausa, Igbo, Yoruba) alongside English.  

---

## Features
- Simple Python wrapper around Hugging Face's `transformers` library
- Chat-style inference with role-based message formatting
- Configurable generation parameters
- Supports CUDA / auto device mapping

---

## Setup

1. Clone the repo:
```bash
git clone https://github.com/your-org/n_atlas_demo.git
cd n_atlas_demo

## HuggingFace Hub .gguf files

https://huggingface.co/inuwamobarak/N-ATLaS-8B-GGUF-Q4_K_M/tree/main


## Try with llama.cpp

```bash
# Install python bindings
pip install llama-cpp-python

```bash
# Use the model
from llama_cpp import Llama

llm = Llama(
    model_path="model_path/models/natlas/ggml-model-Q4_K_M.gguf",
    n_ctx=4096,  # Context length
    n_threads=8,  # Number of CPU threads
)

response = llm(
    "What is the future of AI?",
    max_tokens=100,
    temperature=0.7,
)

print(response['choices'][0]['text'])