#------------------------------------------------------------------------------
# Makefile for converting NCAIR1/N-ATLaS Hugging Face model to GGUF + quantized versions
#
# Usage:
#   make download        Download model from Hugging Face
#   make convert         Convert HF model to GGUF (f32)
#   make quantize        Quantize GGUF model (default: Q4_K_M, local build)
#   make push-quantized  Push quantized GGUF model to Hugging Face Hub
#   make all             Run download, convert, and quantize
#------------------------------------------------------------------------------

# Configuration
MODEL_REPO     = NCAIR1/N-ATLaS
MODEL_NAME     = Repo-8.0B
MODEL_DIR      = $(HOME)/models/natlas
CONTAINER      = ghcr.io/ggerganov/llama.cpp:full

LLAMA_CPP_DIR  = llama.cpp
LLAMA_BUILD_DIR= $(LLAMA_CPP_DIR)/build

# HuggingFace Hub configuration
HF_REPO        = inuwamobarak/N-ATLaS-8B-GGUF-Q4_K_M  # Change to your repo
ENV_FILE       = ../.env
HF_TOKEN       ?= $(shell grep HF_TOKEN $(ENV_FILE) 2>/dev/null | cut -d '=' -f2)

#------------------------------------------------------------------------------
# Environment handling
#------------------------------------------------------------------------------
ifneq (,$(wildcard $(ENV_FILE)))
    include $(ENV_FILE)
    export $(shell sed 's/=.*//' $(ENV_FILE) 2>/dev/null)
endif

#------------------------------------------------------------------------------
# Primary Targets
#------------------------------------------------------------------------------
.PHONY: all download convert quantize quantize-local quantize-docker clean help

all: download convert quantize

download:
	@echo "==> Downloading $(MODEL_REPO) into $(MODEL_DIR)"
	mkdir -p $(MODEL_DIR)
	huggingface-cli download $(MODEL_REPO) --local-dir "$(MODEL_DIR)" --include "*"

convert:
	@echo "==> Converting model to GGUF (f32)"
	docker run --rm -v $(MODEL_DIR):/repo $(CONTAINER) \
		--convert /repo --outtype f32

# Default quantization (local build with Metal acceleration)
quantize: quantize-local

#------------------------------------------------------------------------------
# Local llama.cpp Build + Quantization
#------------------------------------------------------------------------------
build-llama:
	@echo "==> Building llama.cpp with Metal support"
	@if [ ! -d "$(LLAMA_CPP_DIR)" ]; then git clone https://github.com/ggerganov/llama.cpp; fi
	@mkdir -p $(LLAMA_BUILD_DIR)
	@cd $(LLAMA_BUILD_DIR) && cmake .. -DLLAMA_METAL=ON
	@cd $(LLAMA_BUILD_DIR) && make -j$$(sysctl -n hw.ncpu)

quantize-local: build-llama
	@echo "==> Quantizing GGUF model to Q4_K_M (local)"
	@$(LLAMA_BUILD_DIR)/bin/llama-quantize \
		$(MODEL_DIR)/$(MODEL_NAME)-F32.gguf \
		$(MODEL_DIR)/ggml-model-Q4_K_M.gguf Q4_K_M

quantize-local-q4_0: build-llama
	@$(LLAMA_BUILD_DIR)/bin/llama-quantize \
		$(MODEL_DIR)/$(MODEL_NAME)-F32.gguf \
		$(MODEL_DIR)/ggml-model-Q4_0.gguf Q4_0

quantize-local-q8_0: build-llama
	@$(LLAMA_BUILD_DIR)/bin/llama-quantize \
		$(MODEL_DIR)/$(MODEL_NAME)-F32.gguf \
		$(MODEL_DIR)/ggml-model-Q8_0.gguf Q8_0

#------------------------------------------------------------------------------
# Docker-based Quantization (fallback) but I recommend local build
#------------------------------------------------------------------------------
quantize-docker:
	@docker run --platform linux/amd64 --memory=32g --rm -v $(MODEL_DIR):/repo $(CONTAINER) \
		--quantize /repo/$(MODEL_NAME)-F32.gguf /repo/ggml-model-Q4_K_M.gguf Q4_K_M

quantize-docker-q4_0:
	@docker run --platform linux/amd64 --memory=16g --rm -v $(MODEL_DIR):/repo $(CONTAINER) \
		--quantize /repo/$(MODEL_NAME)-F32.gguf /repo/ggml-model-Q4_0.gguf Q4_0

quantize-docker-q8_0:
	@docker run --platform linux/amd64 --memory=16g --rm -v $(MODEL_DIR):/repo $(CONTAINER) \
		--quantize /repo/$(MODEL_NAME)-F32.gguf /repo/ggml-model-Q8_0.gguf Q8_0

#------------------------------------------------------------------------------
# Utility Targets
#------------------------------------------------------------------------------
clean:
	@echo "==> Cleaning generated model files"
	rm -f $(MODEL_DIR)/*.gguf $(MODEL_DIR)/*.bin

clean-llama:
	@echo "==> Cleaning llama.cpp build"
	rm -rf $(LLAMA_BUILD_DIR)

clean-all: clean clean-llama
	@echo "==> Full cleanup completed"

#------------------------------------------------------------------------------
# Push to HuggingFace Hub
#------------------------------------------------------------------------------
push-quantized:
	@echo "==> Pushing quantized model to HuggingFace Hub: $(HF_REPO)"
	@test -n "$(HF_TOKEN)" || { echo "Error: HF_TOKEN not found in .env"; exit 1; }
	hf upload $(HF_REPO) $(MODEL_DIR)/ggml-model-Q4_K_M.gguf ggml-model-Q4_K_M.gguf

push-f32:
	@test -n "$(HF_TOKEN)" || { echo "Error: HF_TOKEN not found in .env"; exit 1; }
	hf upload $(HF_REPO) $(MODEL_DIR)/$(MODEL_NAME)-F32.gguf $(MODEL_NAME)-F32.gguf
	@if [ -f "$(MODEL_DIR)/config.json" ]; then \
		hf upload $(HF_REPO) $(MODEL_DIR)/config.json config.json; fi
	@if [ -f "$(MODEL_DIR)/tokenizer.json" ]; then \
		hf upload $(HF_REPO) $(MODEL_DIR)/tokenizer.json tokenizer.json; fi

push-all: push-quantized push-f32

push-with-readme:
	@test -n "$(HF_REPO)" && [ "$(HF_REPO)" != "inuwamobarak/N-ATLaS-8B-GGUF-Q4_K_M" ] || \
		{ echo "Error: Please set HF_REPO in your Makefile"; exit 1; }
	@echo "# N-ATLaS 8B GGUF Quantized" > README.md
	@echo "" >> README.md
	@echo "## Model Details" >> README.md
	@echo "- **Base Model**: $(MODEL_REPO)" >> README.md
	@echo "- **Quantization**: Q4_K_M" >> README.md
	@echo "- **Context Length**: 131,072 tokens" >> README.md
	@echo "" >> README.md
	@echo "## Usage" >> README.md
	@echo '```python' >> README.md
	@echo 'from llama_cpp import Llama' >> README.md
	@echo 'llm = Llama(model_path=\"ggml-model-Q4_K_M.gguf\", n_ctx=4096, n_threads=8)' >> README.md
	@echo '```' >> README.md
	hf upload $(HF_REPO) README.md
	hf upload $(HF_REPO) $(MODEL_DIR)/ggml-model-Q4_K_M.gguf ggml-model-Q4_K_M.gguf
	rm README.md

create-repo:
	@test -n "$(HF_TOKEN)" || { echo "Error: HF_TOKEN not found in .env"; exit 1; }
	huggingface-cli repo create $(HF_REPO) --type model --organization $$(echo $(HF_REPO) | cut -d'/' -f1)

verify-upload:
	huggingface-cli repo-info $(HF_REPO)

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  download            Download model from Hugging Face"
	@echo "  convert             Convert HF model to GGUF (f32) via Docker"
	@echo "  quantize            Quantize GGUF model to Q4_K_M (local build, default)"
	@echo "  quantize-local-*    Quantize with local build (q4_0, q8_0)"
	@echo "  quantize-docker-*   Quantize with Docker (fallback)"
	@echo "  push-quantized      Upload Q4_K_M model to Hugging Face Hub"
	@echo "  push-f32            Upload F32 model to Hugging Face Hub"
	@echo "  push-all            Upload both quantized and F32 models"
	@echo "  push-with-readme    Upload quantized model with README"
	@echo "  clean               Remove generated model files"
	@echo "  clean-llama         Remove llama.cpp build artifacts"
	@echo "  clean-all           Clean everything"