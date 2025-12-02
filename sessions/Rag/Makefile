# Makefile for RAG Pipeline
# This Makefile orchestrates the RAG pipeline in multiple steps

.PHONY: help load chunk build query all clean setup check install clean-all test

# Python interpreter - use venv if available, otherwise system python
ifeq ($(wildcard .venv/bin/python3),)
    PYTHON := python3
else
    PYTHON := .venv/bin/python3
endif

# Directories
SCRIPTS_DIR := scripts
DOCS_DIR := docs
PROCESSED_DIR := data/processed
VECTOR_STORE_DIR := faiss_store

# Files
DOCUMENTS_FILE := $(PROCESSED_DIR)/documents.pkl
CHUNKS_FILE := $(PROCESSED_DIR)/chunks.pkl

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "=========================================="
	@echo "RAG Pipeline - Makefile Commands"
	@echo "=========================================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install      # Install dependencies (first time setup)"
	@echo "  make all          # Run full pipeline (load -> chunk -> build)"
	@echo "  make load         # Step 1: Load documents"
	@echo "  make chunk        # Step 2: Chunk documents"
	@echo "  make build        # Step 3: Build vector store"
	@echo "  make query        # Step 4: Query vector store (interactive)"
	@echo "  make query Q='What is mentioned about leave policy?'  # Query with specific question"
	@echo "  make clean        # Clean processed files"
	@echo "  make check        # Check pipeline status"
	@echo ""

setup: ## Check and setup environment
	@echo "🔍 Checking environment..."
	@$(PYTHON) --version
	@echo "✅ Python found"
	@if [ ! -d "$(DOCS_DIR)" ]; then \
		echo "⚠️  Warning: docs directory not found. Creating it..."; \
		mkdir -p $(DOCS_DIR); \
		echo "   Please add PDF files to $(DOCS_DIR)"; \
	fi
	@mkdir -p $(PROCESSED_DIR) $(VECTOR_STORE_DIR)
	@echo "✅ Directories ready"

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	@if [ -d ".venv" ]; then \
		echo "   Using virtual environment: .venv"; \
		command -v uv > /dev/null && uv pip install -r requirements.txt || .venv/bin/pip install -r requirements.txt; \
	else \
		echo "   ⚠️  No virtual environment found. Creating one..."; \
		if command -v uv > /dev/null; then \
			uv venv && uv pip install -r requirements.txt; \
		else \
			python3 -m venv .venv && \
			.venv/bin/pip install --upgrade pip && \
			.venv/bin/pip install -r requirements.txt; \
		fi; \
		echo "   ✅ Virtual environment created and dependencies installed"; \
	fi
	@echo "✅ Dependencies installed"

load: setup ## Step 1: Load documents from PDF files
	@echo ""
	@$(PYTHON) $(SCRIPTS_DIR)/load_documents.py

chunk: ## Step 2: Chunk documents into smaller pieces
	@echo ""
	@if [ ! -f "$(DOCUMENTS_FILE)" ]; then \
		echo "❌ Documents file not found. Run 'make load' first."; \
		exit 1; \
	fi
	@$(PYTHON) $(SCRIPTS_DIR)/chunk_documents.py

build: ## Step 3: Build vector store with embeddings
	@echo ""
	@if [ ! -f "$(CHUNKS_FILE)" ]; then \
		echo "❌ Chunks file not found. Run 'make chunk' first."; \
		exit 1; \
	fi
	@$(PYTHON) $(SCRIPTS_DIR)/build_vector_store.py

query: ## Step 4: Query vector store (interactive mode)
	@echo ""
	@if [ -z "$(Q)" ]; then \
		$(PYTHON) $(SCRIPTS_DIR)/query.py; \
	else \
		$(PYTHON) $(SCRIPTS_DIR)/query.py --query "$(Q)"; \
	fi

all: setup load chunk build ## Run full pipeline (all steps in sequence)
	@echo ""
	@echo "=========================================="
	@echo "✅ Full pipeline completed successfully!"
	@echo "=========================================="
	@echo ""
	@echo "You can now query the vector store:"
	@echo "  make query"
	@echo "  or"
	@echo "  make query Q='your question here'"
	@echo ""

pipeline: all ## Alias for 'all' - run full pipeline

clean: ## Clean processed files (keeps vector store)
	@echo "🧹 Cleaning processed files..."
	@rm -rf $(PROCESSED_DIR)
	@echo "✅ Cleaned processed files"
	@echo "   (Vector store preserved at $(VECTOR_STORE_DIR))"

clean-all: clean ## Clean everything including vector store
	@echo "🧹 Cleaning vector store..."
	@rm -rf $(VECTOR_STORE_DIR)
	@echo "✅ Cleaned vector store"
	@echo "   Run 'make all' to rebuild everything"

check: ## Check pipeline status
	@echo "📊 Pipeline Status:"
	@echo ""
	@if [ -f "$(DOCUMENTS_FILE)" ]; then \
		echo "  ✅ Documents loaded"; \
	else \
		echo "  ❌ Documents not loaded (run 'make load')"; \
	fi
	@if [ -f "$(CHUNKS_FILE)" ]; then \
		echo "  ✅ Documents chunked"; \
	else \
		echo "  ❌ Documents not chunked (run 'make chunk')"; \
	fi
	@if [ -d "$(VECTOR_STORE_DIR)" ] && [ -f "$(VECTOR_STORE_DIR)/index.faiss" ]; then \
		echo "  ✅ Vector store built"; \
	else \
		echo "  ❌ Vector store not built (run 'make build')"; \
	fi
	@echo ""

test: ## Run a test query
	@echo "🧪 Running test query..."
	@$(PYTHON) $(SCRIPTS_DIR)/query.py --query "What is mentioned in the documents?" --top-k 2

run: query ## Alias for 'query' - run interactive query mode
