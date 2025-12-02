"""Configuration settings for RAG pipeline"""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR = BASE_DIR / "faiss_store"

# LLM Settings
BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "dummy-key"  # Not used but required by OpenAI format
MODEL_NAME = "qwen2.5-coder-32b-instruct"  # Chat model name (adjust based on your local LLM)
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5-embedding"  # Embedding model name

# Processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Query settings
DEFAULT_TOP_K = 3

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

