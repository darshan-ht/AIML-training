"""
Step 3: Build Vector Store
Generates embeddings using OpenAI-compatible API and stores them in FAISS vector database.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import os
import requests
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from utils.config import (
    PROCESSED_DIR, VECTOR_STORE_DIR, BASE_URL, API_KEY, EMBEDDING_MODEL
)


class CustomOpenAIEmbeddings(Embeddings):
    """Custom embeddings class that directly calls the local LLM embedding API"""
    
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.embedding_url = f"{api_base}/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts"""
        if not texts:
            return []
        
        # Ensure all inputs are strings
        texts = [str(text) for text in texts]
        
        # Make API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "input": texts  # Array of strings
        }
        
        try:
            response = requests.post(
                self.embedding_url,
                json=payload,
                headers=headers,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
        except Exception as e:
            raise Exception(f"Error calling embedding API: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        return self.embed_documents([text])[0]


def load_chunks(input_path: Path) -> list:
    """Load chunks from pickle file"""
    print(f"📂 Loading chunks from: {input_path}")
    with open(input_path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"   ✓ Loaded {len(chunks)} chunks")
    return chunks


def check_llm_health():
    """Check if LLM server is accessible and has a model loaded"""
    import requests
    try:
        # Check if server is reachable
        response = requests.get(f"{BASE_URL.replace('/v1', '')}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ LLM server is reachable")
            return True
    except requests.exceptions.RequestException:
        pass
    
    # Try to check models endpoint
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            if models.get('data'):
                print(f"   ✓ Model loaded: {models['data'][0].get('id', 'unknown')}")
                return True
            else:
                print(f"   ⚠️  LLM server is reachable but no models are loaded")
                return False
    except requests.exceptions.RequestException as e:
        print(f"   ⚠️  Could not check LLM server status: {e}")
        return False
    
    return False


def initialize_embeddings():
    """Initialize OpenAI-compatible embeddings"""
    print(f"\n📦 Initializing embeddings...")
    print(f"   API Base: {BASE_URL}")
    
    # Check LLM health first
    if not check_llm_health():
        print(f"\n   ❌ LLM server issue detected!")
        print(f"   Please ensure:")
        print(f"   1. Your local LLM server is running at {BASE_URL}")
        print(f"   2. A model is loaded (use 'lms load <model>' or load via web UI)")
        raise ConnectionError("LLM server is not ready")
    
    # Use custom embeddings class to ensure correct API format
    embeddings = CustomOpenAIEmbeddings(
        api_base=BASE_URL,
        api_key=API_KEY,
        model=EMBEDDING_MODEL
    )
    print(f"   ✓ Embeddings initialized (model: {EMBEDDING_MODEL})")
    return embeddings


def create_vector_store(chunks: list, embeddings: OpenAIEmbeddings, force_recreate: bool = False):
    """Create or load FAISS vector store"""
    if not chunks:
        raise ValueError("No chunks to create vector store from")
    
    # Check if vector store already exists
    if VECTOR_STORE_DIR.exists() and not force_recreate:
        print(f"\n💾 Loading existing vector store from: {VECTOR_STORE_DIR}")
        try:
            vectorstore = FAISS.load_local(
                str(VECTOR_STORE_DIR),
                embeddings,
                allow_dangerous_deserialization=True
            )
            existing_count = len(vectorstore.index_to_docstore_id)
            print(f"   ✓ Vector store loaded ({existing_count} existing documents)")
            return vectorstore
        except Exception as e:
            print(f"   ⚠️  Error loading vector store: {e}")
            print("   Creating new vector store...")
    
    print(f"\n🔨 Creating vector store...")
    print(f"   Generating embeddings for {len(chunks)} chunks...")
    
    # Extract texts from chunks - ensure they are strings
    texts = [str(chunk.page_content) for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    # Generate embeddings in batches
    print(f"   Generating embeddings (this may take a while)...")
    try:
        # Use embed_documents which handles batching automatically
        # This should send proper format: array of strings
        chunk_embeddings = embeddings.embed_documents(texts)
        print(f"   ✓ Generated {len(chunk_embeddings)} embeddings")
    except Exception as e:
        print(f"   ❌ Error generating embeddings: {e}")
        print(f"   Trying batch processing...")
        # Try smaller batches
        chunk_embeddings = []
        batch_size = 50  # Smaller batches
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            try:
                batch_embeddings = embeddings.embed_documents(batch)
                chunk_embeddings.extend(batch_embeddings)
                print(f"   Processed batch {batch_num}/{total_batches} ({min(i+batch_size, len(texts))}/{len(texts)} chunks)...")
            except Exception as batch_error:
                print(f"   ❌ Error in batch {batch_num}: {batch_error}")
                # Try one at a time for this batch
                print(f"   Trying individual embedding for batch {batch_num}...")
                for j, text in enumerate(batch):
                    try:
                        single_embedding = embeddings.embed_documents([text])
                        chunk_embeddings.extend(single_embedding)
                    except Exception as single_error:
                        print(f"   ❌ Error embedding chunk {i+j}: {single_error}")
                        raise
        print(f"   ✓ Generated {len(chunk_embeddings)} embeddings")
    
    # Create FAISS vector store from embeddings
    # Use from_embeddings method which takes pre-computed embeddings
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, chunk_embeddings)),
        embedding=embeddings,
        metadatas=metadatas
    )
    
    # Save vector store
    vectorstore.save_local(str(VECTOR_STORE_DIR))
    print(f"   ✓ Vector store created and saved to: {VECTOR_STORE_DIR}")
    print(f"   📊 Total documents in store: {len(vectorstore.index_to_docstore_id)}")
    
    return vectorstore


def main():
    """Main function to build vector store"""
    print("=" * 60)
    print("STEP 3: BUILDING VECTOR STORE")
    print("=" * 60 + "\n")
    
    # Load chunks
    input_file = PROCESSED_DIR / "chunks.pkl"
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("   Please run 'make chunk' first to chunk documents.")
        return 1
    
    chunks = load_chunks(input_file)
    
    # Initialize embeddings
    try:
        embeddings = initialize_embeddings()
    except ConnectionError as e:
        print(f"\n❌ Error initializing embeddings: {e}")
        print("\n   Troubleshooting:")
        print(f"   1. Ensure your local LLM server is running at {BASE_URL}")
        print(f"   2. Load a model using: lms load <model-name>")
        print(f"      Or load a model via the web UI at http://127.0.0.1:1234")
        print(f"   3. Verify the model is loaded by checking: {BASE_URL}/models")
        return 1
    except Exception as e:
        print(f"\n❌ Error initializing embeddings: {e}")
        print(f"   Make sure your local LLM is running at {BASE_URL}")
        return 1
    
    # Create vector store
    try:
        vectorstore = create_vector_store(chunks, embeddings)
    except Exception as e:
        print(f"\n❌ Error creating vector store: {e}")
        print("   Make sure your local LLM is running at http://127.0.0.1:1234/v1")
        return 1
    
    print(f"\n✅ Vector store built successfully!")
    print(f"✅ Vector store location: {VECTOR_STORE_DIR}\n")
    return 0


if __name__ == "__main__":
    exit(main())

