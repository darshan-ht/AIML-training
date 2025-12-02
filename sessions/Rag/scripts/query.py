"""
Step 4: Query Vector Store
Query the vector store using RAG chain with local OpenAI-compatible LLM.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import requests
from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from utils.config import (
    VECTOR_STORE_DIR, BASE_URL, API_KEY, MODEL_NAME, EMBEDDING_MODEL, DEFAULT_TOP_K
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


def initialize_vector_store():
    """Initialize FAISS vector store"""
    if not VECTOR_STORE_DIR.exists():
        print(f"❌ Vector store not found: {VECTOR_STORE_DIR}")
        print("   Please run 'make build' first to build the vector store.")
        return None
    
    print(f"📊 Loading vector store from: {VECTOR_STORE_DIR}")
    
    try:
        embeddings = CustomOpenAIEmbeddings(
            api_base=BASE_URL,
            api_key=API_KEY,
            model=EMBEDDING_MODEL
        )
        vectorstore = FAISS.load_local(
            str(VECTOR_STORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        count = len(vectorstore.index_to_docstore_id)
        print(f"   ✓ Vector store loaded ({count} documents)")
        return vectorstore
    except Exception as e:
        print(f"   ❌ Error loading vector store: {e}")
        return None


def create_rag_chain(vectorstore, top_k: int = DEFAULT_TOP_K):
    """Create RAG chain with LLM and retriever"""
    print(f"\n🤖 Initializing LLM...")
    print(f"   API Base: {BASE_URL}")
    print(f"   Model: {MODEL_NAME}")
    
    try:
        llm = ChatOpenAI(
            openai_api_base=BASE_URL,
            openai_api_key=API_KEY,
            model=MODEL_NAME,
            temperature=0
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        print(f"   ✓ RAG chain ready (top_k={top_k})")
        return qa
    except Exception as e:
        print(f"   ❌ Error creating RAG chain: {e}")
        print("   Make sure your local LLM is running at http://127.0.0.1:1234/v1")
        return None


def query_rag(qa_chain, query: str):
    """Query the RAG chain"""
    print(f"\n🔍 Querying: '{query}'")
    print("   Generating answer...")
    
    try:
        result = qa_chain.invoke({"query": query})
        
        print("\n" + "-" * 60)
        print("📝 Answer:")
        print(result["result"])
        print("-" * 60)
        
        # Show source documents
        if result.get("source_documents"):
            print(f"\n📚 Retrieved {len(result['source_documents'])} relevant chunk(s):")
            for i, doc in enumerate(result['source_documents'][:3], 1):
                source = doc.metadata.get('source', 'Unknown')
                preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                print(f"\n   [{i}] Source: {source}")
                print(f"       Preview: {preview}")
        
        return result
    except Exception as e:
        print(f"\n❌ Error querying: {e}")
        print("   Make sure your local LLM is running at http://127.0.0.1:1234/v1")
        return None


def interactive_mode(qa_chain):
    """Run interactive query mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE QUERY MODE")
    print("=" * 60)
    print("Enter queries (type 'quit' or 'exit' to stop):\n")
    
    while True:
        try:
            query = input("❓ Ask a question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            query_rag(qa_chain, query)
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Query the RAG vector store")
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Query string to search'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=DEFAULT_TOP_K,
        help=f'Number of top results (default: {DEFAULT_TOP_K})'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STEP 4: QUERYING VECTOR STORE")
    print("=" * 60 + "\n")
    
    # Initialize vector store
    vectorstore = initialize_vector_store()
    if not vectorstore:
        return 1
    
    # Create RAG chain
    qa_chain = create_rag_chain(vectorstore, args.top_k)
    if not qa_chain:
        return 1
    
    # Query or interactive mode
    if args.query:
        query_rag(qa_chain, args.query)
    else:
        interactive_mode(qa_chain)
    
    return 0


if __name__ == "__main__":
    exit(main())

