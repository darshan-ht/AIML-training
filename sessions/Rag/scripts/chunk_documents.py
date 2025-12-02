"""
Step 2: Chunk Documents
Splits documents into smaller chunks for better RAG performance.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(input_path: Path) -> list:
    """Load documents from pickle file"""
    print(f"📂 Loading documents from: {input_path}")
    with open(input_path, 'rb') as f:
        documents = pickle.load(f)
    print(f"   ✓ Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    """Split documents into chunks"""
    print(f"\n✂️  Chunking documents...")
    print(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"   ✓ Split {len(documents)} documents into {len(chunks)} chunks")
    
    # Show example chunk
    if chunks:
        print(f"\n   Example chunk:")
        print(f"   Content preview: {chunks[0].page_content[:150]}...")
        print(f"   Metadata: {chunks[0].metadata}")
    
    return chunks


def save_chunks(chunks: list, output_path: Path):
    """Save chunks to pickle file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"\n   💾 Saved {len(chunks)} chunks to {output_path}")


def main():
    """Main function to chunk documents"""
    print("=" * 60)
    print("STEP 2: CHUNKING DOCUMENTS")
    print("=" * 60 + "\n")
    
    # Load documents
    input_file = PROCESSED_DIR / "documents.pkl"
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("   Please run 'make load' first to load documents.")
        return 1
    
    documents = load_documents(input_file)
    
    # Chunk documents
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Save chunks
    output_file = PROCESSED_DIR / "chunks.pkl"
    save_chunks(chunks, output_file)
    
    print(f"\n✅ Chunking complete!")
    print(f"✅ Chunks saved to: {output_file}\n")
    return 0


if __name__ == "__main__":
    exit(main())

