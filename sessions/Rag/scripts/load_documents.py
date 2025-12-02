"""
Step 1: Load Documents
Loads all PDF files from the docs directory and saves them as processed documents.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import pickle
from utils.config import DOCS_DIR, PROCESSED_DIR


def load_pdfs(pdf_dir: Path) -> list:
    """Load all PDF files from directory"""
    if not pdf_dir.exists():
        print(f"⚠️  PDF directory not found: {pdf_dir}")
        return []
    
    pdf_files = [f for f in pdf_dir.iterdir() if f.suffix.lower() == ".pdf"]
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in: {pdf_dir}")
        return []
    
    print(f"📄 Loading PDFs from: {pdf_dir}")
    all_documents = []
    
    for pdf_file in pdf_files:
        print(f"   Loading: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            all_documents.extend(documents)
            print(f"      ✓ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"      ✗ Error loading {pdf_file.name}: {e}")
    
    print(f"   ✓ Total: {len(all_documents)} pages from {len(pdf_files)} PDF files")
    return all_documents


def save_documents(documents: list, output_path: Path):
    """Save documents to pickle file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(documents, f)
    print(f"   💾 Saved {len(documents)} documents to {output_path}")


def main():
    """Main function to load all documents"""
    print("=" * 60)
    print("STEP 1: LOADING DOCUMENTS")
    print("=" * 60 + "\n")
    
    # Load PDFs
    documents = load_pdfs(DOCS_DIR)
    
    if not documents:
        print("\n❌ No documents found! Please check your docs directory.")
        print(f"   Expected location: {DOCS_DIR}")
        return 1
    
    # Save processed documents
    output_file = PROCESSED_DIR / "documents.pkl"
    save_documents(documents, output_file)
    
    print(f"\n✅ Total documents loaded: {len(documents)}")
    print(f"✅ Documents saved to: {output_file}\n")
    return 0


if __name__ == "__main__":
    exit(main())

