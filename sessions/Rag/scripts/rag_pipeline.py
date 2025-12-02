"""
Full RAG Pipeline
Runs all steps in sequence: load -> chunk -> build
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
from utils.config import PROCESSED_DIR


def run_step(script_name: str, step_name: str):
    """Run a pipeline step"""
    print(f"\n{'=' * 60}")
    print(f"Running: {step_name}")
    print('=' * 60)
    
    script_path = Path(__file__).parent / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error in {step_name}")
        return False
    
    return True


def main():
    """Run the full RAG pipeline"""
    print("=" * 60)
    print("RAG PIPELINE - FULL RUN")
    print("=" * 60)
    
    steps = [
        ("load_documents.py", "Step 1: Load Documents"),
        ("chunk_documents.py", "Step 2: Chunk Documents"),
        ("build_vector_store.py", "Step 3: Build Vector Store"),
    ]
    
    for script, name in steps:
        if not run_step(script, name):
            print(f"\n❌ Pipeline failed at {name}")
            return 1
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nYou can now query the vector store using:")
    print("  make query")
    print("  or")
    print("  make query Q='your question here'")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())

