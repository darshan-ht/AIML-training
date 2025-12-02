"""
Simple RAG System using LangChain
Main entry point that can run the full pipeline or individual steps.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from scripts.rag_pipeline import main as run_pipeline
from scripts.query import main as run_query


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Simple RAG System - PDF Document Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline then interactive query
  python main.py --pipeline          # Run full pipeline only
  python main.py --query             # Run interactive query mode
  python main.py --query "your question"  # Run single query
        """
    )
    
    parser.add_argument(
        '--pipeline', '-p',
        action='store_true',
        help='Run full pipeline (load -> chunk -> build)'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        nargs='?',
        const=True,
        help='Query mode (interactive if no query provided, or single query)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=3,
        help='Number of top results for query (default: 3)'
    )
    
    args = parser.parse_args()
    
    # If no arguments, run full pipeline then query
    if not args.pipeline and not args.query:
        print("=" * 60)
        print("Simple RAG System - PDF Document Q&A")
        print("=" * 60)
        print("\nRunning full pipeline...\n")
        
        # Run pipeline
        pipeline_result = run_pipeline()
        if pipeline_result != 0:
            print("\n❌ Pipeline failed. Exiting.")
            return pipeline_result
        
        # Run interactive query
        print("\n" + "=" * 60)
        print("Starting interactive query mode...")
        print("=" * 60)
        return run_query()
    
    # Run pipeline only
    if args.pipeline:
        return run_pipeline()
    
    # Run query mode
    if args.query:
        if args.query is True:
            # Interactive mode
            sys.argv = ['query.py']
            return run_query()
        else:
            # Single query
            sys.argv = ['query.py', '--query', args.query, '--top-k', str(args.top_k)]
            return run_query()
    
    return 0


if __name__ == "__main__":
    exit(main())
