"""Quick start script to test the pipeline with a sample query."""
from orchestrator import Orchestrator

def main():
    print("="*80)
    print("Mini Agentic Pipeline - Quick Start")
    print("="*80)
    print("\nThis script demonstrates the pipeline with a sample query.\n")
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Sample query
    query = "What is artificial intelligence and how does it relate to machine learning?"
    
    print(f"Query: {query}\n")
    print("Processing...\n")
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display result
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"\n{result['answer']}\n")
    
    print("-"*80)
    print("METADATA")
    print("-"*80)
    print(f"Total Time: {result['metadata']['total_time_ms']:.2f} ms")
    print(f"Retrieved Documents: {result['metadata']['num_retrieved_docs']}")
    print(f"Tool Used: {result['metadata']['tool_used'] or 'None'}")
    print(f"Confidence: {result['metadata']['confidence']}")
    
    print("\n" + "="*80)
    print("Pipeline executed successfully!")
    print("="*80)
    print("\nTo run more queries, use: python main.py")
    print("To run evaluation, use: python evaluate.py")

if __name__ == "__main__":
    main()


