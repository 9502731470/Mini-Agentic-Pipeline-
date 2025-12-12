"""Main entry point for the Mini Agentic Pipeline."""
import sys
from orchestrator import Orchestrator
import json

def print_result(result):
    """Pretty print the result."""
    print("\n" + "="*80)
    print("QUERY RESULT")
    print("="*80)
    print(f"\nQuery: {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("\n" + "-"*80)
    print("METADATA")
    print("-"*80)
    print(f"Total Time: {result['metadata']['total_time_ms']:.2f} ms")
    print(f"  - Retrieval: {result['metadata']['retrieval_time_ms']:.2f} ms")
    print(f"  - Reasoning: {result['metadata']['reasoning_time_ms']:.2f} ms")
    if result['metadata']['tool_time_ms'] > 0:
        print(f"  - Tool Execution: {result['metadata']['tool_time_ms']:.2f} ms")
    print(f"  - Synthesis: {result['metadata']['synthesis_time_ms']:.2f} ms")
    print(f"Retrieved Documents: {result['metadata']['num_retrieved_docs']}")
    if result['metadata']['tool_used']:
        print(f"Tool Used: {result['metadata']['tool_used']}")
    print(f"Confidence: {result['metadata']['confidence']}")

def interactive_mode():
    """Run in interactive mode."""
    print("="*80)
    print("Mini Agentic Pipeline - Interactive Mode")
    print("="*80)
    print("Type 'quit' or 'exit' to stop\n")
    
    orchestrator = Orchestrator()
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            result = orchestrator.process_query(query)
            print_result(result)
            
            # Option to show full trace
            show_trace = input("\nShow full trace? (y/n): ").strip().lower()
            if show_trace == 'y':
                print(orchestrator.get_trace_summary())
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()

def single_query_mode(query: str):
    """Process a single query."""
    orchestrator = Orchestrator()
    result = orchestrator.process_query(query)
    print_result(result)
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        single_query_mode(query)
    else:
        # Interactive mode
        interactive_mode()
