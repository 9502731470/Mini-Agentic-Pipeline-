"""Evaluation script with test queries."""
import json
import time
from orchestrator import Orchestrator
from datetime import datetime

# Test queries covering different scenarios
TEST_QUERIES = [
    {
        "id": 1,
        "query": "What is artificial intelligence?",
        "category": "KB_only",
        "expected_tool": None
    },
    {
        "id": 2,
        "query": "Explain how neural networks work",
        "category": "KB_only",
        "expected_tool": None
    },
    {
        "id": 3,
        "query": "What are the latest developments in AI in 2024?",
        "category": "needs_web_search",
        "expected_tool": "web_search"
    },
    {
        "id": 4,
        "query": "What is the price of a wireless mouse?",
        "category": "needs_csv_lookup",
        "expected_tool": "csv_lookup"
    },
    {
        "id": 5,
        "query": "How does RAG work?",
        "category": "KB_only",
        "expected_tool": None
    },
    {
        "id": 6,
        "query": "What is the current price of Bitcoin?",
        "category": "needs_web_search",
        "expected_tool": "web_search"
    },
    {
        "id": 7,
        "query": "Find products under $50 in the catalog",
        "category": "needs_csv_lookup",
        "expected_tool": "csv_lookup"
    },
    {
        "id": 8,
        "query": "Compare transformer models and RNNs",
        "category": "KB_only",
        "expected_tool": None
    },
    {
        "id": 9,
        "query": "What are the ethical concerns with AI?",
        "category": "KB_only",
        "expected_tool": None
    },
    {
        "id": 10,
        "query": "What is the weather today in San Francisco?",
        "category": "needs_web_search",
        "expected_tool": "web_search"
    },
    {
        "id": 11,
        "query": "Explain embeddings and vector databases",
        "category": "KB_only",
        "expected_tool": None
    },
    {
        "id": 12,
        "query": "What products are available in the Electronics category?",
        "category": "needs_csv_lookup",
        "expected_tool": "csv_lookup"
    }
]

def evaluate():
    """Run evaluation on test queries."""
    print("="*80)
    print("EVALUATION: Mini Agentic Pipeline")
    print("="*80)
    print(f"Test Queries: {len(TEST_QUERIES)}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    orchestrator = Orchestrator()
    results = []
    
    for i, test_query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Processing: {test_query['query']}")
        print("-" * 80)
        
        try:
            result = orchestrator.process_query(test_query['query'])
            
            eval_result = {
                "test_id": test_query['id'],
                "query": test_query['query'],
                "category": test_query['category'],
                "expected_tool": test_query['expected_tool'],
                "actual_tool": result['metadata']['tool_used'],
                "tool_match": test_query['expected_tool'] == result['metadata']['tool_used'],
                "total_time_ms": result['metadata']['total_time_ms'],
                "retrieval_time_ms": result['metadata']['retrieval_time_ms'],
                "reasoning_time_ms": result['metadata']['reasoning_time_ms'],
                "tool_time_ms": result['metadata']['tool_time_ms'],
                "synthesis_time_ms": result['metadata']['synthesis_time_ms'],
                "num_retrieved_docs": result['metadata']['num_retrieved_docs'],
                "confidence": result['metadata']['confidence'],
                "answer_length": len(result['answer']),
                "answer_preview": result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            }
            
            results.append(eval_result)
            
            print(f"✓ Completed in {result['metadata']['total_time_ms']:.2f} ms")
            print(f"  Tool: {result['metadata']['tool_used'] or 'None'}")
            print(f"  Confidence: {result['metadata']['confidence']}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results.append({
                "test_id": test_query['id'],
                "query": test_query['query'],
                "error": str(e)
            })
    
    # Generate summary report
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\nTotal Queries: {len(TEST_QUERIES)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_time = sum(r['total_time_ms'] for r in successful) / len(successful)
        avg_retrieval = sum(r['retrieval_time_ms'] for r in successful) / len(successful)
        avg_reasoning = sum(r['reasoning_time_ms'] for r in successful) / len(successful)
        avg_tool = sum(r['tool_time_ms'] for r in successful) / len(successful)
        avg_synthesis = sum(r['synthesis_time_ms'] for r in successful) / len(successful)
        
        print(f"\nAverage Latency:")
        print(f"  Total: {avg_time:.2f} ms")
        print(f"  Retrieval: {avg_retrieval:.2f} ms")
        print(f"  Reasoning: {avg_reasoning:.2f} ms")
        print(f"  Tool Execution: {avg_tool:.2f} ms")
        print(f"  Synthesis: {avg_synthesis:.2f} ms")
        
        # Tool usage analysis
        tool_usage = {}
        for r in successful:
            tool = r['actual_tool'] or 'none'
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        print(f"\nTool Usage:")
        for tool, count in tool_usage.items():
            print(f"  {tool}: {count}")
        
        # Tool decision accuracy
        tool_matches = sum(1 for r in successful if r.get('tool_match', False))
        tool_decisions = len([r for r in successful if r['expected_tool'] is not None])
        if tool_decisions > 0:
            accuracy = (tool_matches / tool_decisions) * 100
            print(f"\nTool Decision Accuracy: {accuracy:.1f}% ({tool_matches}/{tool_decisions})")
    
    # Save detailed results
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_queries": len(TEST_QUERIES),
                "successful": len(successful),
                "failed": len(failed)
            },
            "results": results
        }, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    evaluate()
