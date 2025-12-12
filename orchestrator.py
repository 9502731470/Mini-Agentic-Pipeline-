"""Orchestrator/Controller that manages the agentic pipeline workflow."""
import time
from typing import Dict, List, Optional
from retriever import Retriever
from reasoner import Reasoner
from actor import Actor
import json

class Orchestrator:
    """Main orchestrator that coordinates retriever, reasoner, and actor."""
    
    def __init__(self):
        self.retriever = Retriever()
        self.reasoner = Reasoner()
        self.actor = Actor()
        self.trace_log = []
    
    def log_step(self, step_name: str, data: Dict):
        """Log a step in the pipeline.
        
        Args:
            step_name: Name of the step
            data: Step data to log
        """
        log_entry = {
            "timestamp": time.time(),
            "step": step_name,
            "data": data
        }
        self.trace_log.append(log_entry)
        print(f"\n[STEP] {step_name}")
        print(f"Data: {json.dumps(data, indent=2, default=str)}")
    
    def process_query(self, user_query: str) -> Dict:
        """Process a user query through the full pipeline.
        
        Args:
            user_query: User's question
            
        Returns:
            Dict with answer, trace, and metadata
        """
        start_time = time.time()
        self.trace_log = []  # Reset trace for new query
        
        # Step 1: Retrieve relevant context from KB
        self.log_step("RETRIEVAL", {"query": user_query})
        retrieved_docs = self.retriever.search(user_query)
        retrieval_time = time.time() - start_time
        
        self.log_step("RETRIEVAL_COMPLETE", {
            "num_docs": len(retrieved_docs),
            "docs": [{"id": doc.get("id"), "distance": doc.get("distance")} 
                    for doc in retrieved_docs],
            "time_ms": retrieval_time * 1000
        })
        
        # Step 2: Reason about query and decide if tool is needed
        self.log_step("REASONING", {"query": user_query})
        reasoning_result = self.reasoner.reason(user_query, retrieved_docs)
        reasoning_time = time.time() - start_time - retrieval_time
        
        self.log_step("REASONING_COMPLETE", {
            "reasoning": reasoning_result.get("reasoning"),
            "needs_tool": reasoning_result.get("needs_tool"),
            "tool_name": reasoning_result.get("tool_name"),
            "tool_query": reasoning_result.get("tool_query"),
            "confidence": reasoning_result.get("confidence"),
            "time_ms": reasoning_time * 1000
        })
        
        # Step 3: Execute tool if needed
        tool_results = None
        tool_time = 0
        if reasoning_result.get("needs_tool") and reasoning_result.get("tool_name") != "none":
            tool_start = time.time()
            self.log_step("TOOL_EXECUTION", {
                "tool": reasoning_result.get("tool_name"),
                "query": reasoning_result.get("tool_query")
            })
            
            # Execute tool
            tool_results = self.actor.execute_tool(
                reasoning_result.get("tool_name"),
                reasoning_result.get("tool_query")
            )
            
            tool_time = time.time() - tool_start
            self.log_step("TOOL_EXECUTION_COMPLETE", {
                "tool": reasoning_result.get("tool_name"),
                "success": tool_results.get("success"),
                "time_ms": tool_time * 1000,
                "result_preview": str(tool_results.get("result", ""))[:200]
            })
        
        # Step 4: Synthesize final answer
        self.log_step("SYNTHESIS", {"query": user_query})
        synthesis_start = time.time()
        final_answer = self.reasoner.synthesize_answer(
            user_query,
            retrieved_docs,
            tool_results
        )
        synthesis_time = time.time() - synthesis_start
        
        total_time = time.time() - start_time
        
        self.log_step("SYNTHESIS_COMPLETE", {
            "time_ms": synthesis_time * 1000
        })
        
        # Compile result
        result = {
            "query": user_query,
            "answer": final_answer,
            "trace": self.trace_log,
            "metadata": {
                "total_time_ms": total_time * 1000,
                "retrieval_time_ms": retrieval_time * 1000,
                "reasoning_time_ms": reasoning_time * 1000,
                "tool_time_ms": tool_time * 1000,
                "synthesis_time_ms": synthesis_time * 1000,
                "num_retrieved_docs": len(retrieved_docs),
                "tool_used": reasoning_result.get("tool_name") if reasoning_result.get("needs_tool") else None,
                "confidence": reasoning_result.get("confidence")
            }
        }
        
        self.log_step("COMPLETE", {
            "total_time_ms": total_time * 1000,
            "answer_length": len(final_answer)
        })
        
        return result
    
    def get_trace_summary(self) -> str:
        """Get a human-readable trace summary."""
        summary = "\n" + "="*80 + "\n"
        summary += "PIPELINE TRACE SUMMARY\n"
        summary += "="*80 + "\n\n"
        
        for entry in self.trace_log:
            summary += f"[{entry['step']}]\n"
            summary += f"Time: {entry.get('timestamp', 'N/A')}\n"
            summary += f"Data: {json.dumps(entry['data'], indent=2, default=str)}\n\n"
        
        return summary


