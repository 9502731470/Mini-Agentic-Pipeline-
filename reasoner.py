"""Reasoner component using LLM for decision-making and reasoning."""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import Dict, List, Optional
from config import LLM_MODEL, DEVICE, HF_TOKEN, USE_QUANTIZATION, USE_FALLBACK_MODE
import json
import re
from pathlib import Path

class Reasoner:
    """LLM-based reasoner that decides actions and processes information."""
    
    def __init__(self):
        # Determine device
        if DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = DEVICE
        
        self.model = None
        self.tokenizer = None
        self.use_fallback = USE_FALLBACK_MODE
        self.prompt_dir = Path(__file__).resolve().parent / "prompts"
        
        try:
            print(f"Loading LLM model: {LLM_MODEL} on {self.device}")
            
            # Configure quantization if requested
            quantization_config = None
            if USE_QUANTIZATION and USE_QUANTIZATION.lower() in ["4bit", "8bit"]:
                if USE_QUANTIZATION.lower() == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    print("Using 4-bit quantization (reduces model size by ~75%)")
                elif USE_QUANTIZATION.lower() == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    print("Using 8-bit quantization (reduces model size by ~50%)")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if configured
            model_kwargs = {
                "token": HF_TOKEN,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            else:
                model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                **model_kwargs
            )
            
            if not quantization_config and self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Failed to load LLM model: {str(e)}")
            if USE_FALLBACK_MODE:
                print("Using fallback rule-based reasoning mode")
                self.use_fallback = True
            else:
                raise Exception(f"Could not load LLM model. Set USE_FALLBACK_MODE=true to use simple reasoning. Error: {str(e)}")
        
        self.prompt_version = "v1"

    def _load_prompt_file(self, filename: str) -> str:
        """Load a prompt template from the prompts directory."""
        path = self.prompt_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    
    def get_prompt_template(self, version: str = None) -> str:
        """Get prompt template for reasoning.
        
        Args:
            version: Prompt version (v1, v2, etc.)
        """
        version = version or self.prompt_version
        
        if version == "v1":
            return self._load_prompt_file("reasoner_prompt_v1.txt")
        elif version == "v2":
            return self._load_prompt_file("reasoner_prompt_v2.txt")
        
        return self.get_prompt_template("v1")
    
    def reason(self, user_query: str, retrieved_context: List[Dict]) -> Dict:
        """Reason about the query and decide next action.
        
        Args:
            user_query: User's question
            retrieved_context: Retrieved documents from KB
            
        Returns:
            Dict with reasoning, tool decision, etc.
        """
        # Format context
        context_str = "\n\n".join([
            f"Doc {i+1} (ID: {doc.get('id', 'unknown')}):\n{doc.get('content', '')}"
            for i, doc in enumerate(retrieved_context)
        ])
        
        if not context_str:
            context_str = "No relevant context found in knowledge base."
        
        # Get prompt
        prompt_template = self.get_prompt_template()
        prompt = prompt_template
        prompt = prompt.replace("<<KB>>", context_str)
        prompt = prompt.replace("<<KB CONTENT>>", context_str)
        prompt = prompt.replace("<<USER_QUERY>>", user_query)
        prompt = prompt.replace("{retrieved_context}", context_str)
        prompt = prompt.replace("{user_query}", user_query)
        
        # Format prompt for the model
        system_message = "You are a helpful AI assistant that reasons about queries and decides when to use tools. Always respond with valid JSON."
        
        # Format prompt based on model type
        if "phi" in LLM_MODEL.lower() or "mistral" in LLM_MODEL.lower() or "llama" in LLM_MODEL.lower():
            # Chat template format
            formatted_prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple format
            formatted_prompt = f"{system_message}\n\n{prompt}\n\nResponse (JSON only):"
        
        # Call LLM
        try:
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Try to parse the whole response
                result = json.loads(response_text.strip())
            
            # Ensure all required fields
            result.setdefault("reasoning", "")
            result.setdefault("needs_tool", False)
            result.setdefault("tool_name", "none")
            result.setdefault("tool_query", "")
            result.setdefault("confidence", "medium")
            
            return result
            
        except Exception as e:
            return {
                "reasoning": f"Error in reasoning: {str(e)}",
                "needs_tool": False,
                "tool_name": "none",
                "tool_query": "",
                "confidence": "low",
                "error": str(e)
            }
    
    def synthesize_answer(self, user_query: str, retrieved_context: List[Dict], 
                         tool_results: Optional[Dict] = None) -> str:
        """Synthesize final answer from all available information.
        
        Args:
            user_query: Original user query
            retrieved_context: Retrieved KB documents
            tool_results: Results from tool execution (if any)
            
        Returns:
            Final answer string
        """
        # Use fallback if LLM not available
        if self.use_fallback or self.model is None:
            return self._fallback_synthesize(user_query, retrieved_context, tool_results)
        
        context_str = "\n\n".join([
            f"Doc {i+1}:\n{doc.get('content', '')}"
            for i, doc in enumerate(retrieved_context)
        ])
        
        tool_info = ""
        if tool_results:
            tool_info = f"\n\nAdditional information from {tool_results.get('tool', 'tool')}:\n{tool_results.get('result', '')}"
        
        prompt = f"""Based on the following information, provide a comprehensive answer to the user's query.

Knowledge Base Context:
{context_str}
{tool_info}

User Query: {user_query}

Provide a clear, well-structured answer that:
1. Directly addresses the query
2. Cites sources when relevant
3. Acknowledges any limitations or uncertainties
4. Is concise but complete

Answer:"""
        
        try:
            system_message = "You are a helpful assistant that provides clear, accurate answers based on available information."
            
            # Format prompt based on model type
            if "phi" in LLM_MODEL.lower() or "mistral" in LLM_MODEL.lower() or "llama" in LLM_MODEL.lower():
                formatted_prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = f"{system_message}\n\n{prompt}\n\nAnswer:"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return answer.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _fallback_synthesize(self, user_query: str, retrieved_context: List[Dict], 
                             tool_results: Optional[Dict] = None) -> str:
        """Simple answer synthesis when LLM is not available."""
        answer_parts = []
        
        # Add KB context
        if retrieved_context:
            answer_parts.append("Based on the knowledge base:")
            for i, doc in enumerate(retrieved_context[:3], 1):  # Use top 3 docs
                content = doc.get('content', '')[:500]  # Limit length
                answer_parts.append(f"\n{i}. {content}")
        
        # Add tool results
        if tool_results and tool_results.get('result'):
            answer_parts.append(f"\n\nAdditional information:\n{tool_results.get('result', '')[:500]}")
        
        if not answer_parts:
            return f"I found some information about '{user_query}', but I'm using a simplified mode. For better answers, please ensure the LLM model is properly loaded."
        
        return "\n".join(answer_parts)
