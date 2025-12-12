"""Configuration settings for the Mini Agentic Pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Configuration
# Optional: Set HF_TOKEN if using gated models
HF_TOKEN = os.getenv("HF_TOKEN", None)

# LLM Configuration - Using Hugging Face transformers
# Options (sorted by size):
# - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (very small, ~2GB, recommended for low-resource systems)
# - "microsoft/Phi-3-mini-4k-instruct" (small, fast, ~7GB)
# - "mistralai/Mistral-7B-Instruct-v0.2" (better quality, ~14GB)
# - "meta-llama/Llama-2-7b-chat-hf" (good quality, ~14GB)
LLM_MODEL = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Use quantization to reduce memory (4-bit or 8-bit)
# Options: None, "4bit", "8bit"
# 4-bit reduces model size by ~75%, 8-bit by ~50%
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "4bit")  # Default to 4-bit for low-resource systems

# Fallback mode: Use simple rule-based reasoning (avoids loading LLM)
# Default true so the pipeline can run without downloading large models.
USE_FALLBACK_MODE = os.getenv("USE_FALLBACK_MODE", "true").lower() == "true"

# Embedding Model - Using sentence-transformers
# Options:
# - "all-MiniLM-L6-v2" (fast, 384 dims)
# - "all-mpnet-base-v2" (better quality, 768 dims)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Device configuration
DEVICE = os.getenv("DEVICE", "auto")  # "auto", "cuda", "cpu"

# Tavily API (Web Search) - Optional
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Vector Store Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"

# Knowledge Base Path (matches bundled docs directory)
# NOTE: repo stores docs under knowledge_base_docs/, not knowledge_base/
KB_PATH = "./knowledge_base_docs"

# Tool Configuration
MAX_SEARCH_RESULTS = 5
MAX_RETRIEVAL_DOCS = 5

# Logging
LOG_LEVEL = "INFO"
