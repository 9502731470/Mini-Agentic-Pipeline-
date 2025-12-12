# Mini Agentic Pipeline

A complete implementation of an AI-driven agentic workflow that retrieves context from a knowledge base, reasons about queries using an LLM, and executes actions via tools to produce comprehensive answers.

##  Objective

This pipeline demonstrates a working agentic AI system that:
- **Retrieves** relevant context from a knowledge base (16 documents)
- **Reasons** using an LLM to decide the next step
- **Acts** via tools (web search, API calls, CSV lookup)
- **Produces** final answers with clear step-by-step traces

##  Architecture

The system consists of four main components:

### 1. **Retriever** (`retriever.py`)
- Uses ChromaDB as vector store
- Generates embeddings using Hugging Face sentence-transformers (default: `all-MiniLM-L6-v2`)
- Performs semantic search on knowledge base
- Returns top-k relevant documents
- Runs locally, no API costs

### 2. **Reasoner** (`reasoner.py`)
- Uses Hugging Face transformer models (default: `Phi-3-mini-4k-instruct`)
- Supports multiple models: Phi-3, Mistral, Llama-2
- Implements modular prompt templates (v1, v2)
- Analyzes queries and retrieved context
- Decides whether to use tools or answer from KB alone
- Synthesizes final answers from all available information
- Runs locally, no API costs

### 3. **Actor** (`actor.py`)
- Executes three types of tools:
  - **Web Search**: Tavily API for current information
  - **API Calls**: REST API integration
  - **CSV Lookup**: Local CSV file queries (e.g., product prices)

### 4. **Orchestrator** (`orchestrator.py`)
- Coordinates all components
- Manages shared state/context
- Logs each step with timestamps
- Produces comprehensive traces

##  Project Structure

```
Mini_Agentic_Pipeline/
├── knowledge_base/          # 16 AI-related documents
│   ├── doc1_ai_overview.txt
│   ├── doc2_machine_learning.txt
│   ├── ...
│   └── doc16_future_ai.txt
├── data/
│   └── prices.csv          # Sample product catalog
├── chroma_db/              # Vector database (created on first run)
├── config.py               # Configuration settings
├── retriever.py            # Retrieval component
├── reasoner.py             # LLM reasoning component
├── actor.py                # Tool execution component
├── orchestrator.py         # Main controller
├── initialize_kb.py        # Initialize knowledge base
├── main.py                 # Interactive CLI
├── evaluate.py             # Evaluation script
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

##  Setup Instructions

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB+ recommended for larger models)
- (Optional) NVIDIA GPU with CUDA support for faster inference
- (Optional) Hugging Face token (for gated models like Llama-2)
- (Optional) Tavily API key for web search

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (optional):**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` if needed (defaults work without it):
   ```
   # Optional: Hugging Face token for gated models
   HF_TOKEN=your_huggingface_token_here
   
   # Optional: Use different models
   LLM_MODEL=microsoft/Phi-3-mini-4k-instruct
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   
   # Optional: Device (auto/cuda/cpu)
   DEVICE=auto
   
   # Optional: Tavily API key for web search
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
   
   **Note**: Models will be downloaded automatically on first run. This may take several minutes depending on your internet connection.

4. **Initialize the knowledge base:**
   ```bash
   python initialize_kb.py
   ```
   This will:
   - **Download the embedding model automatically** (~80MB, first time only)
   - Load all 16 documents from `knowledge_base/`
   - Generate embeddings
   - Store them in ChromaDB
   
   **Note**: Models are downloaded automatically from Hugging Face Hub on first use. They are cached locally, so subsequent runs are faster.

## 💻 Usage

### Quick Start

Test the pipeline with a sample query:

```bash
python quick_start.py
```

This runs a simple example to verify everything is working.

**Note**: On first run, this will **automatically download the LLM model** (~7GB for Phi-3-mini). This may take several minutes depending on your internet connection. The model is cached locally after download.

### Interactive Mode

Run the pipeline interactively:

```bash
python main.py
```

Then enter queries when prompted. Type `quit` or `exit` to stop.

### Single Query Mode

Process a single query:

```bash
python main.py "What is artificial intelligence?"
```

### Evaluation

Run the evaluation script with 12 test queries:

```bash
python evaluate.py
```

This will:
- Process all test queries
- Measure latency for each component
- Generate a summary report
- Save detailed results to a JSON file

##  Knowledge Base

The knowledge base contains 16 documents covering:

1. AI Overview
2. Machine Learning Fundamentals
3. Neural Networks and Deep Learning
4. Natural Language Processing
5. Computer Vision
6. Reinforcement Learning
7. Transformer Models
8. Large Language Models
9. Agentic AI Systems
10. Retrieval-Augmented Generation (RAG)
11. Text Embeddings
12. Vector Databases
13. Prompt Engineering
14. AI Ethics
15. Real-World AI Applications
16. Future of AI

## 🛠️ Tools

### Web Search (Tavily)
- Searches the web for current information
- Used when KB lacks recent/current data
- Example: "What is the current price of Bitcoin?"

### CSV Lookup
- Queries local CSV files
- Used for structured data lookups
- Example: "What is the price of a wireless mouse?"
- Default file: `data/prices.csv`

### API Calls
- Makes REST API calls
- Can be extended for any API endpoint
- Example: Custom pricing API, weather API, etc.

##  Evaluation Results

The evaluation script tests 12 queries across different scenarios:

- **KB-only queries**: Can be answered from knowledge base
- **Web search queries**: Require current information
- **CSV lookup queries**: Need structured data lookup

Metrics tracked:
- Total latency per query
- Component-wise latency (retrieval, reasoning, tool, synthesis)
- Tool decision accuracy
- Number of retrieved documents
- Confidence levels

##  Example Queries

### KB-Only (No Tool Needed)
- "What is artificial intelligence?"
- "Explain how neural networks work"
- "How does RAG work?"
- "What are the ethical concerns with AI?"

### Requires Web Search
- "What are the latest developments in AI in 2024?"
- "What is the current price of Bitcoin?"
- "What is the weather today in San Francisco?"

### Requires CSV Lookup
- "What is the price of a wireless mouse?"
- "Find products under $50 in the catalog"
- "What products are available in the Electronics category?"

##  Design Decisions

### Vector Store: ChromaDB
- **Why**: Lightweight, embedded, Python-first
- **Alternative considered**: Pinecone (cloud), FAISS (local)
- **Trade-off**: ChromaDB balances ease of use with performance

### Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- **Why**: Fast, good quality, runs locally, no API costs
- **Alternative**: all-mpnet-base-v2 (better quality, slower)
- **Trade-off**: MiniLM provides good balance of speed and quality
- **Benefits**: No API costs, privacy (data stays local), fast inference

### LLM: Phi-3-mini-4k-instruct
- **Why**: Small model (~7GB), fast inference, good reasoning, runs locally
- **Alternatives**: 
  - Mistral-7B-Instruct (better quality, ~14GB)
  - Llama-2-7b-chat (good quality, ~14GB, requires HF token)
- **Trade-off**: Phi-3-mini balances quality, speed, and resource usage
- **Benefits**: No API costs, privacy, customizable, can run on CPU

### Prompt Templates
- **Why**: Modular prompts allow versioning and iteration
- **Implementation**: v1 (basic), v2 (enhanced reasoning)
- **Future**: Easy to add v3, v4, etc.

### Tool Selection
- **Web Search**: Tavily (fast, reliable, good results)
- **CSV Lookup**: Local file (no external dependency)
- **API Calls**: Generic REST client (extensible)

##  Logging and Tracing

The system logs every step:

1. **RETRIEVAL**: Query sent to retriever
2. **RETRIEVAL_COMPLETE**: Documents retrieved with metadata
3. **REASONING**: LLM reasoning initiated
4. **REASONING_COMPLETE**: Tool decision made
5. **TOOL_EXECUTION**: Tool called (if needed)
6. **TOOL_EXECUTION_COMPLETE**: Tool results received
7. **SYNTHESIS**: Final answer generation
8. **SYNTHESIS_COMPLETE**: Answer ready
9. **COMPLETE**: Full pipeline finished

Each log entry includes:
- Timestamp
- Step name
- Relevant data (documents, reasoning, tool results, etc.)

##  Known Limitations

1. **Model Download**: 
   - Models are downloaded automatically on first run (several GB)
   - Requires stable internet connection for initial setup
   - Models are cached locally after first download

2. **Hardware Requirements**:
   - Requires 8GB+ RAM (16GB+ recommended)
   - GPU recommended but not required (CPU works, slower)
   - First model load can take 1-2 minutes

3. **Model Limitations**:
   - Smaller models (Phi-3-mini) may have lower reasoning quality than larger models
   - JSON parsing from LLM responses may occasionally fail (error handling included)
   - Some models require Hugging Face token (Llama-2)

4. **Optional API Keys**: 
   - Tavily API key needed for web search (optional, but queries requiring web search will fail without it)
   - Hugging Face token only needed for gated models

3. **Tool Limitations**:
   - CSV lookup requires exact file path
   - Web search depends on Tavily API availability
   - API calls need proper endpoint configuration

4. **Knowledge Base**:
   - Limited to 16 documents (can be extended)
   - Static content (no automatic updates)
   - Domain-specific (AI topics)

5. **Error Handling**:
   - Basic error handling implemented
   - Some edge cases may not be fully covered
   - Tool failures are logged but may not always recover gracefully

##  Future Improvements

1. **Enhanced Tool Selection**:
   - Multi-tool reasoning (use multiple tools if needed)
   - Tool result validation
   - Retry logic for failed tools

2. **Better Prompting**:
   - Chain-of-thought reasoning
   - Few-shot examples
   - Prompt optimization based on results

3. **Knowledge Base**:
   - Automatic document updates
   - Support for more file formats (PDF, DOCX)
   - Hybrid search (vector + keyword)

4. **Evaluation**:
   - Automated answer quality scoring
   - A/B testing for prompts
   - Performance benchmarking

5. **User Interface**:
   - Web interface
   - API endpoint
   - Streamlit/Gradio demo

##  Dependencies

- `transformers`: Hugging Face transformer models
- `torch`: PyTorch for model inference
- `sentence-transformers`: Embedding models
- `chromadb`: Vector database
- `pandas`: CSV processing
- `requests`: HTTP requests for APIs
- `accelerate`: Model acceleration utilities
- `bitsandbytes`: Quantization support (optional, for memory efficiency)
- `tavily-python`: Web search (optional)
- `python-dotenv`: Environment variable management


##  Demo Video

[Link to demo video will be added here]

The demo video covers:
- Architecture explanation
- Code walkthrough
- Live demo on 3-4 queries
- Learnings and insights


##  Acknowledgments

- Hugging Face for transformer models and infrastructure
- Microsoft for Phi-3 models
- ChromaDB for vector database
- Tavily for web search API
- Sentence Transformers for embedding models

##  Model Selection Guide

### Recommended Models by Use Case:

**For Low-Resource Systems (Limited RAM/Disk):**
- LLM: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~2GB, or ~500MB with 4-bit quantization)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (~80MB)
- **Or use fallback mode**: Set `USE_FALLBACK_MODE=true` (no LLM needed, only ~80MB)

**For Fast Inference (CPU-friendly):**
- LLM: `microsoft/Phi-3-mini-4k-instruct` (~7GB, or ~1.75GB with 4-bit quantization)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (~80MB)

**For Better Quality (GPU recommended):**
- LLM: `mistralai/Mistral-7B-Instruct-v0.2` (~14GB, or ~3.5GB with 4-bit quantization)
- Embedding: `sentence-transformers/all-mpnet-base-v2` (~420MB)

**For Best Quality (requires HF token):**
- LLM: `meta-llama/Llama-2-7b-chat-hf` (~14GB, or ~3.5GB with 4-bit quantization)
- Embedding: `sentence-transformers/all-mpnet-base-v2`

### Changing Models:

Set in `.env` file (create from `.env.example`):
```bash
# Use smaller model for low-resource systems
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Enable quantization to reduce size by 75%
USE_QUANTIZATION=4bit

# Or use fallback mode (no LLM)
USE_FALLBACK_MODE=true
```

### Quantization Options:

- `USE_QUANTIZATION=4bit` - Reduces model size by ~75% (recommended for low-resource)
- `USE_QUANTIZATION=8bit` - Reduces model size by ~50%
- `USE_QUANTIZATION=None` - No quantization (full size)

### Fallback Mode:

If you cannot download any LLM model, set `USE_FALLBACK_MODE=true` in `.env`. This uses simple rule-based reasoning instead of LLM. See `LOW_RESOURCE_SETUP.md` for details.
