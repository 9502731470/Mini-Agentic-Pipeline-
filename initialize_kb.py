"""Initialize the knowledge base with documents."""
import os
from retriever import Retriever
from config import KB_PATH

def load_documents():
    """Load all documents from the knowledge base directory."""
    documents = []
    
    if not os.path.exists(KB_PATH):
        print(f"Knowledge base directory not found: {KB_PATH}")
        return documents
    
    # Load all .txt files from knowledge base
    for filename in os.listdir(KB_PATH):
        if filename.endswith('.txt'):
            filepath = os.path.join(KB_PATH, filename)
            doc_id = filename.replace('.txt', '')
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append({
                'id': doc_id,
                'content': content,
                'metadata': {
                    'filename': filename,
                    'source': 'knowledge_base'
                }
            })
    
    return documents

def initialize_knowledge_base():
    """Initialize the vector store with knowledge base documents."""
    print("Initializing knowledge base...")
    
    retriever = Retriever()
    
    # Clear existing collection
    print("Clearing existing collection...")
    retriever.clear_collection()
    
    # Load documents
    print("Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("No documents found to load!")
        return
    
    print(f"Found {len(documents)} documents")
    
    # Add documents to vector store
    print("Adding documents to vector store...")
    retriever.add_documents(documents)
    
    print(f"Successfully initialized knowledge base with {len(documents)} documents!")
    print("\nDocuments loaded:")
    for doc in documents:
        print(f"  - {doc['id']}")

if __name__ == "__main__":
    initialize_knowledge_base()


