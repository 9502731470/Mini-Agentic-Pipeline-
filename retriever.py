"""Retriever component for knowledge base search using vector embeddings."""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
import os
from typing import List, Dict
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    DEVICE,
    MAX_RETRIEVAL_DOCS
)

class Retriever:
    """Retrieves relevant documents from knowledge base using vector similarity."""
    
    def __init__(self):
        # Determine device
        if DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = DEVICE
        
        print(f"Loading embedding model: {EMBEDDING_MODEL} on {self.device}")
        # Load sentence transformer model
        # SentenceTransformer handles both "sentence-transformers/model" and "model" formats
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
        
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text using Hugging Face sentence-transformers."""
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'id', 'content', and optionally 'metadata'
        """
        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        # Generate embeddings
        embeddings = [self.get_embedding(content) for content in contents]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
    
    def search(self, query: str, n_results: int = MAX_RETRIEVAL_DOCS) -> List[Dict]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant documents with scores
        """
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        retrieved_docs = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                retrieved_docs.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return retrieved_docs
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(COLLECTION_NAME)
        except:
            self.collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
