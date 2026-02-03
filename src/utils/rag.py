"""
Retrieval-Augmented Generation (RAG) Utility

This module provides a skeleton for implementing RAG in radiology report generation.
RAG helps reduce hallucinations by retrieving similar historical cases (image-report pairs)
and providing them as context to the model.
"""

from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class RetrievalSystem:
    """
    Retrieval system for retrieving similar radiology cases.
    
    Student Task: implement the retrieve method to find similar images/reports.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path
        self.index = None
        # TODO: Initialize your vector database or similarity index here (e.g. FAISS)
        
    def build_index(self, images: List[str], reports: List[str]):
        """
        Build the retrieval index from a dataset.
        
        Args:
            images: List of image paths
            reports: List of corresponding radiology reports
        """
        logger.info("Building retrieval index...")
        # TODO:
        # 1. Compute embeddings for images (using CLIP, ResNet, or the model's visual encoder)
        # 2. Store embeddings in a vector database (FAISS, ChromaDB, etc.)
        pass
        
    def retrieve(self, query_image: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar cases for a given query image.
        
        Args:
            query_image: Path to the query image
            k: Number of similar cases to retrieve
            
        Returns:
            List of dictionaries containing 'report' and 'similarity_score'
        """
        # TODO:
        # 1. Compute embedding for query_image
        # 2. Search index for k nearest neighbors
        # 3. Return the associated reports
        
        logger.warning("RAG retrieve method not implemented returning empty list.")
        return [
            {"report": "Previous patient with similar consolidation in right upper lobe...", "score": 0.95},
            {"report": "No acute cardiopulmonary process.", "score": 0.82}
        ] if k > 0 else []

def format_rag_prompt(base_prompt: str, retrieved_cases: List[Dict[str, Any]]) -> str:
    """
    Format the prompt by appending retrieved cases as context.
    """
    context_str = "\n\nSimilar Cases Context:\n"
    for i, case in enumerate(retrieved_cases):
        context_str += f"Case {i+1}: {case['report']}\n"
        
    return base_prompt + context_str
