"""
Retrieval system for finding similar historical support cases.

Simple implementation using TF-IDF similarity.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SimpleRetrieval:
    """Retrieve similar cases using TF-IDF similarity."""
    
    def __init__(self, max_features: int = 500):
        """
        Initialize retrieval system.
        
        Args:
            max_features: Max TF-IDF features
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True)
        self.cases = []
        self.vectors = None
        self.is_fitted = False
    
    def index(self, cases: pd.DataFrame):
        """
        Index conversation cases for retrieval.
        
        Args:
            cases: DataFrame with 'customer_text' and 'brand_text' columns
        """
        logger.info(f"Indexing {len(cases)} cases...")
        
        self.cases = cases.to_dict('records')
        
        # Vectorize customer messages
        customer_texts = cases["customer_text"].fillna("").tolist()
        self.vectors = self.vectorizer.fit_transform(customer_texts)
        self.is_fitted = True
        
        logger.info(f"✓ Indexed {len(self.cases)} cases")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        exclude_ids: list[str] | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k similar cases.
        
        Args:
            query: Customer message (query)
            top_k: Number of results to return
            exclude_ids: Case IDs to exclude (for leakage prevention)
            
        Returns:
            List of retrieved cases with similarity scores
        """
        if not self.is_fitted:
            logger.warning("Retrieval system not indexed yet")
            return []
        
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Compute similarity
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(-similarities)[:top_k * 2]  # Get extra for filtering
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            case = self.cases[idx]
            
            # Skip excluded IDs
            if exclude_ids and case.get("customer_tweet_id") in exclude_ids:
                continue
            
            results.append({
                "customer_message": case.get("customer_text", ""),
                "brand_response": case.get("brand_text", ""),
                "similarity": float(similarities[idx]),
                "intent": case.get("intent", "unknown"),
            })
        
        return results
