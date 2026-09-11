"""
Baseline classifiers for evaluation.

1. Trivial baseline: Always predict majority class / always escalate
2. Simple baseline: TF-IDF + Logistic Regression
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class TrivialBaseline:
    """Trivial baseline for intent classification."""
    
    def __init__(self, majority_intent: str = "other"):
        """
        Initialize trivial baseline.
        
        Args:
            majority_intent: Intent to always predict
        """
        self.majority_intent = majority_intent
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict intent (always returns majority).
        
        Args:
            text: Customer message
            
        Returns:
            Tuple of (intent, confidence)
        """
        return self.majority_intent, 0.5


class SimpleBaseline:
    """Simple ML baseline using TF-IDF + Logistic Regression."""
    
    def __init__(self, max_features: int = 1000):
        """
        Initialize baseline.
        
        Args:
            max_features: Max TF-IDF features
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True)
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def fit(self, texts: list, labels: list):
        """
        Fit classifier.
        
        Args:
            texts: List of customer messages
            labels: List of intents
        """
        logger.info("Fitting TF-IDF + Logistic Regression baseline...")
        
        # Vectorize
        X = self.vectorizer.fit_transform(texts)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Fit classifier
        self.classifier.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"✓ Fitted on {len(texts)} examples")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict intent.
        
        Args:
            text: Customer message
            
        Returns:
            Tuple of (intent, confidence)
        """
        if not self.is_fitted:
            return "other", 0.0
        
        X = self.vectorizer.transform([text])
        pred_idx = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(proba))
        intent = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return intent, confidence


class SimpleEscalationPolicy:
    """Simple rule-based escalation."""
    
    # High-risk keywords that should escalate
    ESCALATE_KEYWORDS = {
        "account", "password", "security", "fraud", "urgent", "angry",
        "court", "lawsuit", "lawyer", "refund", "money", "cancel",
        "delete", "damaged", "broken", "not working"
    }
    
    # Low-confidence intents should escalate
    ESCALATE_INTENTS = {"other", "payment_issue"}
    
    @staticmethod
    def decide(
        intent: str,
        confidence: float,
        customer_text: str,
        min_confidence: float = 0.6,
    ) -> Tuple[str, str]:
        """
        Decide whether to escalate.
        
        Args:
            intent: Predicted intent
            confidence: Intent confidence
            customer_text: Customer message
            min_confidence: Confidence threshold for auto-handle
            
        Returns:
            Tuple of (decision, reason) where decision is "AUTO_HANDLE" or "ESCALATE"
        """
        # Low confidence → escalate
        if confidence < min_confidence:
            return "ESCALATE", f"Low confidence ({confidence:.2f})"
        
        # Risky intent → escalate
        if intent in SimpleEscalationPolicy.ESCALATE_INTENTS:
            return "ESCALATE", f"Intent '{intent}' requires human review"
        
        # Escalate keywords → escalate
        text_lower = customer_text.lower()
        for keyword in SimpleEscalationPolicy.ESCALATE_KEYWORDS:
            if keyword in text_lower:
                return "ESCALATE", f"Escalation keyword detected: '{keyword}'"
        
        # Otherwise auto-handle
        return "AUTO_HANDLE", f"Confident ({confidence:.2f}), low-risk"
