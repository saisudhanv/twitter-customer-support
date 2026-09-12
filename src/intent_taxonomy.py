"""
Intent discovery and taxonomy creation from conversation data.

Discovers natural intents from AmazonHelp customer support conversations
using keyword analysis and manual inspection rather than generic Banking77.
"""

import logging
from collections import Counter
from typing import dict, list, tuple

import pandas as pd

logger = logging.getLogger(__name__)


def extract_keywords(text: str, min_len: int = 3) -> list[str]:
    """
    Extract keywords from text (simplified).
    
    Args:
        text: Text to extract keywords from
        min_len: Minimum keyword length
        
    Returns:
        List of keywords
    """
    if pd.isna(text):
        return []
    
    # Simple lowercasing and splitting
    # In production, would use proper NLP
    words = str(text).lower().split()
    keywords = [w for w in words if len(w) >= min_len and not w.startswith("@")]
    return keywords



def analyze_intent_patterns(conversations_df: pd.DataFrame, sample_size: int = 100) -> dict[str, object]:
    """
    Analyze patterns in customer messages to discover intents.
    
    Looks for:
    - Common keywords
    - Message patterns
    - Problem types
    
    Args:
        conversations_df: DataFrame with customer_text column
        sample_size: Number of messages to analyze
        
    Returns:
        Dictionary with patterns
    """
    logger.info("Analyzing intent patterns...")
    
    # Sample messages
    sample = conversations_df.sample(min(sample_size, len(conversations_df)), random_state=42)
    
    # Extract all keywords
    all_keywords = []
    for text in sample["customer_text"]:
        keywords = extract_keywords(text)
        all_keywords.extend(keywords)
    
    # Get top keywords
    keyword_counts = Counter(all_keywords)
    top_keywords = keyword_counts.most_common(30)
    
    logger.info("\nTop 30 keywords in customer messages:")
    for keyword, count in top_keywords:
        logger.info(f"  {keyword}: {count}")
    
    return {
        "top_keywords": dict(top_keywords),
        "sample_size": len(sample),
    }


def create_default_taxonomy() -> dict[str, dict]:
    """
    Create a default intent taxonomy for AmazonHelp based on domain knowledge.
    
    For e-commerce customer support, typical intents include:
    - Order/Delivery issues
    - Returns/Refunds
    - Product Information
    - Account/Login
    - Payment Issues
    - Tracking
    - Quality/Defect
    - General Inquiry
    - Billing
    - Other
    
    Returns:
        Dictionary mapping intent_id to intent definition
    """
    taxonomy = {
        "order_tracking": {
            "name": "Order Tracking",
            "description": "Customer inquiring about order status, delivery, or tracking",
            "keywords": ["order", "tracking", "ship", "deliver", "when", "status", "where"],
        },
        "refund_return": {
            "name": "Refund/Return",
            "description": "Customer asking about returning a product or receiving a refund",
            "keywords": ["refund", "return", "money back", "cancel", "return shipping"],
        },
        "product_info": {
            "name": "Product Information",
            "description": "Customer asking questions about product features, compatibility, specs",
            "keywords": ["compatible", "feature", "how", "work", "compatible", "specs", "model"],
        },
        "account_access": {
            "name": "Account/Login Issue",
            "description": "Customer having trouble accessing their account or resetting password",
            "keywords": ["account", "login", "password", "access", "reset", "sign in", "username"],
        },
        "payment_issue": {
            "name": "Payment Problem",
            "description": "Customer reporting payment processing errors or card decline",
            "keywords": ["payment", "charge", "card", "declined", "billing", "debit"],
        },
        "quality_defect": {
            "name": "Quality/Defect",
            "description": "Customer reporting product defect, damage, or quality issue",
            "keywords": ["broken", "defect", "damage", "broken", "not working", "quality", "wrong"],
        },
        "general_inquiry": {
            "name": "General Inquiry",
            "description": "General question not fitting other categories",
            "keywords": ["question", "help", "what", "info", "assistance", "support"],
        },
        "other": {
            "name": "Other",
            "description": "Messages that don't fit clearly into other categories",
            "keywords": [],
        },
    }
    
    return taxonomy



def classify_by_keywords(text: str, taxonomy: dict[str, dict]) -> tuple[str, float]:
    """
    Classify text to an intent using keyword matching.
    
    Simple heuristic: count matching keywords and pick best match.
    
    Args:
        text: Text to classify
        taxonomy: Intent taxonomy
        
    Returns:
        Tuple of (intent_id, confidence)
    """
    if pd.isna(text):
        return "other", 0.0
    
    text_lower = str(text).lower()
    
    scores = {}
    for intent_id, intent_def in taxonomy.items():
        score = 0
        for keyword in intent_def.get("keywords", []):
            if keyword.lower() in text_lower:
                score += 1
        scores[intent_id] = score
    
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    
    # Convert to confidence (0-1)
    confidence = min(best_score / max(3, sum(1 for s in scores.values() if s > 0)), 1.0)
    if best_score == 0:
        confidence = 0.0
        best_intent = "other"
    
    return best_intent, confidence


def label_conversations(
    conversations_df: pd.DataFrame,
    taxonomy: dict[str, dict],
) -> pd.DataFrame:
    """
    Label conversations with intents.
    
    Args:
        conversations_df: DataFrame with customer_text
        taxonomy: Intent taxonomy
        
    Returns:
        DataFrame with added intent columns
    """
    logger.info("Labeling conversations with intents...")
    
    results = []
    for text in conversations_df["customer_text"]:
        intent, confidence = classify_by_keywords(text, taxonomy)
        results.append({"intent": intent, "confidence": confidence})
    
    result_df = pd.DataFrame(results)
    conversations_df = pd.concat([conversations_df.reset_index(drop=True), result_df], axis=1)
    
    # Show distribution
    logger.info("\nIntent distribution:")
    intent_counts = conversations_df["intent"].value_counts()
    for intent, count in intent_counts.items():
        pct = 100 * count / len(conversations_df)
        logger.info(f"  {intent}: {count} ({pct:.1f}%)")
    
    return conversations_df



def print_taxonomy(taxonomy: dict[str, dict]):
    """Print taxonomy in readable format."""
    logger.info("\n" + "=" * 80)
    logger.info("INTENT TAXONOMY")
    logger.info("=" * 80)
    
    for intent_id, intent_def in taxonomy.items():
        logger.info(f"\n{intent_id}:")
        logger.info(f"  Name: {intent_def['name']}")
        logger.info(f"  Description: {intent_def['description']}")
        logger.info(f"  Keywords: {', '.join(intent_def.get('keywords', []))}")
