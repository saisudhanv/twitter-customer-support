"""
Data loader for customer support dataset.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_dataset(cache_dir: Optional[Path] = None, force_download: bool = False) -> pd.DataFrame:
    """
    Load customer support dataset from Kaggle.
    
    Downloads the dataset using kagglehub if not cached locally.
    
    Args:
        cache_dir: Directory to cache dataset (uses kagglehub default if None)
        force_download: Force re-download even if cached
        
    Returns:
        DataFrame with raw dataset
        
    Raises:
        ImportError: If kagglehub is not installed
        Exception: If download fails
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError("kagglehub is required. Install with: pip install kagglehub")

    logger.info("Loading customer support dataset from Kaggle...")
    
    try:
        # Download and cache the dataset
        dataset_path = kagglehub.dataset_download(
            "thoughtvector/customer-support-on-twitter",
            path=cache_dir
        )
        
        csv_path = Path(dataset_path) / "twcs" / "twcs.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")
        
        logger.info(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        
        logger.info(f"Loaded {len(df):,} records")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def inspect_brands(df: pd.DataFrame, min_conversations: int = 100) -> pd.DataFrame:
    """
    Analyze brands in the dataset and return candidates.
    
    The dataset has 'inbound' field: True=customer, False=brand/company
    'author_id' field contains the company name.
    
    Args:
        df: Raw dataset DataFrame
        min_conversations: Minimum number of conversations to consider
        
    Returns:
        DataFrame with brand statistics
    """
    logger.info("Analyzing brands in dataset...")
    
    # Separate inbound (customer) vs outbound (brand) messages
    brand_messages = df[~df["inbound"]].copy()  # False = brand messages
    customer_messages = df[df["inbound"]].copy()  # True = customer messages
    
    # Group by author_id (company/brand name)
    brand_stats = brand_messages.groupby("author_id").agg({
        "tweet_id": "count",
        "text": lambda x: x.str.len().mean(),  # Avg text length
        "created_at": ["min", "max"],
    }).reset_index()
    
    brand_stats.columns = ["brand_name", "total_brand_tweets", "avg_text_len", "date_min", "date_max"]
    
    # Count customer messages per brand
    customer_per_brand = customer_messages.groupby("author_id").size().reset_index(name="total_customer_messages")
    brand_stats = brand_stats.merge(customer_per_brand, left_on="brand_name", right_on="author_id", how="left")
    brand_stats = brand_stats.drop("author_id", axis=1, errors="ignore")
    
    # Filter by minimum conversations
    brand_stats = brand_stats[brand_stats["total_brand_tweets"] >= min_conversations].sort_values(
        "total_brand_tweets", ascending=False
    )
    
    logger.info(f"Found {len(brand_stats)} brands with >= {min_conversations} brand tweets")
    
    return brand_stats


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Get comprehensive statistics about the dataset.
    
    Args:
        df: Raw dataset DataFrame
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_records": len(df),
        "unique_brands": df[~df["inbound"]]["author_id"].nunique(),  # Brands
        "unique_customers": df[df["inbound"]]["author_id"].nunique(),  # Customers (approx)
        "unique_tweets": df["tweet_id"].nunique(),
        "brand_tweets": (~df["inbound"]).sum(),
        "customer_tweets": df["inbound"].sum(),
        "date_range": (df["created_at"].min(), df["created_at"].max()),
        "missing_values": df.isnull().sum().to_dict(),
        "columns": df.columns.tolist(),
    }
    
    return stats
