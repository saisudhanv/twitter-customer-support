"""
PHASE 2: Dataset Inspection and Brand Selection

This script:
1. Downloads the customer support dataset
2. Analyzes brand candidates
3. Selects ONE brand for the project
4. Generates preprocessing rules
5. Reports data quality metrics
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data_loader import get_dataset_statistics, inspect_brands, load_raw_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_dataset():
    """Main analysis function."""
    config = get_config()
    
    logger.info("=" * 80)
    logger.info("PHASE 2: Dataset Inspection and Brand Selection")
    logger.info("=" * 80)
    
    # Load dataset
    logger.info("\n1. Loading dataset...")
    try:
        df = load_raw_dataset()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Make sure kagglehub is installed: pip install kagglehub")
        return
    
    # Get basic statistics
    logger.info("\n2. Dataset overview:")
    stats = get_dataset_statistics(df)
    logger.info(f"   Total records: {stats['total_records']:,}")
    logger.info(f"   Unique brands: {stats['unique_brands']}")
    logger.info(f"   Unique customers: {stats['unique_customers']:,}")
    logger.info(f"   Unique tweets: {stats['unique_tweets']:,}")
    logger.info(f"   Brand tweets (outbound): {stats['brand_tweets']:,}")
    logger.info(f"   Customer tweets (inbound): {stats['customer_tweets']:,}")
    logger.info(f"   Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    
    # Analyze brands
    logger.info("\n3. Analyzing brand candidates...")
    brand_candidates = inspect_brands(df, min_conversations=100)
    
    logger.info("\nTop 10 brands by tweet count:")
    print("\n" + brand_candidates.head(10)[["brand_name", "total_brand_tweets", "total_customer_messages"]].to_string())
    
    # Save candidates report
    report_path = Path("data/processed/brand_candidates.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    brand_candidates.to_csv(report_path, index=False)
    logger.info(f"\nBrand candidates saved to {report_path}")
    
    # Select brand
    logger.info("\n4. Selecting brand...")
    # For now, select the largest brand
    selected_brand = brand_candidates.iloc[0]["brand_name"]
    logger.info(f"Selected brand: {selected_brand}")
    
    # Filter to selected brand
    # Both inbound (customer) and outbound (brand) messages
    brand_df = df[df["author_id"] == selected_brand].copy()
    logger.info(f"   Records for selected brand: {len(brand_df):,}")
    
    # Analyze selected brand in detail
    logger.info("\n5. Detailed analysis of selected brand:")
    logger.info(f"   Date range: {brand_df['created_at'].min()} to {brand_df['created_at'].max()}")
    logger.info(f"   Unique conversation partners: {brand_df[brand_df['inbound']]['author_id'].nunique():,}")
    logger.info(f"   Avg text length: {brand_df['text'].str.len().mean():.0f} chars")
    logger.info(f"   Missing text: {brand_df['text'].isnull().sum()}")
    logger.info(f"   Brand tweets (outbound): {(~brand_df['inbound']).sum():,}")
    logger.info(f"   Customer tweets (inbound): {brand_df['inbound'].sum():,}")
    
    # Save selected brand data
    brand_data_path = Path("data/processed/selected_brand_raw.csv")
    brand_df.to_csv(brand_data_path, index=False)
    logger.info(f"\n   Selected brand data saved to {brand_data_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 Complete")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Review brand selection in data/processed/brand_candidates.csv")
    logger.info("2. Run PHASE 3: Data preprocessing")
    logger.info("   python -m scripts.prepare_data")


if __name__ == "__main__":
    analyze_dataset()
