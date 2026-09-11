"""
PHASE 3: Conversation Construction

Convert raw tweets into structured customer-brand conversation pairs.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.conversation_builder import prepare_conversation_data
from src.data_loader import load_raw_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main PHASE 3 execution."""
    config = get_config()
    
    # Load dataset
    logger.info("Loading raw dataset...")
    df = load_raw_dataset()
    
    # Prepare conversations
    conversations_df, stats = prepare_conversation_data(
        df,
        brand_name="AmazonHelp",
        sample_size=config.data_sample_size,
        seed=config.random_seed,
    )
    
    # Save
    output_path = Path("data/processed/conversations.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conversations_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Conversations saved to {output_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3 Complete")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. PHASE 4: Intent discovery")
    logger.info("   python -m scripts.discover_intents")


if __name__ == "__main__":
    main()
