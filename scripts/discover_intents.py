"""
PHASE 4: Intent Discovery and Taxonomy Creation

Analyze customer messages and create a domain-specific intent taxonomy.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intent_taxonomy import (
    analyze_intent_patterns,
    create_default_taxonomy,
    label_conversations,
    print_taxonomy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main PHASE 4 execution."""
    logger.info("=" * 80)
    logger.info("PHASE 4: Intent Discovery and Taxonomy")
    logger.info("=" * 80)
    
    # Load conversations
    logger.info("\n1. Loading conversations...")
    conversations_df = pd.read_csv("data/processed/conversations.csv")
    logger.info(f"   Loaded {len(conversations_df)} conversations")
    
    # Analyze patterns
    logger.info("\n2. Analyzing intent patterns...")
    patterns = analyze_intent_patterns(conversations_df, sample_size=200)
    
    # Create taxonomy
    logger.info("\n3. Creating intent taxonomy...")
    taxonomy = create_default_taxonomy()
    print_taxonomy(taxonomy)
    
    # Label conversations
    logger.info("\n4. Labeling conversations with intents...")
    conversations_df = label_conversations(conversations_df, taxonomy)
    
    # Save
    output_path = Path("data/processed/conversations_with_intents.csv")
    conversations_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Labeled conversations saved to {output_path}")
    
    # Save taxonomy
    taxonomy_path = Path("data/processed/intent_taxonomy.txt")
    with open(taxonomy_path, "w") as f:
        f.write("INTENT TAXONOMY\n")
        f.write("=" * 80 + "\n\n")
        for intent_id, intent_def in taxonomy.items():
            f.write(f"{intent_id}:\n")
            f.write(f"  Name: {intent_def['name']}\n")
            f.write(f"  Description: {intent_def['description']}\n")
            f.write(f"  Keywords: {', '.join(intent_def.get('keywords', []))}\n\n")
    
    logger.info(f"✓ Taxonomy saved to {taxonomy_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4 Complete")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. PHASE 5: Build golden evaluation set")
    logger.info("   python -m scripts.build_golden_set")


if __name__ == "__main__":
    main()
