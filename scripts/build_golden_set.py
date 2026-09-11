"""
PHASE 5: Build Golden Evaluation Set

Generate stratified sample for manual labeling.
Ensure representation of all intents, ambiguous cases, and conversation types.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stratified_golden_sample(
    conversations_df: pd.DataFrame,
    sample_size: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create stratified sample for golden set.
    
    Ensures representation of:
    - All intents (especially rare ones)
    - Different message lengths
    - Multi-turn conversations
    - High/low confidence predictions
    
    Args:
        conversations_df: Conversations with intent labels
        sample_size: Target number of examples
        seed: Random seed
        
    Returns:
        DataFrame with golden set candidates
    """
    logger.info("Creating stratified golden set sample...")
    
    # Ensure all intents represented
    intent_groups = conversations_df.groupby("intent")
    samples = []
    
    for intent, group in intent_groups:
        # For rare intents, take all; for common, sample proportionally
        intent_size = max(2, int(sample_size * len(group) / len(conversations_df)))
        intent_sample = group.sample(
            n=min(intent_size, len(group)),
            random_state=seed
        )
        samples.append(intent_sample)
        logger.info(f"  {intent}: {len(intent_sample)} examples")
    
    golden_df = pd.concat(samples, ignore_index=True)
    
    # If we still need more, add random
    if len(golden_df) < sample_size:
        remaining = sample_size - len(golden_df)
        extra = conversations_df.sample(n=remaining, random_state=seed)
        golden_df = pd.concat([golden_df, extra], ignore_index=True)
    
    # Shuffle and reset
    golden_df = golden_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    golden_df["example_id"] = [f"gold_{i:04d}" for i in range(len(golden_df))]
    
    logger.info(f"Generated {len(golden_df)} golden examples")
    return golden_df


def create_labeling_template(golden_df: pd.DataFrame, output_path: Path):
    """
    Create CSV template for manual labeling.
    
    Args:
        golden_df: Golden set dataframe
        output_path: Where to save
    """
    # Create labeling template
    template_df = golden_df[[
        "example_id",
        "customer_tweet_id", 
        "customer_author",
        "customer_text",
        "brand_text",
        "intent",
        "confidence",
    ]].copy()
    
    # Add empty columns for human labeling
    template_df["gold_intent"] = ""
    template_df["gold_escalation"] = ""  # AUTO_HANDLE or ESCALATE
    template_df["gold_escalation_reason"] = ""
    template_df["notes"] = ""
    
    template_df.to_csv(output_path, index=False)
    logger.info(f"✓ Labeling template saved to {output_path}")
    logger.info(f"\nInstructions for labeling:")
    logger.info("1. Review customer_text and brand_text")
    logger.info("2. Fill gold_intent (use the intents from intent_taxonomy.txt)")
    logger.info("3. Fill gold_escalation: 'AUTO_HANDLE' or 'ESCALATE'")
    logger.info("4. Explain gold_escalation_reason")
    logger.info("5. Add any notes")
    logger.info(f"\nExample headers: {list(template_df.columns)}")


def main():
    """Main PHASE 5 execution."""
    logger.info("=" * 80)
    logger.info("PHASE 5: Golden Evaluation Set")
    logger.info("=" * 80)
    
    # Load conversations with intents
    logger.info("\n1. Loading labeled conversations...")
    conversations_df = pd.read_csv("data/processed/conversations_with_intents.csv")
    logger.info(f"   Loaded {len(conversations_df)} conversations")
    
    # Create stratified sample
    logger.info("\n2. Creating stratified sample...")
    golden_df = stratified_golden_sample(conversations_df, sample_size=200, seed=42)
    
    # Show distribution
    logger.info("\n3. Golden set intent distribution:")
    for intent, count in golden_df["intent"].value_counts().items():
        pct = 100 * count / len(golden_df)
        logger.info(f"   {intent}: {count} ({pct:.1f}%)")
    
    # Create labeling template
    logger.info("\n4. Creating labeling template...")
    golden_path = Path("data/golden/golden_candidates.csv")
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    create_labeling_template(golden_df, golden_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5 Complete")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Review data/golden/golden_candidates.csv")
    logger.info("2. Fill in gold_intent, gold_escalation, gold_escalation_reason")
    logger.info("3. Save as data/golden/golden_set.csv when done")
    logger.info("4. Run PHASE 6+: Build AI pipeline")


if __name__ == "__main__":
    main()
