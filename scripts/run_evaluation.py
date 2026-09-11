"""
PHASE 6: Run Evaluation on Golden Set

Evaluate baselines and AI system on golden examples.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines import SimpleBaseline, SimpleEscalationPolicy, TrivialBaseline
from src.metrics import EscalationEvaluator, IntentEvaluator
from src.pipeline import SupportAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CANONICAL_INTENTS = {
    "order_tracking",
    "refund_return",
    "product_info",
    "account_access",
    "payment_issue",
    "quality_defect",
    "general_inquiry",
    "other",
}


def read_csv_with_encoding_fallback(path: str) -> pd.DataFrame:
    """Read user-edited CSV files saved with common Windows encodings."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("%s is not UTF-8; reading it with a single-byte fallback", path)
        return pd.read_csv(path, encoding="latin-1")


def canonicalize_intent(label: str) -> str:
    """Map human-friendly labels to the project's canonical intent IDs."""
    normalized = str(label).strip().lower().replace("-", " ").replace("/", " ")
    if normalized in CANONICAL_INTENTS:
        return normalized

    if any(term in normalized for term in ("refund", "return", "cancel", "cashback")):
        return "refund_return"
    if any(term in normalized for term in ("password", "account", "login", "content access")):
        return "account_access"
    if any(term in normalized for term in ("payment", "billing", "charge", "cash", "price")):
        return "payment_issue"
    if any(term in normalized for term in ("damage", "defect", "broken", "wrong item", "missing item", "quality")):
        return "quality_defect"
    if any(term in normalized for term in ("delivery", "shipping", "tracking", "package", "courier", "postal", "order")):
        return "order_tracking"
    if any(term in normalized for term in ("product", "device", "app", "feature", "availability", "promotion", "voucher")):
        return "product_info"
    if any(term in normalized for term in ("complaint", "compliment", "thanks", "gratitude", "support", "service", "request", "inquiry")):
        return "general_inquiry"
    return "other"


def validate_and_normalize_gold_labels(golden_df: pd.DataFrame) -> pd.DataFrame:
    """Validate manual labels and normalize granular intent names for scoring."""
    required_columns = {"gold_intent", "gold_escalation"}
    missing_columns = required_columns - set(golden_df.columns)
    if missing_columns:
        raise ValueError(f"Golden set is missing required columns: {sorted(missing_columns)}")

    if golden_df["gold_intent"].isna().any() or golden_df["gold_intent"].astype(str).str.strip().eq("").any():
        raise ValueError("Every golden example must have a non-empty gold_intent.")
    if golden_df["gold_escalation"].isna().any() or golden_df["gold_escalation"].astype(str).str.strip().eq("").any():
        raise ValueError("Every golden example must have a non-empty gold_escalation.")

    escalation_values = set(golden_df["gold_escalation"].astype(str).str.strip().str.upper())
    invalid_escalations = escalation_values - {"AUTO_HANDLE", "ESCALATE"}
    if invalid_escalations:
        raise ValueError(f"Invalid gold_escalation values: {sorted(invalid_escalations)}")

    original_labels = golden_df["gold_intent"].astype(str).str.strip()
    golden_df["gold_intent"] = original_labels.map(canonicalize_intent)
    logger.info("Normalized %d distinct human intent labels to canonical taxonomy", original_labels.nunique())
    return golden_df


def simulate_golden_labels(golden_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate gold labels for evaluation.
    
    In a real scenario, these would come from manual labeling.
    For demonstration, we use the predicted labels (acknowledging the limitation).
    
    Args:
        golden_df: Golden set candidates
        
    Returns:
        DataFrame with gold labels added
    """
    logger.info("Simulating gold labels (in production, these are manual)...")
    
    # For demonstration: use current intent as gold
    # In real scenario, a human would review and correct
    golden_df["gold_intent"] = golden_df["intent"]
    
    # Add escalation labels (simulate: escalate if low confidence or risky intent)
    escalate_intents = {"payment_issue", "account_access", "quality_defect"}
    golden_df["gold_escalation"] = golden_df.apply(
        lambda row: "ESCALATE" if (row["intent"] in escalate_intents or row["confidence"] < 0.6) else "AUTO_HANDLE",
        axis=1
    )
    golden_df["gold_escalation_reason"] = "Demo"
    
    logger.info(f"Simulated gold labels for {len(golden_df)} examples")
    return golden_df


def main():
    """Main evaluation execution."""
    logger.info("=" * 80)
    logger.info("PHASE 6: Evaluation on Golden Set")
    logger.info("=" * 80)
    
    # Load golden set
    logger.info("\n1. Loading golden set...")
    try:
        golden_df = read_csv_with_encoding_fallback("data/golden/golden_set.csv")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "data/golden/golden_set.csv is required for a valid evaluation. "
            "Label golden_candidates.csv and save it as golden_set.csv first."
        ) from exc
    
    # Simulate gold labels if not present
    if "gold_intent" not in golden_df.columns or golden_df["gold_intent"].isna().all():
        golden_df = simulate_golden_labels(golden_df)
    else:
        golden_df = validate_and_normalize_gold_labels(golden_df)
    
    logger.info(f"   Loaded {len(golden_df)} golden examples")
    
    # Load training data and remove every golden example to prevent leakage.
    logger.info("\n2. Loading training conversations...")
    train_df = pd.read_csv("data/processed/conversations_with_intents.csv")
    golden_ids = set(golden_df["customer_tweet_id"].astype(str))
    train_data = train_df[
        ~train_df["customer_tweet_id"].astype(str).isin(golden_ids)
    ].copy()
    if train_data.empty:
        raise ValueError("Removing golden examples left no training data.")
    
    logger.info(f"   Training data: {len(train_data)} examples")
    logger.info(f"   Golden evaluation set: {len(golden_df)} examples")
    
    # Initialize systems
    logger.info("\n3. Setting up systems...")
    
    # Trivial baseline
    trivial = TrivialBaseline(majority_intent="general_inquiry")
    
    # Simple baseline
    simple = SimpleBaseline(max_features=1000)
    simple.fit(train_data["customer_text"].tolist(), train_data["intent"].tolist())
    
    # AI system
    agent = SupportAgent()
    agent.index_cases(train_data)
    
    # Evaluate
    logger.info("\n4. Running evaluations...")
    
    # Intent classification
    logger.info("\n   --- Intent Classification ---")
    
    # Trivial intent predictions
    trivial_intents = [trivial.predict(text)[0] for text in golden_df["customer_text"]]
    trivial_intent_metrics = IntentEvaluator.evaluate(trivial_intents, golden_df["gold_intent"].tolist())
    
    # Simple intent predictions
    simple_intents = [simple.predict(text)[0] for text in golden_df["customer_text"]]
    simple_intent_metrics = IntentEvaluator.evaluate(simple_intents, golden_df["gold_intent"].tolist())
    
    # System intent predictions
    system_outputs = [agent.process(text) for text in golden_df["customer_text"]]
    system_intents = [output.intent.intent_id for output in system_outputs]
    
    system_intent_metrics = IntentEvaluator.evaluate(system_intents, golden_df["gold_intent"].tolist())
    
    # Escalation decisions
    logger.info("\n   --- Escalation Decisions ---")
    
    trivial_escalations = ["ESCALATE"] * len(golden_df)  # Trivial always escalates
    trivial_esc_metrics = EscalationEvaluator.evaluate(
        trivial_escalations,
        golden_df["gold_escalation"].tolist()
    )
    
    simple_escalations = []
    for idx, row in golden_df.iterrows():
        intent = simple_intents[idx]
        conf = 0.6  # Assume medium confidence
        text = row["customer_text"]
        decision, _ = SimpleEscalationPolicy.decide(intent, conf, text)
        simple_escalations.append(decision)
    
    simple_esc_metrics = EscalationEvaluator.evaluate(
        simple_escalations,
        golden_df["gold_escalation"].tolist()
    )
    
    system_escalations = [output.decision.value for output in system_outputs]
    
    system_esc_metrics = EscalationEvaluator.evaluate(
        system_escalations,
        golden_df["gold_escalation"].tolist()
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    logger.info("\n--- Intent Classification Accuracy ---")
    logger.info(f"Trivial Baseline:        {trivial_intent_metrics['accuracy']:.3f}")
    logger.info(f"Simple Baseline (TF-IDF): {simple_intent_metrics['accuracy']:.3f}")
    logger.info(f"AI System:               {system_intent_metrics['accuracy']:.3f}")
    
    logger.info("\n--- Escalation Accuracy ---")
    logger.info(f"Trivial (always escalate): {trivial_esc_metrics['accuracy']:.3f}")
    logger.info(f"Simple Policy:           {simple_esc_metrics['accuracy']:.3f}")
    logger.info(f"AI System:               {system_esc_metrics['accuracy']:.3f}")
    logger.info(f"AI System precision:     {system_esc_metrics['precision']:.3f}")
    logger.info(f"AI System recall:        {system_esc_metrics['recall']:.3f}")
    logger.info(f"AI System F1:            {system_esc_metrics['f1']:.3f}")
    
    logger.info("\n--- False Auto-Handle Rate (lower is better) ---")
    logger.info(f"Trivial:  {trivial_esc_metrics['false_auto_handle_rate']:.3f}")
    logger.info(f"Simple:   {simple_esc_metrics['false_auto_handle_rate']:.3f}")
    logger.info(f"AI System: {system_esc_metrics['false_auto_handle_rate']:.3f}")
    
    # Save results
    logger.info("\n5. Saving results...")
    results_dir = Path("artifacts/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        "approach": ["Trivial", "Simple", "AI System"],
        "intent_accuracy": [
            trivial_intent_metrics["accuracy"],
            simple_intent_metrics["accuracy"],
            system_intent_metrics["accuracy"]
        ],
        "intent_macro_f1": [
            trivial_intent_metrics["macro_f1"],
            simple_intent_metrics["macro_f1"],
            system_intent_metrics["macro_f1"]
        ],
        "escalation_accuracy": [
            trivial_esc_metrics["accuracy"],
            simple_esc_metrics["accuracy"],
            system_esc_metrics["accuracy"]
        ],
        "escalation_precision": [
            trivial_esc_metrics["precision"],
            simple_esc_metrics["precision"],
            system_esc_metrics["precision"]
        ],
        "escalation_recall": [
            trivial_esc_metrics["recall"],
            simple_esc_metrics["recall"],
            system_esc_metrics["recall"]
        ],
        "escalation_f1": [
            trivial_esc_metrics["f1"],
            simple_esc_metrics["f1"],
            system_esc_metrics["f1"]
        ],
        "false_auto_rate": [
            trivial_esc_metrics["false_auto_handle_rate"],
            simple_esc_metrics["false_auto_handle_rate"],
            system_esc_metrics["false_auto_handle_rate"]
        ],
    })
    
    results_df.to_csv(results_dir / "evaluation_results.csv", index=False)
    logger.info(f"✓ Results saved to {results_dir / 'evaluation_results.csv'}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6 Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
