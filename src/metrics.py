"""
Evaluation framework for the support agent.

Measures intent classification, escalation, and reply quality.
"""

import logging
from typing import dict, list

# import numpy as np
# import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class IntentEvaluator:
    """Evaluate intent classification."""
    
    @staticmethod
    def evaluate(predictions: list[str], references: list[str]) -> dict[str, object]:
        """
        Evaluate intent predictions.
        
        Args:
            predictions: Predicted intents
            references: Reference (gold) intents
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            "accuracy": accuracy_score(references, predictions),
            "macro_f1": f1_score(references, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(references, predictions, average="weighted", zero_division=0),
        }
        
        # Per-intent metrics
        unique_intents = set(references) | set(predictions)
        metrics["per_intent"] = {}
        
        for intent in unique_intents:
            ref_mask = [r == intent for r in references]
            pred_mask = [p == intent for p in predictions]
            
            metrics["per_intent"][intent] = {
                "precision": precision_score(ref_mask, pred_mask, zero_division=0),
                "recall": recall_score(ref_mask, pred_mask, zero_division=0),
                "f1": f1_score(ref_mask, pred_mask, zero_division=0),
                "support": sum(ref_mask),
            }
        
        # Confusion matrix
        unique_classes = sorted(unique_intents)
        cm = confusion_matrix(
            references,
            predictions,
            labels=unique_classes
        )
        metrics["confusion_matrix"] = {
            "labels": unique_classes,
            "matrix": cm.tolist()
        }
        
        return metrics


class EscalationEvaluator:
    """Evaluate escalation decisions."""
    
    @staticmethod
    def evaluate(predictions: list[str], references: list[str]) -> dict[str, object]:
        """
        Evaluate escalation decisions.
        
        Args:
            predictions: Predicted decisions ("AUTO_HANDLE" or "ESCALATE")
            references: Reference (gold) decisions
            
        Returns:
            Dictionary with metrics
        """
        # Convert to binary
        pred_binary = [1 if p == "ESCALATE" else 0 for p in predictions]
        ref_binary = [1 if r == "ESCALATE" else 0 for r in references]
        
        metrics = {
            "accuracy": accuracy_score(ref_binary, pred_binary),
            "precision": precision_score(ref_binary, pred_binary, zero_division=0),
            "recall": recall_score(ref_binary, pred_binary, zero_division=0),
            "f1": f1_score(ref_binary, pred_binary, zero_division=0),
        }
        
        # Critical metric: false auto-handles (predicted AUTO when should ESCALATE)
        false_auto = sum(
            1 for p, r in zip(pred_binary, ref_binary)
            if p == 0 and r == 1
        )
        metrics["false_auto_handle_count"] = false_auto
        metrics["false_auto_handle_rate"] = false_auto / sum(ref_binary) if sum(ref_binary) > 0 else 0
        
        return metrics


def print_evaluation_summary(
    system_metrics: dict[str, object],
    baseline1_metrics: dict[str, object],
    baseline2_metrics: dict[str, object],
):
    """
    Print evaluation summary table.
    
    Args:
        system_metrics: Main system metrics
        baseline1_metrics: Trivial baseline metrics
        baseline2_metrics: Simple baseline metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    logger.info("\n--- Intent Classification ---")
    logger.info(f"{'Approach':<25} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    logger.info("-" * 60)
    
    logger.info(
        f"{'Trivial Baseline':<25} "
        f"{baseline1_metrics.get('accuracy', 0):<12.3f} "
        f"{baseline1_metrics.get('macro_f1', 0):<12.3f} "
        f"{baseline1_metrics.get('weighted_f1', 0):<12.3f}"
    )
    
    logger.info(
        f"{'Simple Baseline (TF-IDF)':<25} "
        f"{baseline2_metrics.get('accuracy', 0):<12.3f} "
        f"{baseline2_metrics.get('macro_f1', 0):<12.3f} "
        f"{baseline2_metrics.get('weighted_f1', 0):<12.3f}"
    )
    
    logger.info(
        f"{'AI System':<25} "
        f"{system_metrics.get('accuracy', 0):<12.3f} "
        f"{system_metrics.get('macro_f1', 0):<12.3f} "
        f"{system_metrics.get('weighted_f1', 0):<12.3f}"
    )
    
    logger.info("\n--- Escalation Decision ---")
    logger.info(f"{'Approach':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'False Auto':<12}")
    logger.info("-" * 70)
    
    # (baselines would follow same pattern)
    logger.info(
        f"{'AI System':<25} "
        f"{system_metrics.get('accuracy', 0):<12.3f} "
        f"{system_metrics.get('precision', 0):<12.3f} "
        f"{system_metrics.get('recall', 0):<12.3f} "
        f"{system_metrics.get('false_auto_handle_rate', 0):<12.3f}"
    )
