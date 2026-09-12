"""
Core AI support agent pipeline.

Combines classification, retrieval, generation, and escalation decisions.
"""

import logging

import pandas as pd

from src.baselines import SimpleEscalationPolicy
from src.config import get_config
from src.intent_taxonomy import classify_by_keywords, create_default_taxonomy
from src.llm import GeminiClient, LLMError
from src.retrieval import SimpleRetrieval
from src.schemas import (
    AgentOutput,
    EscalationDecision,
    PredictedIntent,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class SupportAgent:
    """Simple AI support agent."""
    
    def __init__(self):
        """Initialize agent."""
        self.taxonomy = create_default_taxonomy()
        self.retrieval = SimpleRetrieval()
        self.is_ready = False
        self.llm = None
        config = get_config()
        if config.llm_provider.lower() == "gemini" and config.gemini_api_key:
            self.llm = GeminiClient(config)
    
    def index_cases(self, cases_df: pd.DataFrame):
        """
        Index historical cases for retrieval.
        
        Args:
            cases_df: DataFrame with 'customer_text' and 'brand_text'
        """
        logger.info("Indexing conversation cases...")
        self.retrieval.index(cases_df)
        self.is_ready = True
        logger.info("✓ Agent ready")
    
    def process(
        self,
        customer_message: str,
        exclude_tweet_ids: list | None = None,
    ) -> AgentOutput:
        """
        Process a customer message end-to-end.
        
        Args:
            customer_message: Customer's input message
            exclude_tweet_ids: Tweet IDs to exclude from retrieval (leakage prevention)
            
        Returns:
            AgentOutput with intent, reply, decision, evidence
        """
        if not self.is_ready:
            logger.warning("Agent not indexed yet")
            return AgentOutput(
                intent=PredictedIntent(intent_id="other", confidence=0.0),
                reply="System error: agent not initialized",
                decision=EscalationDecision.ESCALATE,
                escalation_reason="System not ready"
            )
        
        # 1. Classify intent
        intent_id, intent_confidence = classify_by_keywords(customer_message, self.taxonomy)
        
        # 2. Retrieve similar cases
        retrieved = self.retrieval.retrieve(
            customer_message,
            top_k=3,
            exclude_ids=exclude_tweet_ids
        )
        
        # Build evidence
        evidence = [
            RetrievalResult(
                conversation_id=f"ref_{i}",
                similarity_score=min(1.0, max(0.0, r["similarity"])),  # Clamp to [0, 1]
                customer_issue=r["customer_message"][:100],
                brand_response=r["brand_response"][:100],
                context_summary=f"Intent: {r['intent']}",
                relevance_reason=f"Similar {r['intent']} issue"
            )
            for i, r in enumerate(retrieved)
        ]
        
        # 3. Generate a grounded reply, keeping retrieval as a local fallback.
        reply = self._fallback_reply(retrieved)
        if self.llm:
            prompt = self._reply_prompt(customer_message, retrieved)
            try:
                reply = self.llm.generate(prompt)
            except LLMError as exc:
                logger.warning("Gemini reply generation failed; using fallback: %s", exc)
        
        # 4. Make escalation decision
        decision_str, escalation_reason = SimpleEscalationPolicy.decide(
            intent_id,
            intent_confidence,
            customer_message
        )
        
        decision = (
            EscalationDecision.AUTO_HANDLE
            if decision_str == "AUTO_HANDLE"
            else EscalationDecision.ESCALATE
        )
        
        return AgentOutput(
            intent=PredictedIntent(
                intent_id=intent_id,
                confidence=intent_confidence
            ),
            reply=reply,
            decision=decision,
            escalation_reason=escalation_reason,
            evidence=evidence
        )

    @staticmethod
    def _fallback_reply(retrieved: list[dict]) -> str:
        if retrieved:
            return f"Based on similar cases: {retrieved[0]['brand_response'][:200]}"
        return "I'd be happy to help. Could you provide more details?"

    @staticmethod
    def _reply_prompt(customer_message: str, retrieved: list[dict]) -> str:
        examples = "\n".join(
            f"Customer: {item['customer_message'][:500]}\nSupport: {item['brand_response'][:500]}"
            for item in retrieved
        )
        return (
            "You are a concise, empathetic customer support agent. Answer the customer's message "
            "using the relevant examples below. Do not invent policies, refunds, order status, or "
            "account actions. If the examples do not provide enough information, ask one useful "
            "clarifying question. Return only the reply text.\n\n"
            f"Customer message: {customer_message}\n\nRelevant examples:\n{examples}"
        )
