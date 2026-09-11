"""
Data schemas for the Hiver support agent.

Defines Pydantic models for structured data throughout the pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message (tweet) in a conversation."""

    tweet_id: str
    author_id: str
    text: str
    created_at: datetime
    in_response_to_tweet_id: Optional[str] = None
    is_brand: bool  # True if from brand, False if from customer


class ConversationTurn(BaseModel):
    """A single turn in a conversation (customer message + brand response)."""

    turn_id: str
    conversation_id: str
    customer_message: Message
    customer_context: List[Message] = Field(default_factory=list)  # Previous customer messages
    brand_response: Optional[Message] = None
    brand_response_context: List[Message] = Field(default_factory=list)  # Previous brand responses
    timestamp: datetime
    source_tweets: List[str] = Field(default_factory=list)  # Original tweet IDs


class IntentDefinition(BaseModel):
    """Definition of a support intent."""

    intent_id: str
    name: str
    description: str
    inclusion_criteria: str
    exclusion_criteria: str
    examples: List[str] = Field(default_factory=list)


class PredictedIntent(BaseModel):
    """Predicted intent for a customer message."""

    intent_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class RetrievalResult(BaseModel):
    """A retrieved historical support case."""

    conversation_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    customer_issue: str
    brand_response: str
    context_summary: str
    relevance_reason: str


class EscalationDecision(Enum):
    """Escalation decision for a support request."""

    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE = "ESCALATE"


class AgentOutput(BaseModel):
    """Final structured output from the support agent."""

    intent: PredictedIntent
    reply: str
    decision: EscalationDecision
    escalation_reason: str
    evidence: List[RetrievalResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GoldenExample(BaseModel):
    """A hand-labelled golden evaluation example."""

    example_id: str
    conversation_id: str
    customer_message: str
    context: str
    gold_intent: str
    gold_escalation: EscalationDecision
    gold_escalation_reason: str
    gold_reference_response: Optional[str] = None
    acceptable_response_criteria: Optional[str] = None
    notes: Optional[str] = None


class EvaluationResult(BaseModel):
    """Result of evaluating a single example."""

    example_id: str
    predicted_intent: str
    gold_intent: str
    intent_correct: bool
    predicted_escalation: EscalationDecision
    gold_escalation: EscalationDecision
    escalation_correct: bool
    predicted_reply: str
    reply_quality_score: Optional[float] = None
    reply_quality_reasoning: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class DatasetStatistics:
    """Statistics about the dataset."""

    total_records: int
    selected_brand_records: int
    usable_conversations: int
    usable_customer_messages: int
    unique_intents: int
    avg_conversation_length: float
    date_range: tuple  # (min_date, max_date)
    missing_data_counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
