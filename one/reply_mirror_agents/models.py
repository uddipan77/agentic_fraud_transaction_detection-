from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


@dataclass(slots=True)
class UserProfile:
    first_name: str
    last_name: str
    birth_year: int
    salary: float
    job: str
    iban: str
    residence_city: str
    residence_lat: float
    residence_lng: float
    description: str
    biotag: str | None = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass(slots=True)
class TransactionRecord:
    transaction_id: str
    sender_id: str
    recipient_id: str
    transaction_type: str
    amount: float
    location: str
    payment_method: str
    sender_iban: str
    recipient_iban: str
    balance_after: float
    description: str
    timestamp: datetime


@dataclass(slots=True)
class LocationObservation:
    biotag: str
    timestamp: datetime
    lat: float
    lng: float
    city: str


@dataclass(slots=True)
class MessageRecord:
    channel: Literal["sms", "mail"]
    sender: str
    recipient: str
    subject: str
    timestamp: datetime | None
    body: str
    raw: str
    classifications: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RiskAssessment:
    risk_score: float
    fraud_signals: list[str]
    legitimacy_signals: list[str]
    uncertainties: list[str]
    economic_high_impact: bool
    economic_amount_threshold: float


@dataclass(slots=True)
class EvidenceBundle:
    transaction: TransactionRecord
    focal_user: UserProfile | None
    counterparty_user: UserProfile | None
    focal_role: str | None
    transaction_context: dict[str, Any]
    party_context: dict[str, Any]
    location_context: dict[str, Any]
    message_context: dict[str, Any]
    behavior_baseline: dict[str, Any]
    risk_assessment: RiskAssessment


class AgentDecision(BaseModel):
    transaction_id: str
    decision: Literal["fraud", "not_fraud"]
    confidence: float
    explanation: str
    evidence_summary: list[str] = Field(default_factory=list)
    evidence_for_fraud: list[str] = Field(default_factory=list)
    evidence_against_fraud: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    fraud_value_sensitivity: bool = False

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, parsed))

    def fraud_probability(self) -> float:
        return self.confidence if self.decision == "fraud" else 1.0 - self.confidence


@dataclass(slots=True)
class PredictionRecord:
    transaction_id: str
    amount: float
    economic_risk_flag: bool
    decision_agent1: str
    confidence_agent1: float
    decision_agent2: str | None
    confidence_agent2: float | None
    review_triggered: bool
    final_decision: str
    tie_break_policy: str
    risk_score: float
    reasons: str