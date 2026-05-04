from __future__ import annotations

from typing import Any

from llm_utils import LLMClient
from models import AgentDecision, EvidenceBundle
from prompts import PRIMARY_SYSTEM_PROMPT, build_primary_prompt
from tools import FraudTools
from tracing import RunTrace
from utils import extract_json_payload


def _parse_freeform_decision(text: str) -> tuple[str | None, float | None]:
    lowered = text.lower()
    decision = None
    if "not_fraud" in lowered:
        decision = "not_fraud"
    elif " fraud" in lowered or lowered.startswith("fraud"):
        decision = "fraud"

    confidence = None
    patterns = [
        r"confidence(?: of [a-z_]+)?\s*(?:=|is|:)?\s*([01](?:\.\d+)?)",
        r"fraud probability(?: maybe| is)?\s*([01](?:\.\d+)?)",
    ]
    for pattern in patterns:
        import re

        match = re.search(pattern, lowered)
        if match:
            value = float(match.group(1))
            if "fraud probability" in pattern and decision == "not_fraud":
                confidence = 1.0 - value
            else:
                confidence = value
            break
    return decision, confidence


class PrimaryFraudInvestigator:
    def __init__(self, tools: FraudTools, llm_client: LLMClient) -> None:
        self.tools = tools
        self.llm_client = llm_client

    def _serialize_evidence(self, bundle: EvidenceBundle) -> dict[str, Any]:
        focal_user = None
        if bundle.focal_user is not None:
            focal_user = {
                "full_name": bundle.focal_user.full_name,
                "salary": bundle.focal_user.salary,
                "job": bundle.focal_user.job,
                "residence_city": bundle.focal_user.residence_city,
                "description": bundle.focal_user.description,
            }
        counterparty_user = None
        if bundle.counterparty_user is not None:
            counterparty_user = {
                "full_name": bundle.counterparty_user.full_name,
                "residence_city": bundle.counterparty_user.residence_city,
            }
        return {
            "transaction": bundle.transaction_context,
            "focal_user": focal_user,
            "counterparty_user": counterparty_user,
            "focal_role": bundle.focal_role,
            "party_context": bundle.party_context,
            "location_context": bundle.location_context,
            "message_context": bundle.message_context,
            "behavior_baseline": bundle.behavior_baseline,
            "rule_based_risk": {
                "risk_score": bundle.risk_assessment.risk_score,
                "fraud_signals": bundle.risk_assessment.fraud_signals,
                "legitimacy_signals": bundle.risk_assessment.legitimacy_signals,
                "uncertainties": bundle.risk_assessment.uncertainties,
                "economic_high_impact": bundle.risk_assessment.economic_high_impact,
                "economic_amount_threshold": bundle.risk_assessment.economic_amount_threshold,
            },
        }

    def _fallback_decision(self, bundle: EvidenceBundle, reason: str) -> AgentDecision:
        risk = bundle.risk_assessment.risk_score
        decision = "fraud" if risk >= 0.60 else "not_fraud"
        confidence = 0.70 if risk >= 0.75 or risk <= 0.30 else 0.58
        explanation = (
            f"Fallback decision used because model output could not be parsed. {reason} "
            f"Rule-based risk was {risk:.2f}."
        )
        return AgentDecision(
            transaction_id=bundle.transaction.transaction_id,
            decision=decision,
            confidence=confidence,
            explanation=explanation,
            evidence_summary=(bundle.risk_assessment.fraud_signals + bundle.risk_assessment.legitimacy_signals)[:4],
            evidence_for_fraud=bundle.risk_assessment.fraud_signals[:4],
            evidence_against_fraud=bundle.risk_assessment.legitimacy_signals[:4],
            uncertainties=bundle.risk_assessment.uncertainties[:4],
            fraud_value_sensitivity=bundle.risk_assessment.economic_high_impact,
        )

    def _recover_from_freeform_text(
        self,
        bundle: EvidenceBundle,
        raw_text: str,
    ) -> AgentDecision | None:
        decision, confidence = _parse_freeform_decision(raw_text)
        if decision is None:
            return None
        explanation = " ".join(raw_text.strip().split())[:420]
        return AgentDecision(
            transaction_id=bundle.transaction.transaction_id,
            decision=decision,
            confidence=confidence if confidence is not None else 0.63,
            explanation=explanation,
            evidence_summary=(bundle.risk_assessment.fraud_signals + bundle.risk_assessment.legitimacy_signals)[:4],
            evidence_for_fraud=bundle.risk_assessment.fraud_signals[:4],
            evidence_against_fraud=bundle.risk_assessment.legitimacy_signals[:4],
            uncertainties=bundle.risk_assessment.uncertainties[:4],
            fraud_value_sensitivity=bundle.risk_assessment.economic_high_impact,
        )

    def investigate(
        self,
        transaction_id: str,
        run_trace: RunTrace,
    ) -> tuple[EvidenceBundle, AgentDecision]:
        evidence = self.tools.build_transaction_evidence_bundle(transaction_id, self.tools.dataset.name)
        prompt_payload = self._serialize_evidence(evidence)
        try:
            raw_response = self.llm_client.invoke_primary(
                system_prompt=PRIMARY_SYSTEM_PROMPT,
                user_prompt=build_primary_prompt(prompt_payload),
                session_id=run_trace.session_id,
                dataset_name=run_trace.dataset_name,
                transaction_id=transaction_id,
            )
        except Exception as exc:
            return evidence, self._fallback_decision(evidence, f"Primary model call failed: {exc}")
        try:
            parsed = extract_json_payload(raw_response)
            decision = AgentDecision.model_validate(parsed)
            if not decision.evidence_summary:
                decision.evidence_summary = (
                    decision.evidence_for_fraud[:2] + decision.evidence_against_fraud[:2]
                )[:4]
            return evidence, decision
        except Exception as exc:
            recovered = self._recover_from_freeform_text(evidence, raw_response)
            if recovered is not None:
                return evidence, recovered
            return evidence, self._fallback_decision(evidence, str(exc))