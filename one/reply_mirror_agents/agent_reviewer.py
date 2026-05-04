from __future__ import annotations

from llm_utils import LLMClient
from models import AgentDecision, EvidenceBundle
from prompts import REVIEWER_SYSTEM_PROMPT, build_reviewer_prompt
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


class ReviewerAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def _fallback_decision(
        self,
        bundle: EvidenceBundle,
        primary_decision: AgentDecision,
        reason: str,
    ) -> AgentDecision:
        explanation = (
            f"Fallback reviewer decision used because model output could not be parsed. {reason}"
        )
        return AgentDecision(
            transaction_id=bundle.transaction.transaction_id,
            decision=primary_decision.decision,
            confidence=max(0.51, primary_decision.confidence - 0.05),
            explanation=explanation,
            evidence_summary=primary_decision.evidence_summary[:4],
            evidence_for_fraud=primary_decision.evidence_for_fraud[:4],
            evidence_against_fraud=primary_decision.evidence_against_fraud[:4],
            uncertainties=(primary_decision.uncertainties + bundle.risk_assessment.uncertainties)[:4],
            fraud_value_sensitivity=bundle.risk_assessment.economic_high_impact,
        )

    def _recover_from_freeform_text(
        self,
        bundle: EvidenceBundle,
        primary_decision: AgentDecision,
        raw_text: str,
    ) -> AgentDecision | None:
        decision, confidence = _parse_freeform_decision(raw_text)
        if decision is None:
            return None
        explanation = " ".join(raw_text.strip().split())[:420]
        return AgentDecision(
            transaction_id=bundle.transaction.transaction_id,
            decision=decision,
            confidence=confidence if confidence is not None else max(0.56, primary_decision.confidence - 0.03),
            explanation=explanation,
            evidence_summary=(primary_decision.evidence_summary + bundle.risk_assessment.fraud_signals[:2])[:4],
            evidence_for_fraud=bundle.risk_assessment.fraud_signals[:4],
            evidence_against_fraud=bundle.risk_assessment.legitimacy_signals[:4],
            uncertainties=bundle.risk_assessment.uncertainties[:4],
            fraud_value_sensitivity=bundle.risk_assessment.economic_high_impact,
        )

    def review(
        self,
        evidence_payload: dict,
        evidence_bundle: EvidenceBundle,
        primary_decision: AgentDecision,
        run_trace: RunTrace,
    ) -> AgentDecision:
        try:
            raw_response = self.llm_client.invoke_reviewer(
                system_prompt=REVIEWER_SYSTEM_PROMPT,
                user_prompt=build_reviewer_prompt(evidence_payload, primary_decision.model_dump()),
                session_id=run_trace.session_id,
                dataset_name=run_trace.dataset_name,
                transaction_id=evidence_bundle.transaction.transaction_id,
            )
        except Exception as exc:
            return self._fallback_decision(
                evidence_bundle,
                primary_decision,
                f"Reviewer model call failed: {exc}",
            )
        try:
            parsed = extract_json_payload(raw_response)
            decision = AgentDecision.model_validate(parsed)
            if not decision.evidence_summary:
                decision.evidence_summary = (
                    decision.evidence_for_fraud[:2] + decision.evidence_against_fraud[:2]
                )[:4]
            return decision
        except Exception as exc:
            recovered = self._recover_from_freeform_text(
                evidence_bundle,
                primary_decision,
                raw_response,
            )
            if recovered is not None:
                return recovered
            return self._fallback_decision(evidence_bundle, primary_decision, str(exc))