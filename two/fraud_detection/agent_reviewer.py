"""Agent 2: Reviewer Agent for low-confidence and high-value cases."""

from typing import Any

from .prompts import REVIEWER_SYSTEM_PROMPT, build_reviewer_prompt
from .llm_client import LLMClient
from .utils import extract_json_payload, parse_freeform_decision


class ReviewerAgent:
    """Independent reviewer agent for ambiguous cases."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def review(
        self,
        evidence: dict[str, Any],
        primary_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Review a low-confidence case and provide independent assessment."""
        transaction_id = evidence.get("transaction_id", "unknown")
        user_prompt = build_reviewer_prompt(evidence, primary_decision)

        raw_response = self.llm.invoke_reviewer(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        if not raw_response:
            return self._fallback_decision(evidence, primary_decision, "Empty response from reviewer")

        try:
            decision = extract_json_payload(raw_response)
            decision["transaction_id"] = transaction_id
            self._validate_decision(decision)
            return decision
        except (ValueError, KeyError) as e:
            recovered = self._recover_from_text(evidence, primary_decision, raw_response)
            if recovered:
                return recovered
            return self._fallback_decision(evidence, primary_decision, f"Parse error: {e}")

    def _validate_decision(self, decision: dict[str, Any]) -> None:
        if "decision" not in decision:
            raise KeyError("Missing 'decision' field")
        decision["decision"] = decision["decision"].lower().strip()
        if decision["decision"] not in ("fraud", "not_fraud"):
            raise ValueError(f"Invalid decision: {decision['decision']}")
        if "confidence" not in decision:
            decision["confidence"] = 0.5
        else:
            decision["confidence"] = max(0.0, min(1.0, float(decision["confidence"])))
        for field in ["evidence_for_fraud", "evidence_against_fraud", "uncertainties"]:
            if field not in decision or not isinstance(decision[field], list):
                decision[field] = []
        if "explanation" not in decision:
            decision["explanation"] = ""
        if "fraud_value_sensitivity" not in decision:
            decision["fraud_value_sensitivity"] = False

    def _recover_from_text(
        self,
        evidence: dict[str, Any],
        primary_decision: dict[str, Any],
        text: str,
    ) -> dict[str, Any] | None:
        decision, confidence = parse_freeform_decision(text)
        if decision is None:
            return None

        risk = evidence.get("rule_based_risk", {})
        return {
            "transaction_id": evidence.get("transaction_id", "unknown"),
            "decision": decision,
            "confidence": confidence if confidence is not None else 0.55,
            "explanation": text[:400],
            "evidence_for_fraud": risk.get("fraud_signals", [])[:4],
            "evidence_against_fraud": risk.get("legitimacy_signals", [])[:4],
            "uncertainties": risk.get("uncertainties", [])[:4],
            "fraud_value_sensitivity": risk.get("economic_high_impact", False),
            "source": "reviewer_freeform_recovery",
        }

    def _fallback_decision(
        self,
        evidence: dict[str, Any],
        primary_decision: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        """When reviewer fails, lean toward primary decision."""
        return {
            "transaction_id": evidence.get("transaction_id", "unknown"),
            "decision": primary_decision.get("decision", "not_fraud"),
            "confidence": max(0.45, primary_decision.get("confidence", 0.5) - 0.05),
            "explanation": f"Reviewer fallback: {reason}. Deferring to primary.",
            "evidence_for_fraud": primary_decision.get("evidence_for_fraud", []),
            "evidence_against_fraud": primary_decision.get("evidence_against_fraud", []),
            "uncertainties": primary_decision.get("uncertainties", []) + [reason],
            "fraud_value_sensitivity": primary_decision.get("fraud_value_sensitivity", False),
            "source": "reviewer_fallback",
        }
