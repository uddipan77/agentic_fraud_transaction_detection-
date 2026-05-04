"""Agent 1: Primary Fraud Investigator."""

from typing import Any

from .prompts import PRIMARY_SYSTEM_PROMPT, build_primary_prompt
from .llm_client import LLMClient
from .utils import extract_json_payload, parse_freeform_decision


class PrimaryAgent:
    """Primary fraud investigation agent."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def investigate(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Analyze evidence and return a fraud decision."""
        transaction_id = evidence.get("transaction_id", "unknown")
        user_prompt = build_primary_prompt(evidence)

        raw_response = self.llm.invoke_primary(
            system_prompt=PRIMARY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        if not raw_response:
            return self._fallback_decision(evidence, "Empty response from primary model")

        # Try to parse JSON
        try:
            decision = extract_json_payload(raw_response)
            decision["transaction_id"] = transaction_id
            self._validate_decision(decision)
            return decision
        except (ValueError, KeyError) as e:
            # Try freeform text recovery
            recovered = self._recover_from_text(evidence, raw_response)
            if recovered:
                return recovered
            return self._fallback_decision(evidence, f"Parse error: {e}")

    def _validate_decision(self, decision: dict[str, Any]) -> None:
        """Ensure decision has required fields and valid values."""
        if "decision" not in decision:
            raise KeyError("Missing 'decision' field")
        decision["decision"] = decision["decision"].lower().strip()
        if decision["decision"] not in ("fraud", "not_fraud"):
            raise ValueError(f"Invalid decision: {decision['decision']}")
        if "confidence" not in decision:
            decision["confidence"] = 0.5
        else:
            decision["confidence"] = max(0.0, min(1.0, float(decision["confidence"])))
        # Ensure list fields exist
        for field in ["evidence_for_fraud", "evidence_against_fraud", "uncertainties"]:
            if field not in decision or not isinstance(decision[field], list):
                decision[field] = []
        if "explanation" not in decision:
            decision["explanation"] = ""
        if "fraud_value_sensitivity" not in decision:
            decision["fraud_value_sensitivity"] = False

    def _recover_from_text(self, evidence: dict[str, Any], text: str) -> dict[str, Any] | None:
        """Try to extract decision from freeform text."""
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
            "source": "freeform_recovery",
        }

    def _fallback_decision(self, evidence: dict[str, Any], reason: str) -> dict[str, Any]:
        """Use rule-based risk as fallback when LLM fails."""
        risk = evidence.get("rule_based_risk", {})
        risk_score = risk.get("risk_score", 0.0)

        decision = "fraud" if risk_score >= 0.45 else "not_fraud"
        confidence = 0.70 if risk_score >= 0.6 or risk_score <= 0.2 else 0.50

        return {
            "transaction_id": evidence.get("transaction_id", "unknown"),
            "decision": decision,
            "confidence": confidence,
            "explanation": f"Fallback: {reason}. Rule-based risk: {risk_score:.2f}",
            "evidence_for_fraud": risk.get("fraud_signals", [])[:4],
            "evidence_against_fraud": risk.get("legitimacy_signals", [])[:4],
            "uncertainties": risk.get("uncertainties", [])[:4],
            "fraud_value_sensitivity": risk.get("economic_high_impact", False),
            "source": "fallback",
        }
