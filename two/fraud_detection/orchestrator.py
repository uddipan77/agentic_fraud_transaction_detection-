"""Orchestrator: coordinate Agent 1, Agent 2, and final decision logic."""

import json
import time
from pathlib import Path
from typing import Any

from .config import Config
from .llm_client import LLMClient
from .agent_primary import PrimaryAgent
from .agent_reviewer import ReviewerAgent
from .evidence_store import load_evidence


class Orchestrator:
    """Orchestrate the 2-agent fraud detection workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.primary_agent = PrimaryAgent(self.llm_client)
        self.reviewer_agent = ReviewerAgent(self.llm_client)

    def should_review(self, primary_decision: dict[str, Any]) -> bool:
        """Decide whether to escalate to Agent 2."""
        confidence = primary_decision.get("confidence", 0.5)

        # Low confidence -> always review
        if confidence < self.config.review_confidence_threshold:
            return True

        # High value + suspicious -> review even if confidence is somewhat high
        if primary_decision.get("fraud_value_sensitivity", False):
            if confidence < self.config.high_value_review_threshold:
                return True

        return False

    def make_final_decision(
        self,
        primary: dict[str, Any],
        reviewer: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Combine Agent 1 and Agent 2 decisions into a final verdict."""
        if reviewer is None:
            # No review needed, accept primary
            return {
                "transaction_id": primary.get("transaction_id"),
                "final_decision": primary.get("decision"),
                "final_confidence": primary.get("confidence"),
                "primary_decision": primary.get("decision"),
                "primary_confidence": primary.get("confidence"),
                "reviewer_decision": None,
                "reviewer_confidence": None,
                "reviewed": False,
                "explanation": primary.get("explanation", ""),
            }

        p_dec = primary.get("decision", "not_fraud")
        p_conf = primary.get("confidence", 0.5)
        r_dec = reviewer.get("decision", "not_fraud")
        r_conf = reviewer.get("confidence", 0.5)

        # Agreement
        if p_dec == r_dec:
            final_decision = p_dec
            final_confidence = max(p_conf, r_conf)
        else:
            # Disagreement: use the one with higher confidence
            # But be conservative about false positives
            if r_conf > p_conf + 0.1:
                final_decision = r_dec
                final_confidence = r_conf
            elif p_conf > r_conf + 0.1:
                final_decision = p_dec
                final_confidence = p_conf
            else:
                # Very close confidence, be conservative
                # If one says fraud and the other says not_fraud, lean toward fraud
                # only if the economic importance is high
                if primary.get("fraud_value_sensitivity") or reviewer.get("fraud_value_sensitivity"):
                    # For high-value cases, err on side of flagging
                    final_decision = "fraud"
                    final_confidence = max(p_conf, r_conf)
                else:
                    # For low-value cases, be conservative (avoid false positive)
                    final_decision = "not_fraud"
                    final_confidence = max(p_conf, r_conf) * 0.9

        return {
            "transaction_id": primary.get("transaction_id"),
            "final_decision": final_decision,
            "final_confidence": final_confidence,
            "primary_decision": p_dec,
            "primary_confidence": p_conf,
            "reviewer_decision": r_dec,
            "reviewer_confidence": r_conf,
            "reviewed": True,
            "explanation": reviewer.get("explanation", primary.get("explanation", "")),
        }

    def process_transaction(
        self, evidence: dict[str, Any], index: int, total: int
    ) -> dict[str, Any]:
        """Process a single transaction through the agent pipeline."""
        txn_id = evidence.get("transaction_id", "unknown")
        risk_score = evidence.get("rule_based_risk", {}).get("risk_score", 0)

        print(f"  [{index + 1}/{total}] {txn_id} (risk={risk_score:.2f})", end=" ")

        # Agent 1: Primary investigation (always LLM)
        primary_decision = self.primary_agent.investigate(evidence)
        p_dec = primary_decision.get("decision", "?")
        p_conf = primary_decision.get("confidence", 0)
        print(f"-> A1: {p_dec} ({p_conf:.2f})", end="")

        # Check if review needed
        reviewer_decision = None
        if self.should_review(primary_decision):
            # Delay between agent calls
            time.sleep(0.5)
            reviewer_decision = self.reviewer_agent.review(evidence, primary_decision)
            r_dec = reviewer_decision.get("decision", "?")
            r_conf = reviewer_decision.get("confidence", 0)
            print(f" -> A2: {r_dec} ({r_conf:.2f})", end="")

        # Final decision
        final = self.make_final_decision(primary_decision, reviewer_decision)
        f_dec = final["final_decision"]
        print(f" => {f_dec}")

        # Store full audit trail
        final["evidence_risk_score"] = risk_score
        final["primary_full"] = primary_decision
        if reviewer_decision:
            final["reviewer_full"] = reviewer_decision

        return final

    def run_split(self, split: str) -> list[dict[str, Any]]:
        """Process all transactions in a dataset split."""
        evidence_path = self.config.evidence_dir(split) / "evidence.jsonl"
        if not evidence_path.exists():
            raise FileNotFoundError(f"Evidence file not found: {evidence_path}. Run preprocessing first.")

        print(f"\n[Orchestrator] Loading evidence for {split}...")
        evidence_list = load_evidence(evidence_path)
        print(f"  Loaded {len(evidence_list)} transaction evidence bundles")

        total = len(evidence_list)
        print(f"\n[Orchestrator] Processing {total} transactions with Agent 1 + Agent 2...\n")

        results = []
        for i, evidence in enumerate(evidence_list):
            result = self.process_transaction(evidence, i, total)
            results.append(result)

            # Inter-transaction delay to avoid rate limiting
            if i < total - 1:
                time.sleep(0.5)

        # Summary
        fraud_count = sum(1 for r in results if r["final_decision"] == "fraud")
        reviewed_count = sum(1 for r in results if r["reviewed"])
        print(f"\n[Orchestrator] {split} complete:")
        print(f"  Total: {total}")
        print(f"  Fraud: {fraud_count}")
        print(f"  Not Fraud: {total - fraud_count}")
        print(f"  Reviewed by Agent 2: {reviewed_count}")

        return results
