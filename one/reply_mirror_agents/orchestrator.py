from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_primary import PrimaryFraudInvestigator
from agent_reviewer import ReviewerAgent
from config import AppConfig
from data_loader import load_dataset
from llm_utils import LLMClient
from models import AgentDecision, PredictionRecord
from output_writer import write_output_txt, write_predictions_csv
from tools import FraudTools
from tracing import RunTrace, TraceManager


@dataclass(slots=True)
class DatasetRunResult:
    split: str
    output_path: Path
    predictions_path: Path
    fraud_ids: list[str]
    session_id: str
    trace_id: str | None
    trace_url: str | None


class FraudOrchestrator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tracer = TraceManager(config)
        self.llm_client = LLMClient(config, self.tracer)

    def _should_review(self, decision: AgentDecision, risk_score: float, economic_high_impact: bool) -> bool:
        if decision.confidence < self.config.review_threshold:
            return True
        if economic_high_impact and decision.fraud_probability() >= self.config.high_value_review_probability:
            return True
        if risk_score >= 0.75 and decision.confidence < 0.9:
            return True
        return False

    def _combine(
        self,
        primary: AgentDecision,
        reviewer: AgentDecision | None,
        risk_score: float,
        economic_high_impact: bool,
    ) -> tuple[str, str]:
        if reviewer is None:
            return primary.decision, "primary_high_confidence"
        if reviewer.decision == primary.decision:
            return primary.decision, "agents_agreed"

        combined_probability = (
            0.45 * risk_score
            + 0.35 * primary.fraud_probability()
            + 0.20 * reviewer.fraud_probability()
        )
        if economic_high_impact and combined_probability >= self.config.disagreement_threshold - 0.05:
            combined_probability += 0.05
        final_decision = "fraud" if combined_probability >= self.config.disagreement_threshold else "not_fraud"
        return final_decision, f"weighted_disagreement_{combined_probability:.2f}"

    def _prediction_record(
        self,
        primary: AgentDecision,
        reviewer: AgentDecision | None,
        final_decision: str,
        tie_break_policy: str,
        amount: float,
        economic_high_impact: bool,
        risk_score: float,
    ) -> PredictionRecord:
        reason_parts = primary.evidence_summary[:2]
        if reviewer is not None:
            reason_parts.extend(reviewer.evidence_summary[:2])
        return PredictionRecord(
            transaction_id=primary.transaction_id,
            amount=amount,
            economic_risk_flag=economic_high_impact,
            decision_agent1=primary.decision,
            confidence_agent1=primary.confidence,
            decision_agent2=reviewer.decision if reviewer else None,
            confidence_agent2=reviewer.confidence if reviewer else None,
            review_triggered=reviewer is not None,
            final_decision=final_decision,
            tie_break_policy=tie_break_policy,
            risk_score=risk_score,
            reasons=" | ".join(reason_parts),
        )

    def run_dataset(self, split: str) -> DatasetRunResult:
        dataset = load_dataset(self.config.dataset_path(split), name=split)
        tools = FraudTools(dataset, self.config, self.tracer)
        primary_agent = PrimaryFraudInvestigator(tools, self.llm_client)
        reviewer_agent = ReviewerAgent(self.llm_client)
        fraud_ids: list[str] = []
        predictions: list[PredictionRecord] = []
        input_payload = {
            "split": split,
            "dataset_path": str(dataset.path),
            "transaction_count": len(dataset.transactions),
            "models": {
                "primary": self.config.primary_model,
                "reviewer": self.config.reviewer_model,
            },
        }

        with self.tracer.start_run(split, input_payload=input_payload) as run_trace:
            for index, transaction in enumerate(dataset.transactions, start=1):
                with self.tracer.span(
                    "transaction_decision_flow",
                    as_type="chain",
                    input_payload={
                        "transaction_id": transaction.transaction_id,
                        "index": index,
                        "split": split,
                    },
                ):
                    evidence, primary_decision = primary_agent.investigate(
                        transaction.transaction_id,
                        run_trace,
                    )
                    serialized_evidence = primary_agent._serialize_evidence(evidence)
                    reviewer_decision = None
                    if self._should_review(
                        primary_decision,
                        evidence.risk_assessment.risk_score,
                        evidence.risk_assessment.economic_high_impact,
                    ):
                        reviewer_decision = reviewer_agent.review(
                            serialized_evidence,
                            evidence,
                            primary_decision,
                            run_trace,
                        )
                    final_decision, tie_break_policy = self._combine(
                        primary_decision,
                        reviewer_decision,
                        evidence.risk_assessment.risk_score,
                        evidence.risk_assessment.economic_high_impact,
                    )
                    if final_decision == "fraud":
                        fraud_ids.append(transaction.transaction_id)
                    predictions.append(
                        self._prediction_record(
                            primary_decision,
                            reviewer_decision,
                            final_decision,
                            tie_break_policy,
                            transaction.amount,
                            evidence.risk_assessment.economic_high_impact,
                            evidence.risk_assessment.risk_score,
                        )
                    )
                    self.tracer.event(
                        "transaction_finalized",
                        input_payload={"transaction_id": transaction.transaction_id},
                        output_payload={
                            "final_decision": final_decision,
                            "risk_score": evidence.risk_assessment.risk_score,
                            "review_triggered": reviewer_decision is not None,
                        },
                    )

            output_path = self.config.output_dir / f"{split}_output.txt"
            predictions_path = self.config.output_dir / f"predictions_{split}.csv"
            write_output_txt(output_path, fraud_ids)
            write_predictions_csv(predictions_path, predictions)

            return DatasetRunResult(
                split=split,
                output_path=output_path,
                predictions_path=predictions_path,
                fraud_ids=fraud_ids,
                session_id=run_trace.session_id,
                trace_id=run_trace.trace_id,
                trace_url=self.tracer.trace_url(run_trace.trace_id),
            )