"""Output writer: generate output.txt and audit files."""

import csv
import json
from pathlib import Path
from typing import Any


def write_output_txt(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write fraudulent transaction IDs to output.txt, one per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fraud_ids = [
        r["transaction_id"]
        for r in results
        if r.get("final_decision") == "fraud"
    ]
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for txn_id in fraud_ids:
            f.write(txn_id + "\n")
    print(f"  Written {len(fraud_ids)} fraud IDs to {output_path}")


def write_predictions_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write detailed predictions to a CSV file for audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "transaction_id", "final_decision", "final_confidence",
            "primary_decision", "primary_confidence",
            "reviewer_decision", "reviewer_confidence",
            "reviewed", "evidence_risk_score", "explanation",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "transaction_id": r.get("transaction_id", ""),
                "final_decision": r.get("final_decision", ""),
                "final_confidence": r.get("final_confidence", ""),
                "primary_decision": r.get("primary_decision", ""),
                "primary_confidence": r.get("primary_confidence", ""),
                "reviewer_decision": r.get("reviewer_decision", ""),
                "reviewer_confidence": r.get("reviewer_confidence", ""),
                "reviewed": r.get("reviewed", False),
                "evidence_risk_score": r.get("evidence_risk_score", ""),
                "explanation": r.get("explanation", "")[:300],
            })
    print(f"  Written predictions to {output_path}")


def write_full_audit(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write full audit trail as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Written full audit to {output_path}")
