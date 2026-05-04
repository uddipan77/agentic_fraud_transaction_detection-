from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import zipfile

from models import PredictionRecord


def write_output_txt(path: Path, fraud_ids: list[str]) -> None:
    payload = "\n".join(fraud_ids)
    path.write_text(payload, encoding="ascii", errors="ignore")


def write_predictions_csv(path: Path, predictions: list[PredictionRecord]) -> None:
    fieldnames = [
        "transaction_id",
        "amount",
        "economic_risk_flag",
        "decision_agent1",
        "confidence_agent1",
        "decision_agent2",
        "confidence_agent2",
        "review_triggered",
        "final_decision",
        "tie_break_policy",
        "risk_score",
        "reasons",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in predictions:
            writer.writerow(asdict(item))


def write_langfuse_sessions(path: Path, sessions: list[dict]) -> None:
    path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def create_source_archive(project_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in project_root.rglob("*"):
            if file_path.is_dir():
                continue
            if "__pycache__" in file_path.parts:
                continue
            if project_root / "output" in file_path.parents:
                continue
            archive.write(file_path, arcname=file_path.relative_to(project_root))