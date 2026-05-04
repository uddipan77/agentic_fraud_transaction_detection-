"""Preprocessing pipeline: load data, build indexes, generate evidence bundles."""

from pathlib import Path

from .config import Config
from .data_loader import load_dataset
from .linking import DataLinker
from .evidence_store import build_all_evidence, save_evidence


def preprocess_split(split: str, config: Config) -> Path:
    """Run the full preprocessing pipeline for a dataset split.
    
    Returns the path to the saved evidence JSONL file.
    """
    print(f"[Preprocess] Loading {split} dataset...")
    dataset = load_dataset(config.dataset_path(split), name=split)
    print(f"  Transactions: {len(dataset.transactions)}")
    print(f"  Users: {len(dataset.users)}")
    print(f"  Locations: {len(dataset.locations)}")
    print(f"  SMS: {len(dataset.sms_messages)}")
    print(f"  Mails: {len(dataset.mail_messages)}")

    print(f"[Preprocess] Building data linker indexes...")
    linker = DataLinker(dataset)
    print(f"  Biotag->User mappings: {len(linker.biotag_to_user)}")
    print(f"  IBAN->User mappings: {len(linker.iban_to_user)}")

    print(f"[Preprocess] Building evidence bundles for {len(dataset.transactions)} transactions...")
    evidence_list = build_all_evidence(dataset, linker, config, split)

    evidence_path = config.evidence_dir(split) / "evidence.jsonl"
    save_evidence(evidence_list, evidence_path)
    print(f"[Preprocess] Evidence saved to {evidence_path}")

    # Summary stats
    risk_scores = [e.get("rule_based_risk", {}).get("risk_score", 0) for e in evidence_list]
    high_risk = sum(1 for s in risk_scores if s >= 0.4)
    medium_risk = sum(1 for s in risk_scores if 0.2 <= s < 0.4)
    low_risk = sum(1 for s in risk_scores if s < 0.2)
    print(f"  Risk distribution: high={high_risk}, medium={medium_risk}, low={low_risk}")

    return evidence_path


def preprocess_all(config: Config) -> dict[str, Path]:
    """Run preprocessing for both train and validation splits."""
    paths = {}
    for split in ["train", "validation"]:
        paths[split] = preprocess_split(split, config)
    return paths
