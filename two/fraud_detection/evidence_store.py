"""Evidence store: build and persist structured evidence bundles."""

import json
from pathlib import Path
from typing import Any

from .config import Config
from .data_loader import Dataset
from .linking import DataLinker
from .tools import (
    get_transaction_context,
    resolve_transaction_parties,
    get_location_context,
    get_message_context,
    get_behavior_baseline,
)
from .rules import score_rule_based_risk


def build_evidence_bundle(
    transaction_id: str,
    linker: DataLinker,
    config: Config,
) -> dict[str, Any]:
    """Build a complete evidence bundle for a single transaction."""
    txn = linker.txn_by_id.get(transaction_id)
    if txn is None:
        return {"transaction_id": transaction_id, "error": "Transaction not found"}

    # Step 1: Transaction context
    txn_context = get_transaction_context(txn)

    # Step 2: Resolve parties
    parties = resolve_transaction_parties(txn, linker)

    # Step 3: Get sender user (for further lookups)
    sender_user = linker.resolve_sender_user(txn)

    # Step 4: Location context
    loc_context = get_location_context(
        txn, sender_user, linker, config.location_window_hours
    )

    # Step 5: Message context
    msg_context = get_message_context(
        sender_user, txn, linker, config.message_lookback_days
    )

    # Step 6: Behavior baseline
    beh_baseline = get_behavior_baseline(txn.sender_id, txn, linker)

    # Step 7: Assemble pre-rule evidence
    evidence = {
        "transaction_id": transaction_id,
        "transaction": txn_context,
        "parties": parties,
        "location_context": loc_context,
        "message_context": msg_context,
        "behavior_baseline": beh_baseline,
    }

    # Step 8: Rule-based risk scoring
    risk = score_rule_based_risk(evidence)
    evidence["rule_based_risk"] = risk

    return evidence


def build_all_evidence(
    dataset: Dataset,
    linker: DataLinker,
    config: Config,
    split: str,
) -> list[dict[str, Any]]:
    """Build evidence bundles for all transactions in a dataset."""
    evidence_list = []
    for txn in dataset.transactions:
        bundle = build_evidence_bundle(txn.transaction_id, linker, config)
        evidence_list.append(bundle)
    return evidence_list


def save_evidence(evidence_list: list[dict[str, Any]], output_path: Path) -> None:
    """Save evidence bundles as a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for bundle in evidence_list:
            f.write(json.dumps(bundle, default=str) + "\n")


def load_evidence(path: Path) -> list[dict[str, Any]]:
    """Load evidence bundles from a JSONL file."""
    bundles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                bundles.append(json.loads(line))
    return bundles
