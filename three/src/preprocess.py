"""Build evidence bundles and compute fast risk scores for all transactions."""
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import math

from .linking import (
    build_user_index, resolve_entity, resolve_by_iban,
    build_message_index, get_messages_near_time,
    build_location_index, check_location_plausibility,
    normalize_str,
)
from .audio_processor import (
    transcribe_audio_files, link_audio_to_user, get_audio_near_transaction,
)


def build_transaction_history(transactions: list[dict]) -> dict:
    """Build per-sender historical stats (computed incrementally by timestamp)."""
    # Sort by time
    txns_sorted = sorted(transactions, key=lambda t: t["timestamp_dt"])

    history = defaultdict(lambda: {
        "amounts": [],
        "recipients": set(),
        "payment_methods": set(),
        "txn_types": set(),
        "locations": set(),
        "timestamps": [],
        "count": 0,
    })

    # We'll store history snapshots per transaction
    txn_history = {}

    for txn in txns_sorted:
        sid = txn["sender_id"]
        h = history[sid]

        # Snapshot current history BEFORE this transaction
        txn_history[txn["transaction_id"]] = {
            "avg_amount": sum(h["amounts"]) / len(h["amounts"]) if h["amounts"] else 0,
            "max_amount": max(h["amounts"]) if h["amounts"] else 0,
            "std_amount": _std(h["amounts"]) if len(h["amounts"]) > 1 else 0,
            "known_recipients": len(h["recipients"]),
            "known_methods": list(h["payment_methods"]),
            "txn_count": h["count"],
            "known_locations": list(h["locations"])[:5],
            "is_new_recipient": (txn.get("recipient_id") or "") not in h["recipients"] and bool(txn.get("recipient_id")),
            "is_new_method": (txn.get("payment_method") or "") not in h["payment_methods"] and bool(txn.get("payment_method")),
        }

        # Update history
        h["amounts"].append(txn["amount"])
        if txn.get("recipient_id"):
            h["recipients"].add(txn["recipient_id"])
        if txn.get("payment_method"):
            h["payment_methods"].add(txn["payment_method"])
        if txn.get("transaction_type"):
            h["txn_types"].add(txn["transaction_type"])
        if txn.get("location"):
            h["locations"].add(txn["location"])
        h["timestamps"].append(txn["timestamp_dt"])
        h["count"] += 1

    return txn_history


def _std(values):
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def build_all_evidence(data: dict, cache_path: Path, whisper_model: str = "base") -> list[dict]:
    """Build complete evidence bundles for all transactions.

    This is the main preprocessing step. Results cached to disk.
    """
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            evidence = json.load(f)
        print(f"  Loaded {len(evidence)} cached evidence bundles")
        return evidence

    transactions = data["transactions"]
    users = data["users"]
    locations = data["locations"]
    sms_list = data["sms"]
    mails = data["mails"]
    audio_files = data["audio_files"]
    data_dir = data["data_dir"]

    print("  Building user index...")
    user_idx = build_user_index(users)

    print("  Building message index...")
    msg_idx = build_message_index(sms_list, mails, user_idx)

    print("  Building location index...")
    loc_idx = build_location_index(locations)

    print("  Building transaction history...")
    txn_history = build_transaction_history(transactions)

    print("  Processing audio...")
    audio_cache = data_dir.parent / "evidence" / f"{data_dir.name}_audio.json"
    audio_evidence = transcribe_audio_files(audio_files, audio_cache, whisper_model)
    audio_evidence = link_audio_to_user(audio_evidence, user_idx)

    print("  Building evidence bundles for each transaction...")
    all_evidence = []

    for i, txn in enumerate(transactions):
        tid = txn["transaction_id"]
        txn_dt = txn["timestamp_dt"]
        sender_id = txn["sender_id"]
        recipient_id = txn.get("recipient_id", "")

        # Resolve sender
        sender_user = resolve_entity(sender_id, user_idx)
        if not sender_user and txn.get("sender_iban"):
            sender_user = resolve_by_iban(txn["sender_iban"], user_idx)

        # Resolve recipient
        recipient_user = resolve_entity(recipient_id, user_idx)
        if not recipient_user and txn.get("recipient_iban"):
            recipient_user = resolve_by_iban(txn["recipient_iban"], user_idx)

        # Check IBAN consistency
        iban_mismatch = False
        if sender_user and txn.get("sender_iban"):
            if txn["sender_iban"] != sender_user["iban"]:
                iban_mismatch = True

        # Location check
        loc_result = check_location_plausibility(
            sender_id, txn.get("location", ""), txn_dt, sender_user, loc_idx
        )

        # Message signals
        msg_signals = get_messages_near_time(msg_idx, sender_user, txn_dt, window_days=14)

        # Audio signals
        audio_nearby = get_audio_near_transaction(audio_evidence, sender_user, txn_dt, window_hours=72)

        # History
        hist = txn_history.get(tid, {})

        # Amount anomaly
        amount = txn["amount"]
        avg = hist.get("avg_amount", 0)
        max_amt = hist.get("max_amount", 0)
        std_amt = hist.get("std_amount", 0)

        amount_anomaly = 0
        if hist.get("txn_count", 0) >= 3:
            if std_amt > 0:
                z_score = (amount - avg) / std_amt
                amount_anomaly = max(0, z_score - 1) / 5  # normalize to 0-1
            if max_amt > 0 and amount > max_amt * 1.5:
                amount_anomaly = max(amount_anomaly, 0.5)
            if amount > avg * 5 and avg > 0:
                amount_anomaly = max(amount_anomaly, 0.7)

        # Time anomaly (late night transactions)
        hour = txn_dt.hour
        time_anomaly = 0.3 if (hour >= 0 and hour <= 5) else 0

        # Balance anomaly
        balance_anomaly = 0
        if txn["balance_after"] < 0:
            balance_anomaly = 0.5
        if sender_user and txn["balance_after"] < 0:
            balance_anomaly = 0.7

        # Salary ratio
        salary_ratio = 0
        if sender_user and sender_user.get("salary", 0) > 0:
            monthly_salary = sender_user["salary"] / 12
            if amount > monthly_salary * 2:
                salary_ratio = min(1.0, amount / (monthly_salary * 4))

        # Is routine? (salary, rent, etc.)
        desc = (txn.get("description") or "").lower()
        is_routine = any(kw in desc for kw in [
            "salary", "rent", "payroll", "insurance", "subscription",
            "utility", "phone", "internet", "pension",
        ]) if desc else False

        # Sender resolution quality
        sender_resolved = sender_user is not None
        sender_is_employer = sender_id.startswith("EMP") if sender_id else False

        evidence = {
            "transaction_id": tid,
            "sender_id": sender_id,
            "recipient_id": recipient_id or "",
            "transaction_type": txn["transaction_type"],
            "amount": amount,
            "location": txn.get("location", ""),
            "payment_method": txn.get("payment_method", ""),
            "balance_after": txn["balance_after"],
            "description": txn.get("description", ""),
            "timestamp": txn["timestamp"],
            "sender_resolved": sender_resolved,
            "sender_name": sender_user["_norm_full"] if sender_user else "",
            "sender_city": sender_user["_norm_city"] if sender_user else "",
            "sender_is_employer": sender_is_employer,
            "recipient_resolved": recipient_user is not None,
            "recipient_name": recipient_user["_norm_full"] if recipient_user else "",
            "iban_mismatch": iban_mismatch,
            "location_plausible": loc_result["plausible"],
            "location_reason": loc_result["reason"],
            "msg_suspicious_count": msg_signals["suspicious"],
            "msg_legit_count": msg_signals["legit"],
            "msg_recent_suspicious": msg_signals["recent_suspicious_texts"],
            "audio_nearby": audio_nearby,
            "hist_txn_count": hist.get("txn_count", 0),
            "hist_avg_amount": round(hist.get("avg_amount", 0), 2),
            "hist_max_amount": round(hist.get("max_amount", 0), 2),
            "is_new_recipient": hist.get("is_new_recipient", False),
            "is_new_method": hist.get("is_new_method", False),
            "amount_anomaly": round(amount_anomaly, 3),
            "time_anomaly": round(time_anomaly, 3),
            "balance_anomaly": round(balance_anomaly, 3),
            "salary_ratio": round(salary_ratio, 3),
            "is_routine": is_routine,
        }

        all_evidence.append(evidence)

        if (i + 1) % 200 == 0:
            print(f"    Processed {i+1}/{len(transactions)} transactions")

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_evidence, f, indent=2, default=str)

    print(f"  Built and cached {len(all_evidence)} evidence bundles")
    return all_evidence


def compute_risk_score(ev: dict) -> float:
    """Compute fast risk score from evidence. Higher = more suspicious."""
    score = 0.0

    # Baseline - employer/salary transactions are clean
    if ev["sender_is_employer"]:
        return 0.0
    if ev["is_routine"]:
        return 0.05

    # IBAN mismatch is very suspicious
    if ev["iban_mismatch"]:
        score += 0.30

    # Location implausibility
    if not ev["location_plausible"]:
        score += 0.25

    # Amount anomaly
    score += ev["amount_anomaly"] * 0.25

    # Time anomaly (late night)
    score += ev["time_anomaly"] * 0.15

    # New recipient with high amount
    if ev["is_new_recipient"] and ev["amount"] > 500:
        score += 0.15
    elif ev["is_new_recipient"]:
        score += 0.08

    # Suspicious messages nearby
    if ev["msg_suspicious_count"] >= 3:
        score += 0.25
    elif ev["msg_suspicious_count"] >= 1:
        score += 0.12

    # Audio evidence
    if ev["audio_nearby"]:
        for aud in ev["audio_nearby"]:
            if aud.get("is_suspicious"):
                score += 0.20
                break

    # Balance anomaly
    score += ev["balance_anomaly"] * 0.15

    # Salary ratio
    if ev["salary_ratio"] > 0.5:
        score += 0.10

    # Sender not resolved (could be spoofed)
    if not ev["sender_resolved"] and not ev["sender_is_employer"]:
        score += 0.10

    # New payment method
    if ev["is_new_method"]:
        score += 0.05

    # e-commerce + new recipient + high amount
    if ev["transaction_type"] == "e-commerce" and ev["is_new_recipient"] and ev["amount"] > 200:
        score += 0.10

    # High value transaction with any suspicious signal
    if ev["amount"] > 1000 and score > 0.2:
        score += 0.10

    return min(1.0, score)


def screen_transactions(evidence_list: list[dict], threshold_llm: float = 0.35,
                         threshold_auto: float = 0.80) -> tuple[list[dict], list[str], list[str]]:
    """Screen all transactions and classify into three buckets.

    Returns:
        shortlist: evidence dicts to send to LLM
        auto_fraud: transaction IDs auto-flagged as fraud
        auto_clean: transaction IDs auto-cleared
    """
    shortlist = []
    auto_fraud = []
    auto_clean = []

    for ev in evidence_list:
        score = compute_risk_score(ev)
        ev["risk_score"] = round(score, 3)

        if score >= threshold_auto:
            auto_fraud.append(ev["transaction_id"])
        elif score >= threshold_llm:
            shortlist.append(ev)
        else:
            auto_clean.append(ev["transaction_id"])

    print(f"  Screening results: {len(auto_fraud)} auto-fraud, {len(shortlist)} for LLM, {len(auto_clean)} auto-clean")
    return shortlist, auto_fraud, auto_clean
