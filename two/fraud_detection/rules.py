"""Rule-based risk scoring using multiple weak signals."""

from typing import Any


def score_rule_based_risk(evidence: dict[str, Any]) -> dict[str, Any]:
    """Combine interpretable fraud signals into a risk score."""
    fraud_signals: list[str] = []
    legitimacy_signals: list[str] = []
    uncertainties: list[str] = []
    score = 0.0
    weights_sum = 0.0

    txn = evidence.get("transaction", {})
    parties = evidence.get("parties", {})
    location = evidence.get("location_context", {})
    messages = evidence.get("message_context", {})
    behavior = evidence.get("behavior_baseline", {})
    amount = txn.get("amount", 0)
    sender_info = parties.get("sender_info")

    # --- Signal 1: Sender not resolved (unknown sender) ---
    if not parties.get("sender_resolved") and not parties.get("sender_is_employer"):
        score += 0.3
        weights_sum += 1.0
        fraud_signals.append("Sender could not be linked to any known user")
    else:
        weights_sum += 1.0

    # --- Signal 2: IBAN mismatch ---
    if parties.get("sender_resolved") and txn.get("sender_iban"):
        if not parties.get("iban_matches_user"):
            score += 0.15
            weights_sum += 0.5
            fraud_signals.append("Sender IBAN does not match user's known IBAN")
        else:
            weights_sum += 0.5
            legitimacy_signals.append("Sender IBAN matches user's known IBAN")

    # --- Signal 3: Location inconsistency ---
    if location.get("has_transaction_location"):
        if location.get("location_plausible") is False:
            score += 0.25
            weights_sum += 1.0
            fraud_signals.append(
                f"Transaction location '{txn.get('location', '')}' inconsistent with user's actual location"
            )
        elif location.get("location_plausible") is True:
            weights_sum += 1.0
            legitimacy_signals.append("Transaction location consistent with user's known locations")
        else:
            weights_sum += 0.5
            uncertainties.append("Could not verify location consistency")

    # --- Signal 4: Amount anomaly ---
    if behavior.get("amount_is_anomalous"):
        ratio = behavior.get("amount_ratio_to_mean")
        score += 0.2
        weights_sum += 1.0
        fraud_signals.append(
            f"Amount {amount} is anomalous (ratio to mean: {ratio}x)"
        )
    elif behavior.get("amount_is_anomalous") is False:
        weights_sum += 1.0
        legitimacy_signals.append("Amount is within normal range for this user")
    elif behavior.get("total_prior_transactions", 0) == 0:
        weights_sum += 0.5
        uncertainties.append("No prior transaction history to compare amount")

    # --- Signal 5: Suspicious messages nearby ---
    if messages.get("has_phishing_indicators"):
        score += 0.25
        weights_sum += 1.0
        fraud_signals.append("Phishing/security alert messages found near transaction time")
    elif messages.get("has_security_alerts"):
        score += 0.15
        weights_sum += 0.8
        fraud_signals.append("Security alert messages found near transaction time")
    elif messages.get("suspicious_message_count", 0) > 0:
        score += 0.1
        weights_sum += 0.5
        fraud_signals.append("Some suspicious messages found near transaction time")

    if messages.get("legitimate_message_count", 0) > 0:
        legitimacy_signals.append("Legitimate messages (receipts/confirmations) found nearby")

    # --- Signal 6: New recipient ---
    if behavior.get("new_recipient") is True:
        if amount > 500:
            score += 0.15
            weights_sum += 0.7
            fraud_signals.append(f"New recipient with high amount ({amount})")
        else:
            score += 0.05
            weights_sum += 0.3
            uncertainties.append("New recipient but low amount")
    elif behavior.get("new_recipient") is False:
        weights_sum += 0.5
        legitimacy_signals.append("Recipient is a known contact/merchant")

    # --- Signal 7: Unusual time ---
    if behavior.get("time_is_unusual"):
        score += 0.1
        weights_sum += 0.5
        fraud_signals.append("Transaction at unusual time of day for this user")
    elif behavior.get("time_is_unusual") is False:
        weights_sum += 0.5
        legitimacy_signals.append("Transaction time is consistent with user's pattern")

    # --- Signal 8: Transaction type / description analysis ---
    desc = txn.get("description", "").lower()
    if any(kw in desc for kw in ["salary", "rent payment", "utility", "insurance", "subscription"]):
        legitimacy_signals.append(f"Description suggests routine transaction: '{txn.get('description', '')}'")
        score -= 0.05

    # --- Signal 9: Employer as sender (salary) ---
    if parties.get("sender_is_employer"):
        legitimacy_signals.append("Sender appears to be an employer (salary payment)")
        score -= 0.1

    # --- Signal 10: Balance check ---
    if txn.get("balance_after", 0) < 0:
        score += 0.1
        weights_sum += 0.5
        fraud_signals.append("Balance after transaction is negative")

    # --- Signal 11: Amount vs salary ---
    if sender_info and sender_info.get("salary"):
        monthly_salary = sender_info["salary"] / 12
        if amount > monthly_salary * 2:
            score += 0.15
            weights_sum += 0.7
            fraud_signals.append(
                f"Amount ({amount}) exceeds 2x monthly salary ({monthly_salary:.0f})"
            )
        elif amount > monthly_salary:
            score += 0.05
            weights_sum += 0.3
            uncertainties.append(f"Amount ({amount}) exceeds monthly salary ({monthly_salary:.0f})")

    # --- Signal 12: New location ---
    if behavior.get("new_location") is True:
        score += 0.1
        weights_sum += 0.5
        fraud_signals.append(f"Transaction at new location: '{txn.get('location', '')}'")

    # --- Signal 13: Withdrawal at unusual place ---
    if txn.get("transaction_type") == "withdrawal":
        if location.get("location_plausible") is False:
            score += 0.15
            weights_sum += 0.5
            fraud_signals.append("Withdrawal at inconsistent location")

    # --- Signal 14: E-commerce with location mismatch ---
    if txn.get("transaction_type") == "e-commerce" and location.get("has_transaction_location"):
        # E-commerce with physical location is unusual
        loc_str = txn.get("location", "")
        if loc_str and "marketplace" not in loc_str.lower() and "online" not in loc_str.lower():
            # Some e-commerce transactions show merchant name as location, which is fine
            pass

    # Normalize score
    risk_score = max(0.0, min(1.0, score))

    # Economic importance
    economic_high_impact = amount > 1000
    if sender_info and sender_info.get("salary"):
        economic_high_impact = economic_high_impact or (amount > sender_info["salary"] / 12)

    return {
        "risk_score": round(risk_score, 3),
        "fraud_signals": fraud_signals,
        "legitimacy_signals": legitimacy_signals,
        "uncertainties": uncertainties,
        "economic_high_impact": economic_high_impact,
        "economic_amount": amount,
    }
