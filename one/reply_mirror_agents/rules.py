from __future__ import annotations

from models import EvidenceBundle, RiskAssessment
from utils import normalize_text


def score_rule_based_risk(
    evidence: EvidenceBundle,
    amount_high_impact_floor: float,
    amount_high_impact_monthly_salary_factor: float,
) -> RiskAssessment:
    transaction = evidence.transaction
    focal_user = evidence.focal_user
    baseline = evidence.behavior_baseline
    location_context = evidence.location_context
    message_context = evidence.message_context

    fraud_signals: list[str] = []
    legitimacy_signals: list[str] = []
    uncertainties: list[str] = []
    risk_score = 0.18

    monthly_salary = (focal_user.salary / 12.0) if focal_user else 0.0
    economic_threshold = max(
        amount_high_impact_floor,
        monthly_salary * amount_high_impact_monthly_salary_factor,
    )
    economic_high_impact = transaction.amount >= economic_threshold

    if baseline.get("history_count", 0) < 3:
        uncertainties.append("Limited historical behavior before this transaction")

    if baseline.get("is_new_counterparty"):
        risk_score += 0.12
        fraud_signals.append("Counterparty or merchant is new for the user")
    else:
        legitimacy_signals.append("Counterparty or merchant has prior history")

    if baseline.get("is_unusual_amount"):
        risk_score += 0.16
        fraud_signals.append(
            f"Amount {transaction.amount:.2f} is materially above the user's historical norm"
        )
    elif baseline.get("history_count", 0) >= 5:
        legitimacy_signals.append("Amount is within the user's normal historical range")

    if baseline.get("is_unusual_hour"):
        risk_score += 0.07
        fraud_signals.append("Transaction happened at an unusual hour for this user")

    if baseline.get("is_new_payment_method"):
        risk_score += 0.08
        fraud_signals.append("Payment method is new for the user")

    if baseline.get("is_new_transaction_type"):
        risk_score += 0.07
        fraud_signals.append("Transaction type is unusual for the user")

    if location_context.get("relevance") == "high":
        if location_context.get("plausible") is False:
            risk_score += 0.22
            fraud_signals.extend(location_context.get("fraud_signals", []))
        elif location_context.get("plausible") is True:
            legitimacy_signals.extend(location_context.get("legitimacy_signals", []))
        else:
            uncertainties.extend(location_context.get("uncertainties", []))

    if message_context.get("recent_suspicious_count", 0) > 0:
        risk_score += min(0.22, 0.09 * message_context["recent_suspicious_count"])
        fraud_signals.extend(message_context.get("fraud_signals", []))

    if message_context.get("recent_legitimate_count", 0) > 0:
        risk_score -= min(0.18, 0.08 * message_context["recent_legitimate_count"])
        legitimacy_signals.extend(message_context.get("legitimacy_signals", []))

    if baseline.get("routine_match"):
        risk_score -= 0.18
        legitimacy_signals.append("Transaction matches a recurring routine pattern")

    description_text = normalize_text(transaction.description)
    if "salary payment" in description_text:
        risk_score -= 0.16
        legitimacy_signals.append("Incoming salary payment pattern strongly suggests legitimacy")
    if "rent payment" in description_text:
        risk_score -= 0.11
        legitimacy_signals.append("Rent payment wording matches a routine household transfer")

    if transaction.transaction_type == "withdrawal" and transaction.amount >= 200:
        risk_score += 0.06
        fraud_signals.append("Cash withdrawal deserves added scrutiny because funds become harder to recover")

    if economic_high_impact and risk_score >= 0.40:
        risk_score += 0.08
        fraud_signals.append("High-value suspicious transaction receives economic-impact escalation")
    elif economic_high_impact:
        uncertainties.append("High-value transaction raises the cost of a false negative")

    risk_score = max(0.01, min(0.99, risk_score))

    if not fraud_signals:
        uncertainties.append("No single dominant fraud signal was found; decision depends on weak combined evidence")
    if not legitimacy_signals:
        uncertainties.append("There is limited direct evidence supporting legitimacy")

    return RiskAssessment(
        risk_score=risk_score,
        fraud_signals=fraud_signals,
        legitimacy_signals=legitimacy_signals,
        uncertainties=uncertainties,
        economic_high_impact=economic_high_impact,
        economic_amount_threshold=economic_threshold,
    )