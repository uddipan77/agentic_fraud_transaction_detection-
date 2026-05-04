"""Tools for extracting evidence from data files."""

from datetime import datetime
from typing import Any

from .data_loader import Transaction, User, Dataset
from .linking import DataLinker
from .utils import haversine_km, truncate_text


def get_transaction_context(txn: Transaction) -> dict[str, Any]:
    """Return structured transaction details."""
    return {
        "transaction_id": txn.transaction_id,
        "sender_id": txn.sender_id,
        "recipient_id": txn.recipient_id,
        "transaction_type": txn.transaction_type,
        "amount": txn.amount,
        "location": txn.location,
        "payment_method": txn.payment_method,
        "sender_iban": txn.sender_iban,
        "recipient_iban": txn.recipient_iban,
        "balance_after": txn.balance_after,
        "description": txn.description,
        "timestamp": txn.timestamp,
    }


def resolve_transaction_parties(
    txn: Transaction, linker: DataLinker
) -> dict[str, Any]:
    """Link sender and recipient to known users."""
    sender = linker.resolve_sender_user(txn)
    recipient = linker.resolve_recipient_user(txn)

    result: dict[str, Any] = {
        "sender_resolved": False,
        "sender_info": None,
        "recipient_resolved": False,
        "recipient_info": None,
        "sender_is_employer": txn.sender_id.startswith("EMP"),
        "iban_matches_user": False,
    }

    if sender:
        result["sender_resolved"] = True
        result["sender_info"] = {
            "full_name": f"{sender.first_name} {sender.last_name}",
            "salary": sender.salary,
            "job": sender.job,
            "residence_city": sender.residence_city,
            "iban": sender.iban,
        }
        # Check if sender IBAN in the transaction matches the user's known IBAN
        if txn.sender_iban:
            result["iban_matches_user"] = txn.sender_iban == sender.iban

    if recipient:
        result["recipient_resolved"] = True
        result["recipient_info"] = {
            "full_name": f"{recipient.first_name} {recipient.last_name}",
            "residence_city": recipient.residence_city,
            "iban": recipient.iban,
        }

    return result


def get_location_context(
    txn: Transaction, sender: User | None, linker: DataLinker, window_hours: float = 24.0
) -> dict[str, Any]:
    """Check if transaction location is consistent with user's actual location."""
    result: dict[str, Any] = {
        "has_transaction_location": bool(txn.location),
        "transaction_location": txn.location,
        "nearby_user_locations": [],
        "location_plausible": None,
        "min_distance_km": None,
        "closest_location_city": None,
        "user_residence_city": None,
        "location_in_residence_city": None,
    }

    if sender:
        result["user_residence_city"] = sender.residence_city
        nearby = linker.get_user_locations_near_time(sender, txn.timestamp, window_hours)
        result["nearby_user_locations"] = nearby[:10]  # limit for prompt size

        if txn.location and nearby:
            # Check if transaction location name contains user's city
            txn_loc_lower = txn.location.lower()
            res_city_lower = sender.residence_city.lower()
            result["location_in_residence_city"] = res_city_lower in txn_loc_lower

            # Find closest user location to residence
            closest_dist = None
            closest_city = None
            for loc in nearby:
                dist = haversine_km(
                    sender.residence_lat, sender.residence_lng,
                    loc["lat"], loc["lng"]
                )
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist
                    closest_city = loc["city"]

            result["min_distance_km"] = round(closest_dist, 2) if closest_dist is not None else None
            result["closest_location_city"] = closest_city

            # Check if transaction mentions a city that's far from where user was
            if nearby:
                user_cities = set(loc["city"].lower() for loc in nearby if loc.get("city"))
                # Transaction location may contain city name
                txn_city_match = any(c in txn_loc_lower for c in user_cities)
                result["location_plausible"] = txn_city_match or (result.get("location_in_residence_city", False))
        elif not txn.location:
            # No location for this transaction type (transfer, etc.)
            result["location_plausible"] = None  # Not applicable

    return result


def classify_message(text: str) -> str:
    """Classify a message as suspicious, legitimate, or neutral."""
    text_lower = text.lower()
    suspicious_keywords = [
        "verify your account", "urgent", "suspicious activity", "click here",
        "reset your password", "confirm your identity", "unauthorized",
        "security alert", "phishing", "compromised", "locked",
        "verify immediately", "act now", "unusual activity", "expire",
        "suspended", "validate", "confirm your details",
    ]
    legitimate_keywords = [
        "order confirmation", "receipt", "invoice", "payment received",
        "salary", "statement", "appointment", "reminder", "community",
        "newsletter", "renewal", "membership", "welcome", "thank you",
    ]

    sus_count = sum(1 for kw in suspicious_keywords if kw in text_lower)
    leg_count = sum(1 for kw in legitimate_keywords if kw in text_lower)

    if sus_count >= 2:
        return "suspicious"
    elif sus_count == 1 and leg_count == 0:
        return "suspicious"
    elif leg_count >= 1:
        return "legitimate"
    return "neutral"


def get_message_context(
    sender: User | None, txn: Transaction, linker: DataLinker, lookback_days: int = 7
) -> dict[str, Any]:
    """Find relevant SMS and emails near the transaction time."""
    result: dict[str, Any] = {
        "recent_sms": [],
        "recent_mails": [],
        "suspicious_message_count": 0,
        "legitimate_message_count": 0,
        "has_security_alerts": False,
        "has_phishing_indicators": False,
    }

    if not sender:
        return result

    sms_list = linker.get_user_sms_near_time(sender, txn.timestamp, lookback_days)
    for sms in sms_list[:8]:
        classification = classify_message(sms.get("message", ""))
        entry = {
            "from": sms.get("from", ""),
            "date": sms.get("date", ""),
            "message_preview": truncate_text(sms.get("message", ""), 200),
            "classification": classification,
        }
        result["recent_sms"].append(entry)
        if classification == "suspicious":
            result["suspicious_message_count"] += 1
            if any(kw in sms.get("message", "").lower() for kw in ["security alert", "suspicious activity", "unauthorized"]):
                result["has_security_alerts"] = True
            if any(kw in sms.get("message", "").lower() for kw in ["verify", "click here", "confirm your", "phishing"]):
                result["has_phishing_indicators"] = True
        elif classification == "legitimate":
            result["legitimate_message_count"] += 1

    mails = linker.get_user_mails_near_time(sender, txn.timestamp, lookback_days)
    for mail in mails[:5]:
        body_preview = mail.get("body_preview", "")
        classification = classify_message(body_preview)
        entry = {
            "from": mail.get("from", ""),
            "subject": mail.get("subject", ""),
            "date": mail.get("date", ""),
            "classification": classification,
        }
        result["recent_mails"].append(entry)
        if classification == "suspicious":
            result["suspicious_message_count"] += 1
            result["has_phishing_indicators"] = True
        elif classification == "legitimate":
            result["legitimate_message_count"] += 1

    return result


def get_behavior_baseline(
    sender_id: str, txn: Transaction, linker: DataLinker
) -> dict[str, Any]:
    """Summarize user's historical transaction behavior before this transaction."""
    history = linker.get_user_transaction_history(sender_id, txn.timestamp)

    result: dict[str, Any] = {
        "total_prior_transactions": len(history),
        "amount_stats": None,
        "usual_transaction_types": [],
        "usual_payment_methods": [],
        "usual_recipients": [],
        "usual_locations": [],
        "usual_hour_range": None,
        "amount_is_anomalous": None,
        "amount_ratio_to_max": None,
        "amount_ratio_to_mean": None,
        "new_recipient": None,
        "new_location": None,
        "time_is_unusual": None,
    }

    if not history:
        result["amount_is_anomalous"] = None
        result["new_recipient"] = True
        result["new_location"] = True if txn.location else None
        return result

    amounts = [t.amount for t in history]
    mean_amt = sum(amounts) / len(amounts)
    max_amt = max(amounts)
    min_amt = min(amounts)

    result["amount_stats"] = {
        "mean": round(mean_amt, 2),
        "max": round(max_amt, 2),
        "min": round(min_amt, 2),
        "count": len(amounts),
    }

    # Transaction types
    type_counts: dict[str, int] = {}
    for t in history:
        type_counts[t.transaction_type] = type_counts.get(t.transaction_type, 0) + 1
    result["usual_transaction_types"] = sorted(type_counts.keys())

    # Payment methods
    method_counts: dict[str, int] = {}
    for t in history:
        if t.payment_method:
            method_counts[t.payment_method] = method_counts.get(t.payment_method, 0) + 1
    result["usual_payment_methods"] = sorted(method_counts.keys())

    # Recipients
    recipient_set = set(t.recipient_id for t in history if t.recipient_id)
    result["usual_recipients"] = list(recipient_set)[:20]

    # Locations
    location_set = set(t.location for t in history if t.location)
    result["usual_locations"] = list(location_set)[:20]

    # Hours
    hours = []
    for t in history:
        try:
            dt = datetime.fromisoformat(t.timestamp)
            hours.append(dt.hour)
        except (ValueError, TypeError):
            pass
    if hours:
        result["usual_hour_range"] = {"min": min(hours), "max": max(hours)}

    # Anomaly checks
    if mean_amt > 0:
        result["amount_ratio_to_mean"] = round(txn.amount / mean_amt, 2)
    if max_amt > 0:
        result["amount_ratio_to_max"] = round(txn.amount / max_amt, 2)

    result["amount_is_anomalous"] = txn.amount > max_amt * 1.5 or txn.amount > mean_amt * 3

    # New recipient?
    result["new_recipient"] = txn.recipient_id not in recipient_set if txn.recipient_id else None

    # New location?
    if txn.location:
        result["new_location"] = txn.location not in location_set
    else:
        result["new_location"] = None

    # Unusual time?
    try:
        txn_hour = datetime.fromisoformat(txn.timestamp).hour
        if hours:
            result["time_is_unusual"] = txn_hour < min(hours) - 2 or txn_hour > max(hours) + 2
        else:
            result["time_is_unusual"] = None
    except (ValueError, TypeError):
        result["time_is_unusual"] = None

    return result
