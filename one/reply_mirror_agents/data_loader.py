from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
from pathlib import Path

from models import LocationObservation, MessageRecord, TransactionRecord, UserProfile
from utils import (
    extract_domains,
    load_json_file,
    looks_suspicious_domain,
    normalize_text,
    normalize_token,
    parse_datetime,
    parse_email_thread,
    safe_float,
)


@dataclass(slots=True)
class DatasetBundle:
    name: str
    path: Path
    transactions: list[TransactionRecord]
    transactions_by_id: dict[str, TransactionRecord]
    users: list[UserProfile]
    users_by_iban: dict[str, UserProfile]
    users_by_biotag: dict[str, UserProfile]
    locations_by_biotag: dict[str, list[LocationObservation]]
    messages_by_user_iban: dict[str, list[MessageRecord]]
    user_transactions: dict[str, list[TransactionRecord]]


def _city_code(city: str) -> str:
    return normalize_token(city)[:3]


def _infer_biotag_for_user(user: UserProfile, biotags: list[str]) -> str | None:
    city_code = _city_code(user.residence_city)
    for biotag in biotags:
        parts = biotag.split("-")
        if len(parts) >= 4 and normalize_token(parts[3])[:3] == city_code:
            return biotag
    return None


def _classify_message(body: str, subject: str, sender: str, domains: list[str]) -> list[str]:
    text = normalize_text(" ".join([subject, body, sender, " ".join(domains)]))
    labels: list[str] = []

    phishing_terms = (
        "urgent",
        "verify",
        "account lock",
        "suspicious login",
        "unusual login",
        "avoid suspension",
        "confirm identity",
        "secure your account",
    )
    legitimate_terms = (
        "order confirmation",
        "invoice",
        "bill",
        "delivery update",
        "account statement",
        "monthly summary",
    )
    civic_terms = (
        "city hall",
        "council",
        "events",
        "maintenance",
        "community",
        "meet-up",
        "workshop",
        "alert",
    )

    civic_hint = any(term in text for term in civic_terms)
    suspicious_domain = any(looks_suspicious_domain(domain) for domain in domains)
    phishing_language = any(term in text for term in phishing_terms)

    if phishing_language or (suspicious_domain and not civic_hint):
        labels.append("suspicious/phishing")
    if ("login" in text or "security" in text or "verify" in text) and not civic_hint:
        labels.append("security alert")
    if any(term in text for term in legitimate_terms):
        labels.append("legitimate order/bill")
    if any(term in text for term in civic_terms):
        labels.append("civic/general")
    if not labels:
        labels.append("general")
    return sorted(set(labels))


def _parse_sms_record(raw: str) -> MessageRecord:
    sender = ""
    recipient = ""
    date_text = ""
    message = raw
    for line in raw.splitlines():
        if line.startswith("From:"):
            sender = line.split(":", 1)[1].strip()
        elif line.startswith("To:"):
            recipient = line.split(":", 1)[1].strip()
        elif line.startswith("Date:"):
            date_text = line.split(":", 1)[1].strip()
        elif line.startswith("Message:"):
            message = line.split(":", 1)[1].strip()
    domains = extract_domains(raw)
    return MessageRecord(
        channel="sms",
        sender=sender,
        recipient=recipient,
        subject="",
        timestamp=parse_datetime(date_text),
        body=message,
        raw=raw,
        classifications=_classify_message(message, "", sender, domains),
        domains=domains,
    )


def _parse_mail_record(raw: str) -> MessageRecord:
    parsed = parse_email_thread(raw)
    domains = extract_domains(raw)
    return MessageRecord(
        channel="mail",
        sender=parsed["from"],
        recipient=parsed["to"],
        subject=parsed["subject"],
        timestamp=parsed["date"],
        body=parsed["body"],
        raw=raw,
        classifications=_classify_message(
            parsed["body"], parsed["subject"], parsed["from"], domains
        ),
        domains=domains,
    )


def _message_matches_user(message: MessageRecord, user: UserProfile) -> bool:
    text = normalize_text(
        " ".join([message.recipient, message.subject, message.body, message.sender])
    )
    full_name = normalize_text(user.full_name)
    first_name = normalize_text(user.first_name)
    email_local = normalize_token(f"{user.first_name}.{user.last_name}")
    return (
        full_name in text
        or first_name in text
        or email_local in normalize_token(text)
        or normalize_text(user.residence_city) in text
    )


def load_dataset(path: Path, name: str | None = None) -> DatasetBundle:
    raw_users = load_json_file(path / "users.json")
    users = [
        UserProfile(
            first_name=item["first_name"],
            last_name=item["last_name"],
            birth_year=int(item["birth_year"]),
            salary=safe_float(item["salary"]),
            job=item["job"],
            iban=item["iban"],
            residence_city=item["residence"]["city"],
            residence_lat=safe_float(item["residence"]["lat"]),
            residence_lng=safe_float(item["residence"]["lng"]),
            description=item.get("description", ""),
        )
        for item in raw_users
    ]

    raw_locations = load_json_file(path / "locations.json")
    biotags = sorted({item["biotag"] for item in raw_locations})
    for user in users:
        user.biotag = _infer_biotag_for_user(user, biotags)

    locations_by_biotag: dict[str, list[LocationObservation]] = defaultdict(list)
    for item in raw_locations:
        observation = LocationObservation(
            biotag=item["biotag"],
            timestamp=parse_datetime(item["timestamp"]),
            lat=safe_float(item["lat"]),
            lng=safe_float(item["lng"]),
            city=item["city"],
        )
        locations_by_biotag[observation.biotag].append(observation)
    for observations in locations_by_biotag.values():
        observations.sort(key=lambda obs: obs.timestamp or 0)

    users_by_iban = {user.iban: user for user in users}
    users_by_biotag = {user.biotag: user for user in users if user.biotag}

    raw_sms = load_json_file(path / "sms.json")
    raw_mails = load_json_file(path / "mails.json")
    all_messages = [_parse_sms_record(item["sms"]) for item in raw_sms] + [
        _parse_mail_record(item["mail"]) for item in raw_mails
    ]
    all_messages.sort(key=lambda item: item.timestamp or parse_datetime("1970-01-01T00:00:00"))

    messages_by_user_iban: dict[str, list[MessageRecord]] = defaultdict(list)
    for user in users:
        for message in all_messages:
            if _message_matches_user(message, user):
                messages_by_user_iban[user.iban].append(message)

    transactions: list[TransactionRecord] = []
    with open(path / "transactions.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            transactions.append(
                TransactionRecord(
                    transaction_id=row["transaction_id"],
                    sender_id=row["sender_id"],
                    recipient_id=row["recipient_id"],
                    transaction_type=row["transaction_type"],
                    amount=safe_float(row["amount"]),
                    location=row.get("location", "") or "",
                    payment_method=row.get("payment_method", "") or "",
                    sender_iban=row.get("sender_iban", "") or "",
                    recipient_iban=row.get("recipient_iban", "") or "",
                    balance_after=safe_float(row.get("balance_after", "")),
                    description=row.get("description", "") or "",
                    timestamp=parse_datetime(row["timestamp"]),
                )
            )
    transactions.sort(key=lambda item: item.timestamp)

    transactions_by_id = {item.transaction_id: item for item in transactions}
    user_transactions: dict[str, list[TransactionRecord]] = defaultdict(list)
    for transaction in transactions:
        if transaction.sender_iban in users_by_iban:
            user_transactions[transaction.sender_iban].append(transaction)
        if transaction.recipient_iban in users_by_iban:
            user_transactions[transaction.recipient_iban].append(transaction)

    return DatasetBundle(
        name=name or path.name,
        path=path,
        transactions=transactions,
        transactions_by_id=transactions_by_id,
        users=users,
        users_by_iban=users_by_iban,
        users_by_biotag=users_by_biotag,
        locations_by_biotag=dict(locations_by_biotag),
        messages_by_user_iban=dict(messages_by_user_iban),
        user_transactions=dict(user_transactions),
    )