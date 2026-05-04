from __future__ import annotations

from models import TransactionRecord, UserProfile
from utils import extract_city_from_location, merchant_hint, stable_counterparty_key


def resolve_transaction_parties(
    transaction: TransactionRecord,
    users_by_iban: dict[str, UserProfile],
) -> dict[str, UserProfile | None]:
    sender_user = users_by_iban.get(transaction.sender_iban)
    recipient_user = users_by_iban.get(transaction.recipient_iban)
    return {
        "sender_user": sender_user,
        "recipient_user": recipient_user,
    }


def determine_focal_user(
    transaction: TransactionRecord,
    users_by_iban: dict[str, UserProfile],
) -> tuple[UserProfile | None, str | None, UserProfile | None]:
    parties = resolve_transaction_parties(transaction, users_by_iban)
    sender_user = parties["sender_user"]
    recipient_user = parties["recipient_user"]

    if sender_user and not recipient_user:
        return sender_user, "sender", None
    if recipient_user and not sender_user:
        return recipient_user, "recipient", None
    if sender_user and recipient_user:
        return sender_user, "sender", recipient_user
    return None, None, None


def describe_counterparty(
    transaction: TransactionRecord,
    focal_role: str | None,
    counterparty_user: UserProfile | None,
) -> str:
    if counterparty_user is not None:
        return counterparty_user.full_name
    if focal_role == "sender":
        return (
            transaction.recipient_id
            or transaction.recipient_iban
            or merchant_hint(
                transaction.location, transaction.description, transaction.transaction_type
            )
        )
    if focal_role == "recipient":
        return transaction.sender_id or transaction.sender_iban or transaction.description
    return merchant_hint(transaction.location, transaction.description, transaction.transaction_type)


def transaction_counterparty_key(
    transaction: TransactionRecord,
    focal_role: str | None,
) -> str:
    return stable_counterparty_key(
        sender_iban=transaction.sender_iban,
        recipient_iban=transaction.recipient_iban,
        sender_id=transaction.sender_id,
        recipient_id=transaction.recipient_id,
        location=transaction.location,
        description=transaction.description,
        focal_role=focal_role,
    )


def transaction_city(transaction: TransactionRecord) -> str | None:
    return extract_city_from_location(transaction.location)