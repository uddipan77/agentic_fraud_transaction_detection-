"""Linking logic: connect transactions, users, locations, SMS, and mails."""

import re
from datetime import datetime, timedelta
from typing import Any

from .data_loader import Dataset, Transaction, User, LocationRecord


def build_user_iban_index(dataset: Dataset) -> dict[str, User]:
    """Map IBAN -> User."""
    return {u.iban: u for u in dataset.users}


def build_user_name_index(dataset: Dataset) -> dict[str, User]:
    """Map 'FirstName LastName' -> User."""
    return {f"{u.first_name} {u.last_name}": u for u in dataset.users}


def build_biotag_to_user(dataset: Dataset) -> dict[str, User]:
    """Map biotag/sender_id -> User using city abbreviation matching."""
    biotags = set()
    for loc in dataset.locations:
        biotags.add(loc.biotag)
    for txn in dataset.transactions:
        if not txn.sender_id.startswith("EMP"):
            biotags.add(txn.sender_id)

    mapping = {}
    for bt in biotags:
        parts = bt.split("-")
        if len(parts) >= 4:
            city_abbr = parts[3].upper()
            for u in dataset.users:
                if u.residence_city[:3].upper() == city_abbr[:3].upper():
                    # Also verify last name initials match
                    ln_rev = u.last_name[:4].upper()
                    bt_ln = parts[0].upper()
                    if bt_ln[:2] == ln_rev[:2] or bt_ln[:3] == ln_rev[:3]:
                        mapping[bt] = u
                        break
            else:
                # Fallback: just city match
                for u in dataset.users:
                    if u.residence_city[:3].upper() == city_abbr[:3].upper():
                        mapping[bt] = u
                        break
    return mapping


def build_user_to_biotag(dataset: Dataset) -> dict[str, str]:
    """Map User IBAN -> biotag."""
    bt_to_user = build_biotag_to_user(dataset)
    return {u.iban: bt for bt, u in bt_to_user.items()}


def build_location_index(dataset: Dataset) -> dict[str, list[LocationRecord]]:
    """Map biotag -> list of location records sorted by timestamp."""
    index: dict[str, list[LocationRecord]] = {}
    for loc in dataset.locations:
        index.setdefault(loc.biotag, []).append(loc)
    for k in index:
        index[k].sort(key=lambda x: x.timestamp)
    return index


def build_transaction_index(dataset: Dataset) -> dict[str, list[Transaction]]:
    """Map sender_id -> list of transactions sorted by timestamp."""
    index: dict[str, list[Transaction]] = {}
    for txn in dataset.transactions:
        index.setdefault(txn.sender_id, []).append(txn)
    for k in index:
        index[k].sort(key=lambda x: x.timestamp)
    return index


def parse_sms_fields(sms_text: str) -> dict[str, str]:
    """Parse structured fields from an SMS text."""
    result = {"from": "", "to": "", "date": "", "message": "", "raw": sms_text}
    lines = sms_text.strip().split("\n")
    msg_lines = []
    in_message = False
    for line in lines:
        if line.startswith("From:") and not in_message:
            result["from"] = line[5:].strip()
        elif line.startswith("To:") and not in_message:
            result["to"] = line[3:].strip()
        elif line.startswith("Date:") and not in_message:
            result["date"] = line[5:].strip()
        elif line.startswith("Message:"):
            in_message = True
            result["message"] = line[8:].strip()
        elif in_message:
            msg_lines.append(line)
    if msg_lines:
        result["message"] += " " + " ".join(msg_lines)
    return result


def parse_mail_headers(mail_text: str) -> dict[str, str]:
    """Parse From, To, Subject, Date from a mail."""
    result = {"from": "", "to": "", "subject": "", "date": "", "to_name": "", "to_email": ""}
    for line in mail_text.split("\n")[:20]:
        if line.startswith("From:"):
            result["from"] = line[5:].strip()
        elif line.startswith("To:"):
            result["to"] = line[3:].strip()
            # Extract name and email
            m = re.match(r'"([^"]+)"\s*<([^>]+)>', result["to"])
            if m:
                result["to_name"] = m.group(1)
                result["to_email"] = m.group(2)
        elif line.startswith("Subject:"):
            result["subject"] = line[8:].strip()
        elif line.startswith("Date:"):
            result["date"] = line[5:].strip()
    return result


def build_sms_index_by_user(dataset: Dataset) -> dict[str, list[dict[str, str]]]:
    """Map user first_name -> list of parsed SMS mentioning them."""
    index: dict[str, list[dict[str, str]]] = {}
    parsed_all = [parse_sms_fields(s["sms"]) for s in dataset.sms_messages]

    for user in dataset.users:
        user_sms = []
        for parsed in parsed_all:
            if user.first_name in parsed["raw"]:
                user_sms.append(parsed)
        index[user.first_name] = user_sms
    return index


def build_mail_index_by_user(dataset: Dataset) -> dict[str, list[dict[str, str]]]:
    """Map user first_name -> list of parsed mails addressed to them."""
    index: dict[str, list[dict[str, str]]] = {}

    parsed_all = []
    for m in dataset.mail_messages:
        headers = parse_mail_headers(m["mail"])
        headers["body_preview"] = m["mail"][:1500]
        parsed_all.append(headers)

    for user in dataset.users:
        user_mails = []
        full_name = f"{user.first_name} {user.last_name}"
        for parsed in parsed_all:
            if full_name in parsed.get("to_name", "") or user.first_name in parsed.get("to", ""):
                user_mails.append(parsed)
        index[user.first_name] = user_mails
    return index


class DataLinker:
    """Central class holding all indexes for fast lookups."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.iban_to_user = build_user_iban_index(dataset)
        self.name_to_user = build_user_name_index(dataset)
        self.biotag_to_user = build_biotag_to_user(dataset)
        self.user_to_biotag = build_user_to_biotag(dataset)
        self.location_index = build_location_index(dataset)
        self.transaction_index = build_transaction_index(dataset)
        self.sms_by_user = build_sms_index_by_user(dataset)
        self.mail_by_user = build_mail_index_by_user(dataset)
        self.txn_by_id = {t.transaction_id: t for t in dataset.transactions}

    def resolve_sender_user(self, txn: Transaction) -> User | None:
        """Find the user who sent this transaction."""
        # Try biotag/sender_id match
        user = self.biotag_to_user.get(txn.sender_id)
        if user:
            return user
        # Try IBAN match
        if txn.sender_iban:
            user = self.iban_to_user.get(txn.sender_iban)
            if user:
                return user
        return None

    def resolve_recipient_user(self, txn: Transaction) -> User | None:
        """Find the user who received this transaction (if they're a known user)."""
        # Try biotag/recipient_id match
        user = self.biotag_to_user.get(txn.recipient_id)
        if user:
            return user
        # Try IBAN match
        if txn.recipient_iban:
            user = self.iban_to_user.get(txn.recipient_iban)
            if user:
                return user
        return None

    def get_user_locations_near_time(
        self, user: User, timestamp: str, window_hours: float = 24.0
    ) -> list[dict[str, Any]]:
        """Get location records for a user near a given timestamp."""
        biotag = self.user_to_biotag.get(user.iban)
        if not biotag:
            # Try finding biotag from biotag_to_user reverse
            for bt, u in self.biotag_to_user.items():
                if u.iban == user.iban:
                    biotag = bt
                    break
        if not biotag or biotag not in self.location_index:
            return []

        try:
            target_dt = datetime.fromisoformat(timestamp)
        except (ValueError, TypeError):
            return []

        window = timedelta(hours=window_hours)
        results = []
        for loc in self.location_index[biotag]:
            try:
                loc_dt = datetime.fromisoformat(loc.timestamp)
            except (ValueError, TypeError):
                continue
            if abs(loc_dt - target_dt) <= window:
                results.append({
                    "timestamp": loc.timestamp,
                    "lat": loc.lat,
                    "lng": loc.lng,
                    "city": loc.city,
                    "hours_from_txn": round((loc_dt - target_dt).total_seconds() / 3600, 2),
                })
        return results

    def get_user_sms_near_time(
        self, user: User, timestamp: str, lookback_days: int = 7
    ) -> list[dict[str, str]]:
        """Get SMS messages for a user near a timestamp."""
        sms_list = self.sms_by_user.get(user.first_name, [])
        if not sms_list:
            return []

        try:
            target_dt = datetime.fromisoformat(timestamp)
        except (ValueError, TypeError):
            return sms_list[:5]

        window = timedelta(days=lookback_days)
        results = []
        for sms in sms_list:
            date_str = sms.get("date", "")
            try:
                sms_dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue
            if abs(sms_dt - target_dt) <= window:
                results.append(sms)
        return results

    def get_user_mails_near_time(
        self, user: User, timestamp: str, lookback_days: int = 30
    ) -> list[dict[str, str]]:
        """Get mail messages for a user near a timestamp."""
        mails = self.mail_by_user.get(user.first_name, [])
        if not mails:
            return []
        # Mails have complex date formats; return all for now
        return mails

    def get_user_transaction_history(
        self, sender_id: str, before_timestamp: str
    ) -> list[Transaction]:
        """Get all transactions by this sender before the given timestamp."""
        txns = self.transaction_index.get(sender_id, [])
        try:
            target_dt = datetime.fromisoformat(before_timestamp)
        except (ValueError, TypeError):
            return txns

        return [t for t in txns if datetime.fromisoformat(t.timestamp) < target_dt]
