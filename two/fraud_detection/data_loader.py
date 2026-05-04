"""Data loading utilities."""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Transaction:
    transaction_id: str
    sender_id: str
    recipient_id: str
    transaction_type: str
    amount: float
    location: str
    payment_method: str
    sender_iban: str
    recipient_iban: str
    balance_after: float
    description: str
    timestamp: str


@dataclass
class User:
    first_name: str
    last_name: str
    birth_year: int
    salary: float
    job: str
    iban: str
    residence_city: str
    residence_lat: float
    residence_lng: float
    description: str


@dataclass
class LocationRecord:
    biotag: str
    timestamp: str
    lat: float
    lng: float
    city: str


@dataclass
class Dataset:
    name: str
    transactions: list[Transaction] = field(default_factory=list)
    users: list[User] = field(default_factory=list)
    locations: list[LocationRecord] = field(default_factory=list)
    sms_messages: list[dict[str, str]] = field(default_factory=list)
    mail_messages: list[dict[str, str]] = field(default_factory=list)


def load_transactions(path: Path) -> list[Transaction]:
    txns = []
    with open(path / "transactions.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txns.append(Transaction(
                transaction_id=row["transaction_id"],
                sender_id=row["sender_id"],
                recipient_id=row["recipient_id"],
                transaction_type=row["transaction_type"],
                amount=float(row["amount"]) if row["amount"] else 0.0,
                location=row.get("location", ""),
                payment_method=row.get("payment_method", ""),
                sender_iban=row.get("sender_iban", ""),
                recipient_iban=row.get("recipient_iban", ""),
                balance_after=float(row["balance_after"]) if row.get("balance_after") else 0.0,
                description=row.get("description", ""),
                timestamp=row["timestamp"],
            ))
    return txns


def load_users(path: Path) -> list[User]:
    with open(path / "users.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    users = []
    for u in raw:
        res = u.get("residence", {})
        users.append(User(
            first_name=u["first_name"],
            last_name=u["last_name"],
            birth_year=int(u.get("birth_year", 0)),
            salary=float(u.get("salary", 0)),
            job=u.get("job", ""),
            iban=u.get("iban", ""),
            residence_city=res.get("city", ""),
            residence_lat=float(res.get("lat", 0)),
            residence_lng=float(res.get("lng", 0)),
            description=u.get("description", ""),
        ))
    return users


def load_locations(path: Path) -> list[LocationRecord]:
    with open(path / "locations.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [LocationRecord(
        biotag=r["biotag"],
        timestamp=r["timestamp"],
        lat=float(r["lat"]),
        lng=float(r["lng"]),
        city=r.get("city", ""),
    ) for r in raw]


def load_sms(path: Path) -> list[dict[str, str]]:
    with open(path / "sms.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_mails(path: Path) -> list[dict[str, str]]:
    with open(path / "mails.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path: Path, name: str) -> Dataset:
    return Dataset(
        name=name,
        transactions=load_transactions(path),
        users=load_users(path),
        locations=load_locations(path),
        sms_messages=load_sms(path),
        mail_messages=load_mails(path),
    )
