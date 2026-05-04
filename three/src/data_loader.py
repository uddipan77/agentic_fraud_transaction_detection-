"""Load all dataset files into structured Python objects."""
import json
import csv
from pathlib import Path
from datetime import datetime


def load_transactions(data_dir: Path) -> list[dict]:
    """Load transactions.csv into list of dicts with parsed types."""
    rows = []
    with open(data_dir / "transactions.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["amount"] = float(r["amount"]) if r["amount"] else 0.0
            r["balance_after"] = float(r["balance_after"]) if r["balance_after"] else 0.0
            r["timestamp_dt"] = datetime.fromisoformat(r["timestamp"])
            rows.append(r)
    return rows


def load_json(data_dir: Path, filename: str) -> list:
    with open(data_dir / filename, encoding="utf-8") as f:
        return json.load(f)


def load_all(data_dir: Path) -> dict:
    """Load all data files from a dataset directory."""
    transactions = load_transactions(data_dir)
    users = load_json(data_dir, "users.json")
    locations = load_json(data_dir, "locations.json")
    sms_list = load_json(data_dir, "sms.json")
    mails = load_json(data_dir, "mails.json")

    audio_dir = data_dir / "audio"
    audio_files = sorted(audio_dir.glob("*.mp3")) if audio_dir.exists() else []

    return {
        "transactions": transactions,
        "users": users,
        "locations": locations,
        "sms": sms_list,
        "mails": mails,
        "audio_files": audio_files,
        "data_dir": data_dir,
    }
