from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from email import policy
from email.parser import Parser
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ISO_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%a, %d %b %Y %H:%M:%S %z",
)

URL_PATTERN = re.compile(r"https?://[^\s'\">]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    for fmt in ISO_DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def safe_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    collapsed = re.sub(r"\s+", " ", unescape(value))
    return collapsed.strip().lower()


def normalize_token(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(value))


def strip_html(value: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", value, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def parse_email_thread(raw: str) -> dict[str, Any]:
    message = Parser(policy=policy.default).parsestr(raw)
    payload = message.get_body(preferencelist=("plain", "html"))
    body = payload.get_content() if payload is not None else message.get_payload()
    if isinstance(body, list):
        body = "\n".join(str(part) for part in body)
    body_text = strip_html(str(body or ""))
    return {
        "from": str(message.get("From", "")),
        "to": str(message.get("To", "")),
        "subject": str(message.get("Subject", "")),
        "date": parse_datetime(str(message.get("Date", ""))),
        "body": body_text,
    }


def extract_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


def extract_domains(text: str) -> list[str]:
    domains = []
    for url in extract_urls(text):
        domain = urlparse(url).netloc.lower()
        if domain:
            domains.append(domain)
    for domain in EMAIL_PATTERN.findall(text):
        if domain:
            domains.append(domain.lower())
    return sorted(set(domains))


def looks_suspicious_domain(domain: str) -> bool:
    suspicious_markers = (
        "bit.ly",
        "tinyurl",
        "verify",
        "wallet",
        "secure-login",
        "support-check",
        "paypa1",
        "amaz0n",
        "micr0soft",
    )
    domain_text = domain.lower()
    return any(marker in domain_text for marker in suspicious_markers)


def extract_city_from_location(location: str | None) -> str | None:
    if not location:
        return None
    parts = [part.strip() for part in location.split(" - ") if part.strip()]
    if not parts:
        return None
    first = parts[0]
    if "online" in first.lower() or "marketplace" in first.lower():
        return None
    return first


def merchant_hint(location: str, description: str, transaction_type: str) -> str:
    if location:
        return location
    if description:
        return description
    return transaction_type


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_json_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fence_match = JSON_FENCE_PATTERN.search(candidate)
    if fence_match:
        candidate = fence_match.group(1)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(candidate[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start : index + 1])
    raise ValueError("Could not recover JSON payload from model output")


def ascii_lines(lines: list[str]) -> str:
    return "\n".join(line.encode("ascii", "ignore").decode("ascii") for line in lines)


def make_session_id(team_name: str, suffix: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    clean_team = normalize_token(team_name) or "replymirror"
    clean_suffix = normalize_token(suffix) or "run"
    return f"{clean_team}-{clean_suffix}-{timestamp}"


def stable_counterparty_key(
    *,
    sender_iban: str,
    recipient_iban: str,
    sender_id: str,
    recipient_id: str,
    location: str,
    description: str,
    focal_role: str | None,
) -> str:
    if focal_role == "sender":
        return recipient_iban or recipient_id or location or description or "unknown"
    if focal_role == "recipient":
        return sender_iban or sender_id or description or "unknown"
    return recipient_iban or sender_iban or recipient_id or sender_id or location or description or "unknown"


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))