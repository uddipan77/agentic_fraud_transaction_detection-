"""Utility functions."""

import json
import math
import re
from typing import Any


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lng points in kilometers."""
    R = 6371.0
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def extract_json_payload(text: str) -> dict[str, Any]:
    """Extract the first JSON object from text that may contain extra content."""
    # Try direct parse
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find JSON block in markdown code fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None
                    continue

    raise ValueError("No valid JSON object found in response")


def parse_freeform_decision(text: str) -> tuple[str | None, float | None]:
    """Try to extract a fraud/not_fraud decision from freeform text."""
    text_lower = text.lower()

    decision = None
    if "not_fraud" in text_lower or "not fraud" in text_lower or "legitimate" in text_lower:
        decision = "not_fraud"
    elif "fraud" in text_lower:
        decision = "fraud"

    confidence = None
    m = re.search(r"confidence[:\s]*([0-9]*\.?[0-9]+)", text_lower)
    if m:
        try:
            confidence = float(m.group(1))
            if confidence > 1:
                confidence = confidence / 100
        except ValueError:
            pass

    return decision, confidence


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def truncate_text(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
