"""Audio transcription and evidence extraction using faster-whisper."""
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from .linking import normalize_str


def parse_audio_filename(filepath: Path) -> dict:
    """Extract timestamp and candidate user name from audio filename.

    Example: 20870418_094313-juliette_brunet.mp3
    -> timestamp: 2087-04-18T09:43:13, name: juliette brunet
    """
    stem = filepath.stem  # e.g. 20870418_094313-juliette_brunet
    result = {"file": filepath.name, "timestamp": None, "candidate_name": None}

    m = re.match(r"(\d{8})_(\d{6})-(.+)", stem)
    if m:
        date_s, time_s, name_s = m.groups()
        try:
            ts = datetime(
                int(date_s[:4]), int(date_s[4:6]), int(date_s[6:8]),
                int(time_s[:2]), int(time_s[2:4]), int(time_s[4:6])
            )
            result["timestamp"] = ts
        except ValueError:
            pass
        result["candidate_name"] = normalize_str(name_s.replace("_", " "))

    return result


FRAUD_AUDIO_KEYWORDS = [
    "verification", "verify", "transfer", "urgent", "payment",
    "otp", "one-time", "password", "pin", "security", "bank",
    "account", "locked", "suspended", "paypal", "amazon",
    "merchant", "card", "credit", "debit", "wire", "authorize",
    "confirm", "fraud", "unauthorized", "refund", "claim",
    "bitcoin", "crypto", "wallet", "hacked", "compromised",
    "phishing", "scam", "prize", "lottery", "won", "inheritance",
    "customs", "fee", "tax", "irs", "revenue", "police",
    "arrest", "warrant", "social security", "identity",
    "click", "link", "immediately", "deadline", "expire",
]


def transcribe_audio_files(audio_files: list[Path], cache_path: Path,
                            model_size: str = "base") -> list[dict]:
    """Transcribe audio files using faster-whisper. Results cached to disk.

    Returns list of audio evidence dicts.
    """
    # Check cache
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(audio_files):
            # Restore datetime objects
            for item in cached:
                if item.get("timestamp"):
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])
            print(f"  Loaded {len(cached)} cached audio transcriptions")
            return cached

    print(f"  Transcribing {len(audio_files)} audio files with faster-whisper ({model_size})...")

    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    results = []
    for i, fpath in enumerate(audio_files):
        meta = parse_audio_filename(fpath)

        # Transcribe
        try:
            segments, info = model.transcribe(str(fpath), beam_size=1, language=None)
            transcript = " ".join(seg.text.strip() for seg in segments)
        except Exception as e:
            transcript = f"[transcription error: {e}]"

        # Extract keywords
        transcript_lower = transcript.lower()
        keywords = [kw for kw in FRAUD_AUDIO_KEYWORDS if kw in transcript_lower]

        evidence = {
            "file": meta["file"],
            "timestamp": meta["timestamp"].isoformat() if meta["timestamp"] else None,
            "candidate_name": meta["candidate_name"],
            "transcript": transcript[:500],
            "keywords": keywords,
            "is_suspicious": len(keywords) >= 2,
            "language": getattr(info, "language", "unknown") if "info" in dir() else "unknown",
        }
        results.append(evidence)

        if (i + 1) % 10 == 0:
            print(f"    Transcribed {i+1}/{len(audio_files)}")

    # Cache results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Transcribed {len(results)} audio files, cached to {cache_path.name}")

    # Restore datetime for in-memory use
    for item in results:
        if item.get("timestamp"):
            item["timestamp"] = datetime.fromisoformat(item["timestamp"])

    return results


def link_audio_to_user(audio_evidence: list[dict], user_idx: dict) -> list[dict]:
    """Link audio evidence to users by candidate name matching."""
    for ev in audio_evidence:
        name = ev.get("candidate_name", "")
        if not name:
            continue

        parts = name.split()
        if len(parts) >= 2:
            fn = parts[0]
            users_match = user_idx.get("firstname_to_users", {}).get(fn, [])
            if len(users_match) == 1:
                ev["linked_user"] = users_match[0]["_norm_full"]
                ev["linked_iban"] = users_match[0]["iban"]
            elif len(users_match) > 1:
                # Try full name match
                full = " ".join(parts)
                for u in users_match:
                    if u["_norm_full"] == full:
                        ev["linked_user"] = u["_norm_full"]
                        ev["linked_iban"] = u["iban"]
                        break

    return audio_evidence


def get_audio_near_transaction(audio_evidence: list[dict], user: dict,
                                txn_dt: datetime, window_hours: int = 48) -> list[dict]:
    """Find audio evidence near a transaction time for a specific user."""
    if not user:
        return []

    user_name = user.get("_norm_full", "")
    results = []

    from datetime import timedelta
    window = timedelta(hours=window_hours)

    for ev in audio_evidence:
        # Match by user
        if ev.get("linked_user") != user_name and ev.get("candidate_name") != user.get("_norm_fn"):
            continue

        # Match by time
        ts = ev.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts and abs(txn_dt - ts) <= window:
            results.append({
                "file": ev["file"],
                "keywords": ev.get("keywords", []),
                "is_suspicious": ev.get("is_suspicious", False),
                "transcript_snippet": ev.get("transcript", "")[:200],
                "time_diff_hours": round(abs((txn_dt - ts).total_seconds()) / 3600, 1),
            })

    return results
