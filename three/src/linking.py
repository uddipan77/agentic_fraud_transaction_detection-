"""Link biotags, IBANs, names, SMS, mails, locations to users."""
import re
import unicodedata
from datetime import datetime, timedelta
from collections import defaultdict


def normalize_str(s: str) -> str:
    """Normalize Unicode and lowercase."""
    if not s:
        return ""
    # NFKD decompose then strip combining marks, lowercase
    n = unicodedata.normalize("NFKD", s)
    n = "".join(c for c in n if not unicodedata.combining(c))
    return n.lower().strip()


def build_user_index(users: list[dict]) -> dict:
    """Build lookup indexes from user list.

    Returns dict with:
      - iban_to_user: {iban -> user_dict}
      - name_to_user: {normalized_full_name -> user_dict}
      - firstname_to_users: {normalized_firstname -> [user_dict]}
      - city_to_users: {normalized_city -> [user_dict]}
      - users_list: original list with added norm fields
    """
    iban_to_user = {}
    name_to_user = {}
    firstname_to_users = defaultdict(list)
    city_to_users = defaultdict(list)

    for u in users:
        fn = u.get("first_name", "")
        ln = u.get("last_name", "")
        u["_norm_fn"] = normalize_str(fn)
        u["_norm_ln"] = normalize_str(ln)
        u["_norm_full"] = f"{u['_norm_fn']} {u['_norm_ln']}"
        u["_norm_city"] = normalize_str(u.get("residence", {}).get("city", ""))
        u["_res_lat"] = float(u.get("residence", {}).get("lat", 0))
        u["_res_lng"] = float(u.get("residence", {}).get("lng", 0))

        iban_to_user[u["iban"]] = u
        name_to_user[u["_norm_full"]] = u
        firstname_to_users[u["_norm_fn"]].append(u)
        city_to_users[u["_norm_city"]].append(u)

    return {
        "iban_to_user": iban_to_user,
        "name_to_user": name_to_user,
        "firstname_to_users": dict(firstname_to_users),
        "city_to_users": dict(city_to_users),
        "users_list": users,
    }


def resolve_biotag(biotag: str, user_idx: dict) -> dict | None:
    """Try to resolve a sender/recipient biotag to a user.

    Biotag pattern: LASTNAME_PREFIX-FIRSTNAME_PREFIX-HEX-CITY_ABBR-DIGIT
    Example: LDCK-SLVT-7F8-MUN-0 -> Salvatore Ladeck, Munich
    """
    if not biotag or biotag.startswith("EMP") or not "-" in biotag:
        return None

    parts = normalize_str(biotag).split("-")
    if len(parts) < 4:
        return None

    ln_prefix = parts[0][:3]  # first 3 chars of last name
    fn_prefix = parts[1][:3]  # first 3 chars of first name
    city_abbr = parts[3][:3]  # city abbreviation

    best_match = None
    best_score = 0

    for u in user_idx["users_list"]:
        score = 0
        nln = u["_norm_ln"]
        nfn = u["_norm_fn"]

        # Check last name prefix
        if nln[:3] == ln_prefix or nln[:4] == parts[0][:4]:
            score += 3
        elif nln[:2] == ln_prefix[:2]:
            score += 1

        # Check first name prefix
        if nfn[:3] == fn_prefix or nfn[:4] == parts[1][:4]:
            score += 3
        elif nfn[:2] == fn_prefix[:2]:
            score += 1

        # Check city
        ncity = u["_norm_city"]
        if ncity[:3] == city_abbr:
            score += 2

        if score > best_score:
            best_score = score
            best_match = u

    return best_match if best_score >= 5 else None


def resolve_entity(entity_id: str, user_idx: dict) -> dict | None:
    """Resolve any entity ID to user. Returns None for employers/merchants."""
    if not entity_id or entity_id != entity_id:  # NaN check
        return None
    entity_id = str(entity_id).strip()
    if entity_id.startswith("EMP") or not "-" in entity_id:
        return None
    return resolve_biotag(entity_id, user_idx)


def resolve_by_iban(iban: str, user_idx: dict) -> dict | None:
    """Resolve user by IBAN."""
    if not iban or iban != iban:
        return None
    return user_idx["iban_to_user"].get(str(iban).strip())


# ── SMS/Mail indexing ──────────────────────────────────────────────────

PHISHING_KEYWORDS = [
    "urgent", "verify your", "verify identity", "suspended", "blocked",
    "unauthorized", "security alert", "act now", "compromised", "locked",
    "confirm your", "unusual activity", "click here", "pay now",
    "release fee", "customs", "frozen", "restore access", "immediately",
    "within 24", "within 48", "expire", "limited time", "otp",
    "suspicious", "reset your password", "account will be",
]

LEGIT_KEYWORDS = [
    "order confirmation", "receipt", "salary", "invoice", "payment received",
    "appointment", "thank you for your purchase", "subscription", "welcome",
    "newsletter", "orientation", "reminder", "library", "student services",
    "boarding pass", "booking confirmed", "itinerary",
]


def classify_message(text: str) -> str:
    """Classify a message as 'suspicious', 'legit', or 'neutral'."""
    t = text.lower()
    sus_count = sum(1 for kw in PHISHING_KEYWORDS if kw in t)
    leg_count = sum(1 for kw in LEGIT_KEYWORDS if kw in t)
    if sus_count >= 2 or (sus_count >= 1 and "urgent" in t):
        return "suspicious"
    if leg_count >= 1 and sus_count == 0:
        return "legit"
    if sus_count >= 1:
        return "suspicious"
    return "neutral"


def extract_name_from_sms(sms_text: str) -> str | None:
    """Extract recipient first name from SMS greeting."""
    patterns = [
        r"(?:hi|hello|dear|hey)\s+([A-ZÀ-Ž][a-zà-ž]+)",
        r"(?:URGENT|Urgent)[:\s]+([A-ZÀ-Ž][a-zà-ž]+)",
    ]
    for pat in patterns:
        m = re.search(pat, sms_text)
        if m:
            return normalize_str(m.group(1))
    return None


def extract_name_from_mail(mail_text: str) -> str | None:
    """Extract recipient name from mail To: header."""
    m = re.search(r'To:\s*"?([^"<\n]+)"?\s*<', mail_text)
    if m:
        name = m.group(1).strip()
        parts = name.split()
        if parts:
            return normalize_str(parts[0])
    return None


def extract_date_from_message(text: str) -> datetime | None:
    """Extract date from SMS/mail Date: field."""
    m = re.search(r"Date:\s*(?:\w+,\s*)?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    m = re.search(r"Date:\s*\w+,\s*(\d{1,2}\s+\w+\s+\d{4}\s+\d{2}:\d{2}:\d{2})", text)
    if m:
        for fmt in ["%d %b %Y %H:%M:%S", "%d %B %Y %H:%M:%S"]:
            try:
                return datetime.strptime(m.group(1), fmt)
            except ValueError:
                pass
    return None


def build_message_index(sms_list: list[dict], mails: list[dict], user_idx: dict) -> dict:
    """Index messages by normalized first name with classification.

    Returns: {norm_firstname: [{"text": str, "date": datetime|None, "class": str, "source": "sms"|"mail"}]}
    """
    idx = defaultdict(list)

    for s in sms_list:
        text = s.get("sms", "")
        name = extract_name_from_sms(text)
        cls = classify_message(text)
        dt = extract_date_from_message(text)
        if name:
            idx[name].append({"text": text[:500], "date": dt, "class": cls, "source": "sms"})

    for m in mails:
        text = m.get("mail", "")
        name = extract_name_from_mail(text)
        cls = classify_message(text)
        dt = extract_date_from_message(text)
        if name:
            idx[name].append({"text": text[:500], "date": dt, "class": cls, "source": "mail"})

    return dict(idx)


def get_messages_near_time(msg_idx: dict, user: dict, txn_dt: datetime,
                           window_days: int = 14) -> dict:
    """Get message counts near a transaction time for a user."""
    if not user:
        return {"suspicious": 0, "legit": 0, "total": 0, "recent_suspicious_texts": []}

    fn = user.get("_norm_fn", "")
    msgs = msg_idx.get(fn, [])
    if not msgs:
        return {"suspicious": 0, "legit": 0, "total": 0, "recent_suspicious_texts": []}

    window = timedelta(days=window_days)
    sus = 0
    leg = 0
    total = 0
    sus_texts = []

    for m in msgs:
        if m["date"] and abs(txn_dt - m["date"]) <= window:
            total += 1
            if m["class"] == "suspicious":
                sus += 1
                # Extract just the message part for compact evidence
                lines = m["text"].split("\n")
                msg_line = ""
                for l in lines:
                    if l.strip().startswith("Message:"):
                        msg_line = l.strip()[8:].strip()[:150]
                        break
                    if l.strip().startswith("Subject:"):
                        msg_line = l.strip()[8:].strip()[:100]
                        break
                if msg_line and len(sus_texts) < 3:
                    sus_texts.append(msg_line)
            elif m["class"] == "legit":
                leg += 1

    return {
        "suspicious": sus,
        "legit": leg,
        "total": total,
        "recent_suspicious_texts": sus_texts,
    }


# ── Location indexing ──────────────────────────────────────────────────

def build_location_index(locations: list[dict]) -> dict:
    """Index locations by biotag for fast lookup.

    Returns: {norm_biotag: [{"dt": datetime, "lat": float, "lng": float, "city": str}]}
    """
    idx = defaultdict(list)
    for loc in locations:
        bt = normalize_str(loc.get("biotag", ""))
        if bt:
            try:
                dt = datetime.fromisoformat(loc["timestamp"])
            except (ValueError, KeyError):
                dt = None
            idx[bt].append({
                "dt": dt,
                "lat": float(loc.get("lat", 0)),
                "lng": float(loc.get("lng", 0)),
                "city": loc.get("city", ""),
            })
    # Sort by time
    for bt in idx:
        idx[bt].sort(key=lambda x: x["dt"] if x["dt"] else datetime.min)
    return dict(idx)


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    """Approximate distance in km between two coordinates."""
    import math
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return R * 2 * math.asin(min(1, math.sqrt(a)))


def check_location_plausibility(sender_id: str, txn_location: str, txn_dt: datetime,
                                 user: dict, loc_idx: dict) -> dict:
    """Check if transaction location is plausible for the user."""
    result = {"plausible": True, "reason": "", "distance_km": 0}

    if not txn_location or not user:
        return result

    # Get user's biotag-based location data
    sender_norm = normalize_str(sender_id)
    user_locs = loc_idx.get(sender_norm, [])

    if not user_locs:
        return result

    # Find nearest GPS reading to transaction time
    nearest = None
    min_diff = timedelta(days=999)
    for loc in user_locs:
        if loc["dt"]:
            diff = abs(txn_dt - loc["dt"])
            if diff < min_diff:
                min_diff = diff
                nearest = loc

    if not nearest or min_diff > timedelta(hours=24):
        return result

    # Check if transaction location city matches GPS city
    txn_city = normalize_str(txn_location.split(" - ")[0] if " - " in txn_location else txn_location)
    gps_city = normalize_str(nearest["city"])

    if txn_city and gps_city and txn_city != gps_city:
        # Check distance
        dist = haversine_km(nearest["lat"], nearest["lng"],
                           user.get("_res_lat", 0), user.get("_res_lng", 0))
        result["plausible"] = False
        result["reason"] = f"GPS in {nearest['city']} but txn in {txn_location}"
        result["distance_km"] = round(dist, 1)

    return result
