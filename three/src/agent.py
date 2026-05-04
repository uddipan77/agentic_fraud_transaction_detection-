"""Two-tier LLM fraud-review agent.

Pipeline used by main.py:
  1. Rule-based screening already split transactions into auto-fraud / auto-clean / shortlist.
  2. PRIMARY (cheap, fast) reviews the shortlist in parallel batches.
  3. REVIEWER (stronger) re-checks only verdicts whose confidence is below ESCALATION_CONFIDENCE.

Both tiers share the same compact evidence format, so escalation is a single extra call per borderline txn.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PRIMARY_MODEL,
    REVIEWER_MODEL,
    BATCH_SIZE,
    MAX_PARALLEL_BATCHES,
    ESCALATION_CONFIDENCE,
)

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


SYSTEM_PROMPT = """You are a fraud detection agent for MirrorPay (year 2087).
You receive batches of transactions, each with evidence already gathered by tools.

For EVERY transaction in the batch you MUST emit one verdict.

Evidence sources (already extracted, not raw):
- Transaction details: amount, type, sender, recipient, timestamp, location, payment method
- User profile: resolved identity, residence city, salary, IBAN
- IBAN check, Location/GPS check, Message classification, Audio call classification
- Sender behaviour history: avg/max/std amount, known recipients/methods

Fraud signals: IBAN mismatch, location mismatch, phishing/scam messages or calls near time,
unusual amount vs history, new recipient + high amount, late-night + anomalies,
negative balance, amount disproportionate to salary.

Legitimacy signals: salary/rent/subscription, repeated recipient, normal amount,
consistent location, no suspicious nearby messages or calls.

OUTPUT FORMAT — JSON array only, no commentary:
[{"id":"<transaction_id>","v":"F" or "L","c":0.0-1.0,"r":"brief reason"}]

v=F means FRAUD, v=L means LEGIT. Reasons under 15 words."""


def format_evidence(ev: dict) -> str:
    """Compact textual representation of one evidence bundle."""
    lines = [f"[{ev['transaction_id']}]"]
    lines.append(
        f"  Txn: {ev['transaction_type']} | amt={ev['amount']} | bal={ev['balance_after']} | ts={ev['timestamp']}"
    )

    sender_info = ev["sender_id"]
    if ev.get("sender_name"):
        sender_info += f" ({ev['sender_name']}, {ev.get('sender_city', '')})"
    elif ev.get("sender_is_employer"):
        sender_info += " [employer]"
    lines.append(f"  Sender: {sender_info}")

    rcpt_info = ev.get("recipient_id", "") or "n/a"
    if ev.get("recipient_name"):
        rcpt_info += f" ({ev['recipient_name']})"
    lines.append(f"  Recipient: {rcpt_info}")

    if ev.get("location"):
        lines.append(f"  Location: {ev['location']}")
    if ev.get("payment_method"):
        lines.append(f"  Payment: {ev['payment_method']}")
    if ev.get("description"):
        lines.append(f"  Desc: {ev['description']}")

    if ev.get("iban_mismatch"):
        lines.append("  IBAN-tool: MISMATCH – sender IBAN does not match known user IBAN")
    elif ev.get("sender_resolved") and not ev.get("sender_is_employer"):
        lines.append("  IBAN-tool: OK – matches known user")

    if not ev.get("location_plausible") and ev.get("location_reason"):
        lines.append(f"  Location-tool: MISMATCH – {ev['location_reason']}")
    elif ev.get("location") and ev.get("sender_resolved"):
        lines.append("  Location-tool: consistent")

    sus = ev.get("msg_suspicious_count", 0)
    leg = ev.get("msg_legit_count", 0)
    if sus or leg:
        lines.append(f"  Msg-tool: {sus} suspicious, {leg} legit msgs within 14 days")
        for txt in ev.get("msg_recent_suspicious", [])[:2]:
            lines.append(f"    >{txt[:140]}")

    for a in (ev.get("audio_nearby") or [])[:2]:
        kw = ", ".join(a.get("keywords", []))
        lines.append(f"  Audio-tool: call {a.get('time_diff_hours', '')}h away, keywords=[{kw}]")
        if a.get("is_suspicious") and a.get("transcript_snippet"):
            lines.append(f"    >{a['transcript_snippet'][:150]}")

    lines.append(
        f"  History-tool: {ev.get('hist_txn_count', 0)} prior txns, "
        f"avg={ev.get('hist_avg_amount', 0)}, max={ev.get('hist_max_amount', 0)}, "
        f"new_rcpt={'yes' if ev.get('is_new_recipient') else 'no'}, "
        f"new_method={'yes' if ev.get('is_new_method') else 'no'}"
    )

    facts = []
    if ev.get("balance_anomaly", 0) > 0:
        facts.append("negative_balance")
    if ev.get("time_anomaly", 0) > 0:
        facts.append("late_night(00-05h)")
    if ev.get("salary_ratio", 0) > 0.5:
        facts.append(f"amount>{ev['salary_ratio']:.0%}_of_monthly_salary")
    if ev.get("amount_anomaly", 0) > 0.3:
        facts.append(f"amount_unusual(z={ev['amount_anomaly']:.2f})")
    if ev.get("is_routine"):
        facts.append("routine_payment")
    if facts:
        lines.append(f"  Facts: {', '.join(facts)}")

    return "\n".join(lines)


def _call_llm(model: str, prompt: str, max_tokens: int = 3000) -> tuple[str, int, int]:
    """Single LLM call. Returns (content, in_tokens, out_tokens)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content.strip()
    in_tok = response.usage.prompt_tokens if response.usage else 0
    out_tok = response.usage.completion_tokens if response.usage else 0
    return content, in_tok, out_tok


def _parse_verdicts(content: str) -> list[dict]:
    """Parse and normalize a JSON array of verdicts (handles markdown wrapping)."""
    parsed = content
    if parsed.startswith("```"):
        parsed = parsed.split("```")[1]
        if parsed.startswith("json"):
            parsed = parsed[4:]
        parsed = parsed.strip()
    raw = json.loads(parsed)
    out = []
    for v in raw:
        out.append({
            "transaction_id": v.get("id", v.get("transaction_id", "")),
            "verdict": "FRAUD" if v.get("v", v.get("verdict", "")) in ("F", "FRAUD") else "LEGIT",
            "confidence": float(v.get("c", v.get("confidence", 0.5))),
            "reason": v.get("r", v.get("reason", "")),
        })
    return out


def _review_one_batch(
    batch: list[dict],
    batch_num: int,
    model: str,
    trace_fn,
) -> list[dict]:
    """Run one LLM batch call. On failure mark txns LEGIT (avoids false positives)."""
    evidence_text = "\n\n".join(format_evidence(ev) for ev in batch)
    prompt = (
        f"Review {len(batch)} transactions. Return a JSON array with one verdict per txn:\n\n"
        f"{evidence_text}"
    )
    try:
        content, in_tok, out_tok = _call_llm(model, prompt)
        verdicts = _parse_verdicts(content)
        if trace_fn:
            trace_fn(prompt, content, batch_num, input_tokens=in_tok, output_tokens=out_tok)
        return verdicts
    except (json.JSONDecodeError, Exception) as exc:
        return [
            {
                "transaction_id": ev["transaction_id"],
                "verdict": "LEGIT",
                "confidence": 0.3,
                "reason": f"LLM error fallback: {type(exc).__name__}",
            }
            for ev in batch
        ]


def batch_review(
    shortlist: list[dict],
    trace_fn=None,
    model: str = PRIMARY_MODEL,
) -> list[dict]:
    """Review every shortlisted transaction with the cheap primary model in parallel batches."""
    if not shortlist:
        return []

    batches = [shortlist[i : i + BATCH_SIZE] for i in range(0, len(shortlist), BATCH_SIZE)]
    print(f"  Dispatching {len(batches)} batches × ≤{BATCH_SIZE} txns "
          f"({MAX_PARALLEL_BATCHES} in parallel) to {model}")

    results: list[tuple[int, list[dict]]] = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as pool:
        futures = {
            pool.submit(_review_one_batch, batch, idx + 1, model, trace_fn): idx
            for idx, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            verdicts = fut.result()
            n_fraud = sum(1 for v in verdicts if v["verdict"] == "FRAUD")
            print(f"    batch {idx + 1}/{len(batches)}: {n_fraud}F/{len(verdicts) - n_fraud}L")
            results.append((idx, verdicts))

    results.sort(key=lambda r: r[0])
    return [v for _, batch_verdicts in results for v in batch_verdicts]


def escalate_borderline(
    primary_verdicts: list[dict],
    evidence_by_id: dict[str, dict],
    trace_fn=None,
) -> list[dict]:
    """Re-review only the verdicts whose primary confidence is below ESCALATION_CONFIDENCE.

    Returns the merged final verdict list (escalated entries replace primary entries).
    """
    borderline = [v for v in primary_verdicts if v["confidence"] < ESCALATION_CONFIDENCE]
    if not borderline:
        print(f"  No borderline verdicts to escalate (threshold={ESCALATION_CONFIDENCE})")
        return primary_verdicts

    print(f"  Escalating {len(borderline)}/{len(primary_verdicts)} borderline verdicts to {REVIEWER_MODEL}")
    borderline_evidence = [evidence_by_id[v["transaction_id"]] for v in borderline if v["transaction_id"] in evidence_by_id]
    reviewer_verdicts = batch_review(borderline_evidence, trace_fn=trace_fn, model=REVIEWER_MODEL)

    by_id = {v["transaction_id"]: v for v in primary_verdicts}
    for rv in reviewer_verdicts:
        # Reviewer wins on disagreement OR low primary confidence
        rv["reason"] = f"[reviewer] {rv['reason']}"
        by_id[rv["transaction_id"]] = rv
    return list(by_id.values())
