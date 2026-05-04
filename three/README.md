# Variant 3 — Reply Mirror Level 3 (Audio + Tiered LLM)

Solution for **Level 3** of the Reply Mirror challenge. Level 3 adds **audio recordings** (phone-call transcripts) to the data sources from Levels 1–2.

> Problem statement: `../AIAgentChallenge-ProblemStatement16April.pdf`

This variant is optimised for **low cost and low latency** at scale. Where variants 1–2 send every transaction to two heterogeneous LLMs, variant 3 layers a deterministic risk gate, a cheap LLM, and a stronger LLM so each tier only sees the cases it is best suited for.

---

## 1. Problem at a glance

You are *The Eye*. Decide **fraud / not_fraud** for every transaction and emit a newline-separated list of fraudulent `transaction_id`s. Score combines accuracy, false-positive cost, latency, and adaptability. Only **agent-based** solutions are permitted. Output is invalid if no transactions, all transactions, or <15 % of true frauds are reported.

## 2. Data description

Each split (`train_data/`, `val_data/`) contains:

| File / Dir | Schema | Joins via |
|---|---|---|
| `transactions.csv` | `transaction_id, sender_id, recipient_id, transaction_type, amount, location, payment_method, sender_iban, recipient_iban, balance_after, description, timestamp` | `sender_iban` / `recipient_iban` ↔ `users.iban` |
| `users.json` | `first_name, last_name, birth_year, salary, job, iban, residence{city,lat,lng}, description` | self |
| `locations.json` | `biotag, timestamp, lat, lng, city` | `biotag` city-code prefix ↔ user `residence.city` |
| `sms.json` | raw SMS thread per record | first-name greeting / `Hi <name>` / `URGENT: <name>` |
| `mails.json` | raw email thread per record | `To: "Name" <addr>` header |
| `audio/*.mp3` | filename pattern `YYYYMMDD_HHMMSS-firstname_lastname.mp3` | filename → user; transcribed locally with faster-whisper |

## 3. Architecture (rules → cheap LLM → stronger LLM)

```
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 1 — load_all(data_dir)                                          │
│    transactions / users / locations / sms / mails / audio paths        │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 2 — build_all_evidence()  (CACHED to evidence/*.json)           │
│    ─ user index        (biotag, IBAN, name, city)                      │
│    ─ message index     (per-firstname, classified suspicious/legit)    │
│    ─ location index    (per-biotag, time-sorted GPS)                   │
│    ─ history snapshot  (per-sender pre-txn baseline)                   │
│    ─ audio transcribe  (faster-whisper base int8 on CPU, CACHED)       │
│    For each txn: assemble compact evidence dict (~1 KB)                │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 3 — screen_transactions()  (deterministic risk score)           │
│                                                                        │
│   risk = f(iban_mismatch, location_implausible, amount_anomaly,        │
│            time_anomaly, msg_suspicious, audio_suspicious,             │
│            balance_anomaly, salary_ratio, new_recipient, …)            │
│                                                                        │
│         ┌──────────── risk ≥ 0.85 ────────────┐                        │
│         │                                     │                        │
│         ▼                                     │                        │
│   AUTO-FRAUD       0.30 ≤ risk < 0.85 ──→ SHORTLIST                    │
│   (no LLM)                                    │                        │
│                                               │                        │
│         ┌──────────── risk < 0.30 ────────────┘                        │
│         ▼                                                              │
│   AUTO-CLEAN (no LLM)                                                  │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │  shortlist only (typically ≤ 20 % of txns)
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 4 — Primary LLM review  (parallel batches)                      │
│            anthropic/claude-haiku-4.5  via OpenRouter                  │
│                                                                        │
│   shortlist split into batches of BATCH_SIZE (12),                     │
│   dispatched MAX_PARALLEL_BATCHES (4) at a time via ThreadPoolExecutor.│
│   Each batch returns a JSON array of compact verdicts                  │
│   {id, v∈{F,L}, c∈[0,1], r}.                                           │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │  borderline = confidence < 0.70
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 5 — Reviewer escalation  (parallel batches)                     │
│            anthropic/claude-sonnet-4.6  via OpenRouter                 │
│                                                                        │
│   ONLY borderline verdicts are re-reviewed by the stronger model.      │
│   Reviewer verdict overwrites the primary one for those txns.          │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
       fraud_ids = AUTO-FRAUD ∪ {v.id : v.verdict == FRAUD}
       output/train.txt   (or val_fraud.txt)
       output/<split>_audit.json
       output/langfuse_sessions.txt
```

### Why this shape (vs. variants 1 & 2)

| Decision | Cost / latency rationale |
|---|---|
| **Audio transcribed locally** with `faster-whisper base int8` | Zero LLM tokens for ~hours of audio; the whole transcription set fits on a CPU. Cached to JSON so reruns skip the slow step. |
| **Evidence cached to disk** | Re-tuning thresholds or models replays the LLM stage only — preprocessing is amortised across runs. |
| **Deterministic risk gate first** | Most transactions have an obvious verdict from rules alone (employer salary deposits, zero-anomaly retail). Sending these to an LLM is pure waste. Typical gate retains only ~10–20 % of txns for LLM review. |
| **Cheap model for shortlist (Haiku 4.5)** | ~10× cheaper and ~3× faster than Sonnet, plenty strong for the high-signal compact evidence format. |
| **Stronger model only for borderline (Sonnet 4.6)** | The asymmetric cost matters most where the cheap model is uncertain. Escalation typically affects <30 % of the shortlist → <6 % of total transactions. |
| **Parallel batch dispatch** | OpenRouter / Anthropic accepts concurrent requests; we run 4 batches in flight to amortise network latency without tripping rate limits. |
| **Compact evidence format** (`amt=…`, `IBAN-tool: MISMATCH`, …) | Dense, low-token representation. A batch of 12 txns is ~2.5 K input tokens. |

### Estimated savings vs. naive single-tier

For ~2 000 transactions on validation:
- Naive (every txn to Sonnet, batch 12): ~167 Sonnet batches.
- Variant 3: rules absorb ~80 %, Haiku reviews ~400 txns ≈ 34 batches, Sonnet escalates ~120 txns ≈ 10 batches. **~5 % of the Sonnet calls**, plus parallelism cuts wall-clock further.

## 4. Project layout

```
three/
├── main.py                    Pipeline entrypoint (5 stages above)
├── requirements.txt
└── src/
    ├── config.py              .env, paths, model names, thresholds, batching
    ├── data_loader.py         CSV/JSON/audio file loading
    ├── linking.py             biotag/IBAN/name resolution + msg/location index
    ├── audio_processor.py     faster-whisper transcription + keyword match
    ├── preprocess.py          Evidence builder + risk score + screening
    ├── agent.py               Two-tier LLM (parallel batches + escalation)
    └── tracing.py             Langfuse v3 spans (preprocessing/screening/llm)
├── train_data/                Training split (with audio/)
├── val_data/                  Validation split (with audio/)
├── evidence/                  Cached evidence + audio transcriptions
└── output/                    Submission files + audits + Langfuse sessions
```

## 5. Setup & run

```powershell
cd "C:\hackathon reply\three"
"C:/hackathon reply/.venv/Scripts/python.exe" -m pip install -r requirements.txt

"C:/hackathon reply/.venv/Scripts/python.exe" main.py
```

The first run transcribes all audio (slow on CPU); subsequent runs reuse `evidence/<split>_audio.json`. To force re-preprocessing, delete `evidence/<split>_evidence.json` (and/or the audio cache).

### Required environment (`C:\hackathon reply\.env`)

`OPENROUTER_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `TEAM_NAME`.

Tunables in `src/config.py`:

| Knob | Default | Effect |
|---|---|---|
| `PRIMARY_MODEL` | `anthropic/claude-haiku-4.5` | Cheap reviewer over the shortlist |
| `REVIEWER_MODEL` | `anthropic/claude-sonnet-4.6` | Stronger reviewer over borderline cases |
| `RISK_THRESHOLD_LLM` | `0.30` | Below → auto-clean (no LLM) |
| `RISK_THRESHOLD_AUTO_FRAUD` | `0.85` | Above → auto-fraud (no LLM) |
| `BATCH_SIZE` | `12` | Transactions per LLM call |
| `MAX_PARALLEL_BATCHES` | `4` | Concurrent in-flight batches |
| `ESCALATION_CONFIDENCE` | `0.70` | Primary verdict below this → reviewer |
| `WHISPER_MODEL_SIZE` | `base` | faster-whisper model (int8 CPU) |

## 6. Output

- `output/train.txt` and `output/val_fraud.txt` – official deliverable, one fraud `transaction_id` per line.
- `output/<run>_audit.json` – full per-txn audit including rule-based shortlist counts, primary/escalated verdict counts, timings.
- `output/langfuse_sessions.txt` – Langfuse session IDs to submit alongside the outputs.

## 7. Langfuse tracing

Each split run creates one trace with nested spans:
- `preprocessing` – data loading + evidence build
- `fast-screening` – auto-fraud / shortlist / auto-clean counts
- `agent-primary-N` – each Haiku batch (input/output tokens recorded)
- `agent-reviewer-N` – each Sonnet escalation batch
- `final-output` – fraud count + rate

Session-id format: `{TEAM_NAME}-{run_name}-{ulid}`.
