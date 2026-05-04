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

## 4. Per-transaction flow (sequence)

Variant 3 has **two granularities** of flow: a per-split pipeline (5 stages) and a per-transaction journey through the cascade. Both are shown below.

### 4.1 Per-split pipeline (one call to `run_pipeline`)

```
   main.py
     │
     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Stage 1 — load_all(data_dir)                                 │
   │   transactions.csv, users.json, locations.json,              │
   │   sms.json, mails.json, audio/*.mp3                          │
   └────────────────┬─────────────────────────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Stage 2 — build_all_evidence(data, cache_path, whisper)      │
   │   if cache exists: load and return                           │
   │   else:                                                      │
   │     • build_user_index()       biotag/IBAN/name              │
   │     • build_message_index()    classify per firstname        │
   │     • build_location_index()   per biotag, time-sorted GPS   │
   │     • build_transaction_history()  per-sender pre-txn stats  │
   │     • transcribe_audio_files() faster-whisper base int8 CPU  │
   │       (cached separately to <split>_audio.json)              │
   │     • link_audio_to_user()     filename → user               │
   │     • for each txn: assemble compact evidence dict (~1 KB)   │
   │   save list[evidence] → evidence/<split>_evidence.json       │
   └────────────────┬─────────────────────────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Stage 3 — screen_transactions(evidence,                      │
   │                  threshold_llm=0.30, threshold_auto=0.85)    │
   │   for each ev: ev.risk_score = compute_risk_score(ev)        │
   │   bucket into:                                               │
   │     • auto_fraud  (risk ≥ 0.85, no LLM, IDs go to output)    │
   │     • shortlist   (0.30 ≤ risk < 0.85, sent to primary LLM)  │
   │     • auto_clean  (risk < 0.30, no LLM, dropped)             │
   └────────────────┬─────────────────────────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Stage 4 — batch_review(shortlist, model=PRIMARY_MODEL)       │
   │   split shortlist into batches of BATCH_SIZE (12)            │
   │   ThreadPoolExecutor(max_workers=4) dispatches batches       │
   │   in parallel → each batch is one LLM call to                │
   │   anthropic/claude-haiku-4.5                                 │
   │   → primary_verdicts: list of {id, verdict, confidence, r}   │
   └────────────────┬─────────────────────────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Stage 5 — escalate_borderline(primary_verdicts, …)           │
   │   borderline = [v for v if v.confidence < 0.70]              │
   │   if empty: skip                                              │
   │   else: batch_review(borderline_evidence,                    │
   │                       model=REVIEWER_MODEL)                  │
   │   → reviewer_verdicts overwrite the borderline entries       │
   │   → final_verdicts                                           │
   └────────────────┬─────────────────────────────────────────────┘
                    ▼
   fraud_ids = auto_fraud  ∪  {v.id for v in final_verdicts if v.verdict == FRAUD}
   write output/<split>.txt, <split>_audit.json, langfuse_sessions.txt
```

### 4.2 Per-transaction journey through the cascade

This is what one `transaction_id` experiences. **Key invariant: each transaction is touched by at most one LLM call per tier**, and most never reach an LLM.

```
   one Transaction record
        │
        ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ Evidence-building tools (run once per txn during stage 2)      │
   │ ──────────────────────────────────────────────────────────     │
   │   • resolve_entity(sender_id)  → user via biotag prefix match  │
   │   • resolve_by_iban(iban)      → user via IBAN                 │
   │   • iban_mismatch check        → sender IBAN ↔ known IBAN      │
   │   • check_location_plausibility(sender, txn_loc, txn_dt,       │
   │                                  user, loc_idx)                │
   │       → finds nearest GPS within 24 h, compares cities         │
   │   • get_messages_near_time(msg_idx, user, txn_dt, ±14 d)       │
   │       → counts of suspicious vs legit + 3 snippets             │
   │   • get_audio_near_transaction(audio_evidence, user, txn_dt,   │
   │                                  ±72 h)                        │
   │       → matched calls + transcript snippets + keyword hits     │
   │   • txn_history snapshot (pre-this-txn) → avg/max/std amount,  │
   │                                            new_recipient,      │
   │                                            new_method, count   │
   │   • derived facts: amount_anomaly (z-score), time_anomaly      │
   │       (00–05 h), balance_anomaly (negative), salary_ratio,     │
   │       is_routine (keyword in description)                      │
   │ → ev: compact dict (~1 KB)                                     │
   └────────────────┬───────────────────────────────────────────────┘
                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ compute_risk_score(ev)  →  risk ∈ [0, 1]                       │
   │   employer-paid / routine?      → return 0.00 / 0.05           │
   │   else add weighted contributions:                             │
   │     iban_mismatch              +0.30                           │
   │     location_implausible       +0.25                           │
   │     amount_anomaly · 0.25                                      │
   │     time_anomaly   · 0.15                                      │
   │     new_recipient + high amt   +0.15  (else +0.08)             │
   │     msg_suspicious_count ≥ 3   +0.25  (≥1 +0.12)               │
   │     audio_suspicious           +0.20                           │
   │     balance_anomaly · 0.15                                     │
   │     salary_ratio > 0.5         +0.10                           │
   │     sender unresolved          +0.10                           │
   │     new payment method         +0.05                           │
   │     ecom + new_recipient + amt>200  +0.10                      │
   │     amt > 1000 AND any signal       +0.10                      │
   └────────────────┬───────────────────────────────────────────────┘
                    │
       ┌────────────┼─────────────────────────┐
       │ risk≥0.85  │ 0.30 ≤ risk < 0.85      │ risk < 0.30
       ▼            ▼                          ▼
   ┌─────────┐  ┌─────────────────────────┐  ┌──────────────┐
   │ AUTO-   │  │ SHORTLISTED             │  │ AUTO-CLEAN   │
   │ FRAUD   │  │  (sent to primary LLM)  │  │  (no LLM,    │
   │ (no LLM)│  │                         │  │   dropped)   │
   └────┬────┘  └────────────┬────────────┘  └──────┬───────┘
        │                    ▼                       │
        │   ┌────────────────────────────────────┐   │
        │   │ PRIMARY — anthropic/haiku-4.5      │   │
        │   │   format_evidence(ev) → ~200 chars │   │
        │   │   batched 12-at-a-time             │   │
        │   │   parallel × 4 via ThreadPool      │   │
        │   │   one batch = one POST to          │   │
        │   │     OpenRouter chat/completions    │   │
        │   │   T=0.1, max_tokens=3000           │   │
        │   │   verdict: {id, v∈{F,L},           │   │
        │   │             c∈[0,1], r}            │   │
        │   └─────────────┬──────────────────────┘   │
        │                 │                          │
        │       ┌─────────┴────────┐                 │
        │       │ confidence < 0.70?               │
        │       ▼                  ▼                 │
        │     YES                NO                  │
        │       │                  │                 │
        │       ▼                  │                 │
        │   ┌────────────────────┐ │                 │
        │   │ REVIEWER —         │ │                 │
        │   │ anthropic/         │ │                 │
        │   │ sonnet-4.6         │ │                 │
        │   │  same compact      │ │                 │
        │   │  evidence + format │ │                 │
        │   │  batched 12 / 4-   │ │                 │
        │   │  parallel          │ │                 │
        │   │  reviewer verdict  │ │                 │
        │   │  REPLACES primary  │ │                 │
        │   │  (reason prefixed  │ │                 │
        │   │   "[reviewer] …")  │ │                 │
        │   └─────────┬──────────┘ │                 │
        │             │            │                 │
        ▼             ▼            ▼                 ▼
       fraud      fraud_ids ← {v.id : v.verdict == FRAUD}    (skipped)
        │             │            │
        └─────────────┴────────────┘
                      ▼
            output/<split>.txt
```

### Quick reference — what touches an LLM?

| Bucket | Risk score | Primary LLM (Haiku) | Reviewer LLM (Sonnet) |
|---|---|---|---|
| auto-clean | < 0.30 | no | no |
| shortlist, primary confident | 0.30–0.85, conf ≥ 0.70 | **yes** | no |
| shortlist, borderline | 0.30–0.85, conf < 0.70 | **yes** | **yes** |
| auto-fraud | ≥ 0.85 | no | no |

Estimated traffic for ~2 000 validation transactions: ~80 % bypass the LLM entirely → ~400 hit Haiku → ~120 escalate to Sonnet.

### Failure behaviour

If a Haiku or Sonnet batch returns unparseable JSON or raises, every transaction in that batch is marked `LEGIT` with confidence 0.3 (false-positive-conservative). The pipeline never crashes on a bad LLM response.

## 5. Project layout

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

## 6. Setup & run

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

## 7. Output

- `output/train.txt` and `output/val_fraud.txt` – official deliverable, one fraud `transaction_id` per line.
- `output/<run>_audit.json` – full per-txn audit including rule-based shortlist counts, primary/escalated verdict counts, timings.
- `output/langfuse_sessions.txt` – Langfuse session IDs to submit alongside the outputs.

## 8. Langfuse tracing

Each split run creates one trace with nested spans:
- `preprocessing` – data loading + evidence build
- `fast-screening` – auto-fraud / shortlist / auto-clean counts
- `agent-primary-N` – each Haiku batch (input/output tokens recorded)
- `agent-reviewer-N` – each Sonnet escalation batch
- `final-output` – fraud count + rate

Session-id format: `{TEAM_NAME}-{run_name}-{ulid}`.
