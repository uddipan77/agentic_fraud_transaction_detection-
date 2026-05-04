# Variant 2 — Fraud Detection (Two-Agent / Groq)

Solution for the **Reply Mirror** AI Agent Challenge: detect fraudulent transactions in MirrorPay (year 2087) using cooperative agents.

> Problem statement: `../AIAgentChallenge-ProblemStatement16April.pdf`

This variant follows the same two-agent shape as `one/` but rewires it for **Groq's free hosted models** and splits the work into a separate **preprocess phase** that persists evidence to JSONL on disk before any LLM call.

---

## 1. Problem at a glance

You are *The Eye*. For every transaction, decide **fraud / not_fraud** and emit a newline-separated list of fraudulent `transaction_id`s. Score combines accuracy, false-positive cost, latency, and adaptability. Only **agent-based** solutions are permitted; output is invalid if no transactions, all transactions, or <15 % of true frauds are reported.

## 2. Data description

Each split (`train_data/`, `validation_data/`) contains:

| File | Schema | Joins via |
|---|---|---|
| `transactions.csv` | `transaction_id, sender_id, recipient_id, transaction_type {bank_transfer, in-person, e-commerce, direct_debit, withdrawal}, amount, location, payment_method, sender_iban, recipient_iban, balance_after, description, timestamp` | `sender_iban` / `recipient_iban` ↔ `users.iban` |
| `users.json` | `first_name, last_name, birth_year, salary, job, iban, residence{city,lat,lng}, description` | self |
| `locations.json` | `biotag, timestamp, lat, lng, city` | `biotag` city-code prefix ↔ user `residence.city` |
| `sms.json` | raw SMS thread per record | recipient name / first-name match |
| `mails.json` | raw email thread per record | `To:` header / email-local-part match |

(No audio in this split — handled separately in variant 3.)

## 3. Architecture (preprocess → 2-agent decision)

```
┌────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — preprocess_split(split, config)                         │
│  ─────────────────────────────────────────────────────────────     │
│  load_dataset()  → Dataset (txns, users, locs, sms, mails)         │
│  DataLinker      → biotag↔user, iban↔user, message indexes         │
│  build_evidence_bundle(txn) for every txn:                         │
│    ─ get_transaction_context()                                     │
│    ─ resolve_transaction_parties()  (IBAN cross-check)             │
│    ─ get_location_context(window=24h)                              │
│    ─ get_message_context(lookback=7d)                              │
│    ─ get_behavior_baseline()                                       │
│    ─ score_rule_based_risk()  (weighted multi-signal)              │
│  save_evidence() → evidence/<split>/evidence.jsonl                 │
└────────────────────────────┬───────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│  PHASE 2 — Orchestrator.run_split(split)                           │
│  ─────────────────────────────────────────────────────────────     │
│  for each evidence bundle:                                         │
│                                                                    │
│            ┌────────────────────────────┐                          │
│            │ AGENT 1 — PrimaryAgent     │                          │
│            │ Llama-4-scout-17b (Groq)   │                          │
│            │ → JSON {decision, conf, …} │                          │
│            └────────────┬───────────────┘                          │
│                         │                                          │
│           ┌─────────────┴────────────┐                             │
│           │ should_review()          │                             │
│           │  conf < 0.75   OR        │                             │
│           │ (high_value AND conf<0.8)│                             │
│           └─────────────┬────────────┘                             │
│                         │                                          │
│            ┌────────────▼───────────────┐                          │
│            │ AGENT 2 — ReviewerAgent    │                          │
│            │ Qwen3-32b (Groq)           │                          │
│            └────────────┬───────────────┘                          │
│                         │                                          │
│        ┌────────────────▼────────────────────┐                     │
│        │ make_final_decision():              │                     │
│        │   agree   → accept (max conf)       │                     │
│        │   gap>0.1 → higher-conf agent wins  │                     │
│        │   tie     → high-value⇒fraud,       │                     │
│        │            else conservative legit  │                     │
│        └─────────────────┬───────────────────┘                     │
└──────────────────────────┼─────────────────────────────────────────┘
                           ▼
        output/<split>_output.txt        (fraud IDs)
        output/<split>_predictions.csv   (full audit)
        output/<split>_audit.json        (full per-txn detail)
```

### Why this shape

- **Disk-resident evidence** – preprocess once, run agents N times. Lets you tune thresholds and re-run only Phase 2.
- **Two heterogeneous Groq models** – Llama-4 and Qwen3 disagree on different cases; using both gives orthogonal signal.
- **Conservative-on-tie** – when both agents have similar confidence and disagree, low-value cases default to `not_fraud` (cheap to be wrong) and high-value cases default to `fraud` (expensive to miss). Mirrors the asymmetric cost in the scoring rules.

## 4. Project layout

```
two/
├── main.py                   2-phase entrypoint
└── fraud_detection/
    ├── config.py             dataclass config + .env loader
    ├── data_loader.py        CSV/JSON parsing into Dataset
    ├── linking.py            biotag↔user / IBAN / location / message indexes
    ├── tools.py              evidence-builder functions
    ├── rules.py              weighted multi-signal risk scorer
    ├── evidence_store.py     build / save / load JSONL evidence
    ├── preprocess.py         Phase 1 driver
    ├── prompts.py            system + user prompts (primary + reviewer)
    ├── llm_client.py         Groq HTTP client w/ 429 retry/backoff
    ├── agent_primary.py      Agent 1 + JSON repair + fallback
    ├── agent_reviewer.py     Agent 2 + JSON repair + fallback
    ├── orchestrator.py       per-txn flow + tie-break + summary
    ├── output_writer.py      txt / csv / json outputs
    └── utils.py              shared helpers
```

## 5. Setup & run

```powershell
cd "C:\hackathon reply\two"
"C:/hackathon reply/.venv/Scripts/python.exe" -m pip install -r fraud_detection/requirements.txt

# Both splits (default):
"C:/hackathon reply/.venv/Scripts/python.exe" main.py
# Single split:
"C:/hackathon reply/.venv/Scripts/python.exe" main.py train
```

Phase 1 (preprocess) writes `evidence/<split>/evidence.jsonl`. Delete that file to force re-preprocessing; otherwise Phase 2 re-uses it.

### Required environment (`C:\hackathon reply\.env`)

`GROQ_API_KEY` — variant 2 uses Groq's hosted endpoints, *not* OpenRouter (despite the broader repo's defaults). `main.py` exits with an error if it is missing.

Tunable defaults live in `fraud_detection/config.py`:
- `primary_model = "meta-llama/llama-4-scout-17b-16e-instruct"`
- `reviewer_model = "qwen/qwen3-32b"`
- `review_confidence_threshold = 0.75`
- `high_value_review_threshold = 0.80`
- `location_window_hours = 24.0`
- `message_lookback_days = 7`

## 6. Output

- `output/<split>_output.txt` – the official deliverable, one fraud `transaction_id` per line.
- `output/<split>_predictions.csv` – per-txn audit row.
- `output/<split>_audit.json` – full structured audit including both agents' raw outputs.

## 7. Operational notes

- LLM client retries 5× with exponential backoff on 429 (Groq rate limits aggressively on free tier).
- Inter-transaction sleep of 0.5 s and inter-agent sleep of 0.5 s built into the orchestrator to stay under burst limits.
- All agents run sequentially per transaction; no parallelism. Latency scales linearly with `N_transactions`.
