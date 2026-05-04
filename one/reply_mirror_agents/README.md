# Variant 1 — Reply Mirror Agents (Two-Agent / OpenRouter)

Solution for the **Reply Mirror** AI Agent Challenge: detect fraudulent transactions in MirrorPay (year 2087) using a cooperative system of intelligent agents.

> Problem statement: `../../AIAgentChallenge-ProblemStatement16April.pdf`

---

## 1. Problem at a glance

You are *The Eye*. Each level provides one training and one evaluation dataset. For every transaction you must decide **fraud / not_fraud** and emit a newline-separated list of fraudulent `transaction_id`s. Score combines accuracy, false-positive cost, latency and adaptability.

Constraints from the brief:
- Only **agent-based** solutions are permitted; fully deterministic approaches are evaluated with reservation.
- Output is invalid if no transactions, all transactions, or <15 % of true frauds are reported.
- Asymmetric cost: false negatives are expensive (lost money), false positives are expensive (blocked legit customer).

## 2. Data description

Each split (`The Truman Show - train/`, `The Truman Show - validation/`) contains:

| File | Schema | Joins via |
|---|---|---|
| `transactions.csv` | `transaction_id, sender_id, recipient_id, transaction_type {bank_transfer, in-person, e-commerce, direct_debit, withdrawal}, amount, location, payment_method, sender_iban, recipient_iban, balance_after, description, timestamp` | `sender_iban` / `recipient_iban` ↔ `users.iban` |
| `users.json` | `first_name, last_name, birth_year, salary, job, iban, residence{city,lat,lng}, description` | self |
| `locations.json` | `biotag, timestamp, lat, lng, city` | `biotag` city-code prefix ↔ user `residence.city` |
| `sms.json` | raw SMS thread per record | recipient name / first-name match |
| `mails.json` | raw email thread per record | `To:` header / email-local-part match |

## 3. Architecture (two cooperating LLM agents)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  load_dataset(split)  → DatasetBundle                                   │
│    transactions, users, locations, messages (SMS+mail), pre-built       │
│    indexes by IBAN / biotag / user                                      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │   for each transaction:
                               ▼
            ┌────────────────────────────────────┐
            │       FraudTools (evidence)        │
            │  ─ get_transaction_context()       │
            │  ─ resolve_transaction_parties()   │
            │  ─ get_location_context()  (GPS)   │
            │  ─ get_message_context()  (SMS+mail)│
            │  ─ get_behavior_baseline() (history)│
            │  ─ score_rule_based_risk() (rules.py)│
            └────────────────┬───────────────────┘
                             │  EvidenceBundle + RiskAssessment
                             ▼
                ┌────────────────────────────┐
                │  AGENT 1 — Primary         │
                │  Nemotron-3-super-120b     │
                │  (OpenRouter, T=0)         │
                │  → AgentDecision JSON      │
                └─────────────┬──────────────┘
                              │
                  ┌───────────┴────────────┐
                  │  Orchestrator gate     │
                  │  review if:            │
                  │   confidence < 0.75    │
                  │   OR (high_value AND   │
                  │        p(fraud) ≥0.45) │
                  │   OR (risk≥0.75 AND    │
                  │        confidence<0.9) │
                  └───────────┬────────────┘
                              │ (~30–40 % of cases)
                              ▼
                ┌────────────────────────────┐
                │  AGENT 2 — Reviewer        │
                │  Gemma-4-31b-it            │
                │  (OpenRouter, T=0)         │
                │  → AgentDecision JSON      │
                └─────────────┬──────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  _combine() — disagreement tie-break:      │
        │   blended = 0.45·rule_risk                 │
        │           + 0.35·primary_p(fraud)          │
        │           + 0.20·reviewer_p(fraud)         │
        │   (+0.05 nudge if economically high impact)│
        │   fraud iff blended ≥ 0.62                 │
        └─────────────────────┬──────────────────────┘
                              ▼
              output/<split>_output.txt   (fraud IDs)
              output/predictions_<split>.csv
              output/langfuse_sessions.json
              output/reply_mirror_agents_source.zip
```

### Why this shape

- **Structured evidence over raw files** – tools build a compact JSON bundle, so the LLM never sees the whole CSV/JSON dump per call.
- **Two heterogeneous models** – Nemotron and Gemma have different failure modes; disagreement is informative.
- **Rule-weighted disagreement** – when the agents disagree the deterministic risk score breaks the tie, weighted highest. This protects against a single noisy LLM tipping the verdict.
- **Economic-impact escalation** – high-value transactions get a small fraud-leaning nudge, mirroring the asymmetric cost in the scoring rules.

## 4. Project layout

```
reply_mirror_agents/
├── main.py                  CLI entrypoint
├── config.py                .env loader, thresholds, paths
├── data_loader.py           Schema parsing + indexing
├── linking.py               IBAN/biotag/party resolution
├── tools.py                 Evidence-building "tools"
├── rules.py                 Deterministic risk scoring
├── prompts.py               System + user prompts for both agents
├── llm_utils.py             OpenRouter LangChain wiring
├── tracing.py               Langfuse spans (run / tool / agent / event)
├── agent_primary.py         Agent 1 logic + JSON repair
├── agent_reviewer.py        Agent 2 logic + JSON repair
├── orchestrator.py          Per-txn flow + tie-break
├── output_writer.py         TXT/CSV/zip emission
├── models.py                Pydantic + dataclass schemas
├── utils.py                 Parsing, haversine, etc.
└── notes/linking_logic.md   Discovered schema relationships
```

## 5. Setup & run

```powershell
cd "C:\hackathon reply\one\reply_mirror_agents"
"C:/hackathon reply/.venv/Scripts/python.exe" -m pip install -r requirements.txt

# Both splits (default):
"C:/hackathon reply/.venv/Scripts/python.exe" main.py --dataset all
# Single split:
"C:/hackathon reply/.venv/Scripts/python.exe" main.py --dataset train
"C:/hackathon reply/.venv/Scripts/python.exe" main.py --dataset validation
```

### Required environment (`C:\hackathon reply\.env`)

`OPENROUTER_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `TEAM_NAME`. `AppConfig.validate()` raises immediately if any of the first three are missing.

Optional overrides: `PRIMARY_MODEL`, `REVIEWER_MODEL`, `REVIEW_THRESHOLD`, `HIGH_VALUE_REVIEW_PROBABILITY`, `DISAGREEMENT_THRESHOLD`, `LOCATION_WINDOW_HOURS`, `MESSAGE_LOOKBACK_DAYS`, `AMOUNT_HIGH_IMPACT_FLOOR`, `AMOUNT_HIGH_IMPACT_MONTHLY_SALARY_FACTOR`.

## 6. Output

- `output/<split>_output.txt` – newline-separated fraud `transaction_id`s (the official deliverable).
- `output/predictions_<split>.csv` – per-transaction audit: amount, both agent verdicts/confidences, tie-break policy, risk score, reasons.
- `output/langfuse_sessions.json` – Langfuse trace IDs/URLs for the run.
- `output/reply_mirror_agents_source.zip` – source archive (auto-emitted for submission).

## 7. Langfuse tracing

Each split run becomes one Langfuse trace with nested spans: per-tool calls (`get_transaction_context`, `resolve_transaction_parties`, …), per-agent generations, and a final `transaction_finalized` event with the verdict. Session id format: `{TEAM_NAME}-{ulid}`.
