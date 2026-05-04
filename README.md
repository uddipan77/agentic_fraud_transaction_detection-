# Reply Mirror — The Eye

Three agentic fraud-detection systems built for the **Reply Mirror** AI Agent Challenge (April 2026).

> *It is 2087. The digital metropolis of Reply Mirror runs on MirrorPay. Most data is public; fraudsters hide in plain sight. You are **The Eye** — build an autonomous intelligence that observes, adapts, and acts when the system comes under siege.*
>
> Full brief: [`AIAgentChallenge-ProblemStatement16April.pdf`](AIAgentChallenge-ProblemStatement16April.pdf)

---

## What's in this repo

Three independent solutions, one per challenge level. Each ingests the same shape of dataset (transactions, users, locations, SMS, mails — and audio at level 3) and emits a newline-separated list of fraudulent `transaction_id`s.

| Folder | Level | Architecture | LLM provider | README |
|---|---|---|---|---|
| [`one/`](one/reply_mirror_agents/README.md) | L1 | **Two cooperating agents** — Primary (Nemotron-3-super-120b) + Reviewer (Gemma-4-31b) with rule-weighted disagreement tie-break | OpenRouter | [link](one/reply_mirror_agents/README.md) |
| [`two/`](two/README.md) | L2 | **Two cooperating agents** — Primary (Llama-4-scout-17b) + Reviewer (Qwen3-32b), preprocess phase persists evidence to JSONL | Groq | [link](two/README.md) |
| [`three/`](three/README.md) | L3 | **Three-tier cascade** — rule gate → cheap LLM (Haiku 4.5) → reviewer escalation (Sonnet 4.6), all in parallel batches | OpenRouter | [link](three/README.md) |

## Why `three/` is different

Level 3 added **audio call recordings** to the dataset (folder `audio/*.mp3`, named `YYYYMMDD_HHMMSS-firstname_lastname.mp3`). That changes the problem in two important ways:

1. **A new evidence channel.** Phone calls near a transaction time can be the smoking gun (verification scams, OTP requests, urgent payment instructions). They have to be transcribed and classified before any fraud reasoning can use them.
2. **Volume + cost pressure.** Audio is heavy. If we naively transcribed via an LLM API and then ran the same Primary + Reviewer pattern from levels 1–2, every transaction would touch two paid LLMs *and* an audio model. That blows the cost/latency budget, which the scoring rules explicitly penalise.

So `three/` deliberately re-shapes the architecture:

- **Audio transcribed locally** with `faster-whisper` (`base`, int8, CPU). Zero LLM tokens for audio. Cached to JSON, so reruns skip it.
- **One LLM tier originally** — variant 3 started life as a single LLM (Sonnet 4.6) reviewing every transaction in batches. That worked but was expensive: ~167 Sonnet batches per validation run.
- **Now upgraded to a tiered cascade** while keeping the "single decision-maker per transaction" spirit:
  1. **Deterministic rule gate** absorbs obvious cases (employer salary deposits, zero-anomaly retail, clear IBAN+location+amount mismatches). ~80 % of transactions never reach an LLM.
  2. **Primary LLM = Haiku 4.5** reviews the shortlist (~10× cheaper, ~3× faster than Sonnet) in parallel batches via `ThreadPoolExecutor`.
  3. **Reviewer LLM = Sonnet 4.6** is invoked *only* on borderline-confidence verdicts (typically <30 % of the shortlist → <6 % of total transactions).

Net effect vs. the original single-LLM version: roughly **5 % of the Sonnet calls**, plus parallelism cuts wall-clock further. See [`three/README.md`](three/README.md) for the full diagram.

Variants 1 and 2 keep the classic two-agent shape because their datasets don't include audio and the level budgets allow two LLMs per transaction.

## Repo layout

```
.
├── AIAgentChallenge-ProblemStatement16April.pdf   Official brief
├── README.md                                       (this file)
├── CLAUDE.md                                       Notes for Claude Code sessions
├── demo_lanfuse.py                                 Reference Langfuse + LangChain wiring
├── one/   reply_mirror_agents/                     L1 — 2-agent OpenRouter solution
├── two/   fraud_detection/                         L2 — 2-agent Groq solution + preprocess
└── three/ src/                                     L3 — tiered cascade with audio + Whisper
```

## Setup

A single shared virtualenv at `./.venv/` (Windows). The `.env` at the repo root is loaded by every variant:

```
OPENROUTER_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=Fanatics
# variant 2 only:
GROQ_API_KEY=...
```

Per-variant install + run instructions live in each variant's README.

## Getting the data

The MirrorPay challenge dataset is the property of **Reply** and is not redistributed in this repository.

To run any variant end-to-end you need the per-level dataset folders:

| Variant | Expected location | Files |
|---|---|---|
| L1 | `one/The Truman Show - train/` and `one/The Truman Show - validation/` | `transactions.csv`, `users.json`, `locations.json`, `sms.json`, `mails.json` |
| L2 | `two/train_data/` and `two/validation_data/` | same five files |
| L3 | `three/train_data/` and `three/val_data/` | same five files **plus** `audio/*.mp3` |

For access to the full dataset, contact:

**Uddipan Basu Bir** — [uddipanbb95@gmail.com](mailto:uddipanbb95@gmail.com)

Once dropped in place, follow each variant's README to install dependencies and run.

## Submission artefacts

Each variant writes its official deliverable to `output/<split>_output.txt` (or `train.txt` / `val_fraud.txt` in variant 3) — newline-separated fraudulent `transaction_id`s, exactly as the challenge requires. Per-transaction audits, prediction CSVs, and Langfuse session IDs are emitted alongside.

## Credits

Built by **Uddipan Basu Bir** (team **Fanatics**) for the Reply Mirror AI Agent Challenge, April 2026.
