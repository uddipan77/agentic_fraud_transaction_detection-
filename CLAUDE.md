# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Three sibling Python solutions to the Reply "Mirror Agent" / "Truman Show" fraud-detection hackathon (problem statement PDF lives at the repo root). Each subfolder (`one/`, `two/`, `three/`) is a self-contained pipeline that ingests the same shape of dataset (transactions + users + locations + SMS + mails, plus audio in `three/`) and emits a list of fraudulent transaction IDs.

The three variants are independent — they share only the `.env` file and the `.venv` virtualenv at the repo root. Do **not** factor code across them.

| Variant | Architecture | LLM provider | Distinguishing feature |
|---|---|---|---|
| `one/reply_mirror_agents/` | Primary agent + Reviewer agent, structured evidence tools | OpenRouter (Nemotron / Gemma free tiers) | LangChain + Langfuse `observe`, source-zip archive emitted |
| `two/fraud_detection/` | Primary agent + Reviewer agent (separate preprocess phase) | Groq direct HTTP (`requests`) | Two-phase: `preprocess_split` writes `evidence.jsonl`, then `Orchestrator.run_split` consumes it |
| `three/src/` | **Tiered**: rule-based screen → cheap LLM (Haiku 4.5) → reviewer escalation (Sonnet 4.6) | OpenRouter (via `openai` SDK) | Local audio transcription with `faster-whisper`; evidence + audio both cached; parallel batched LLM via `ThreadPoolExecutor` |

## Common run commands

The shared interpreter is `C:\hackathon reply\.venv\Scripts\python.exe`. Always invoke each variant from inside its own folder so relative imports resolve.

```bash
# Variant 1 — both splits
cd "C:\hackathon reply\one\reply_mirror_agents"
"C:/hackathon reply/.venv/Scripts/python.exe" main.py --dataset all
# single split: --dataset train | --dataset validation

# Variant 2 — both splits by default; pass split names as positional args
cd "C:\hackathon reply\two"
"C:/hackathon reply/.venv/Scripts/python.exe" main.py
"C:/hackathon reply/.venv/Scripts/python.exe" main.py train

# Variant 3 — runs train then validation; no CLI args
cd "C:\hackathon reply\three"
"C:/hackathon reply/.venv/Scripts/python.exe" main.py
```

Install per-variant deps with `python -m pip install -r requirements.txt` from inside that variant's folder. There is no test suite, no linter config, and no build step.

## Environment

A single `.env` at `C:\hackathon reply\.env` is loaded by every variant (each variant computes the path differently — see `config.py` in each folder). Required keys vary:

- **`one/`** and **`three/`**: `OPENROUTER_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `TEAM_NAME`. `one/` will hard-fail in `AppConfig.validate()` if these are missing.
- **`two/`**: `GROQ_API_KEY` (it talks to Groq, not OpenRouter, despite some variable names suggesting otherwise — see `llm_client.py`). `main.py` exits with an error if absent.

Override knobs in `one/reply_mirror_agents/config.py` (thresholds like `REVIEW_THRESHOLD`, `HIGH_VALUE_REVIEW_PROBABILITY`, `DISAGREEMENT_THRESHOLD`, `PRIMARY_MODEL`, `REVIEWER_MODEL`) can be set via env vars without code changes.

## Dataset locations (do not relocate)

Each variant's config hard-codes its dataset paths:

- `one/`: reads from `one/The Truman Show - train/` and `one/The Truman Show - validation/` (note the spaces and capitalization — these are the canonical names from the challenge).
- `two/`: reads from `two/train_data/` and `two/validation_data/`.
- `three/`: reads from `three/train_data/` and `three/val_data/` (note: `val_data`, not `validation_data`), and the train/val folders contain an extra `audio/` subdirectory.

All variants expect the same five files (`transactions.csv`, `users.json`, `locations.json`, `sms.json`, `mails.json`), joined via:
- `transactions.sender_iban` / `recipient_iban` ↔ `users.iban`
- `locations.biotag` → user via city-code embedded in the biotag matched to `users` residence city
- SMS/mail linked to users by full name / first name / email-local-part fuzzy match

See `one/reply_mirror_agents/notes/linking_logic.md` for the canonical schema notes.

## Output convention

All variants write a newline-separated list of fraudulent transaction IDs:
- `one/`: `output/{train,validation}_output.txt`, plus `predictions_{split}.csv`, `langfuse_sessions.json`, and a source-zip (`reply_mirror_agents_source.zip`) emitted by `output_writer.create_source_archive`.
- `two/`: `output/{split}_output.txt`, `{split}_predictions.csv`, `{split}_audit.json`.
- `three/`: `output/train.txt` and `output/val_fraud.txt` (asymmetric names — the val file is `val_fraud.txt`, not `val.txt`), plus per-split `*_audit.json` and `langfuse_sessions.txt`.

## Decision-policy specifics worth knowing before editing

**Variants `one/` and `two/` share the same two-agent shape but differ in tie-break logic:**
- `one/orchestrator.py::_combine` does a weighted blend on disagreement: `0.45·rule_risk + 0.35·primary_p(fraud) + 0.20·reviewer_p(fraud)`, with a `+0.05` nudge for high-economic-impact transactions; final fraud iff blend ≥ `disagreement_threshold` (default 0.62).
- `two/orchestrator.py::make_final_decision` picks the higher-confidence agent if the gap exceeds 0.1; on a near-tie it leans **fraud** for high-value cases and **not_fraud** otherwise.

**Variant `three/`** uses a three-tier cascade: `screen_transactions` in `src/preprocess.py` splits txns into `auto_fraud / shortlist / auto_clean` by deterministic risk score; only the shortlist hits the LLM. The shortlist is reviewed by `PRIMARY_MODEL` (Haiku 4.5) in parallel batches (`MAX_PARALLEL_BATCHES = 4`); only verdicts with confidence below `ESCALATION_CONFIDENCE` (0.70) are re-reviewed by `REVIEWER_MODEL` (Sonnet 4.6). Evidence is cached to `three/evidence/{split_name}_evidence.json` and audio transcripts to `{split_name}_audio.json`; delete those to force re-preprocessing (the Whisper pass is the slow step).

## Langfuse tracing pattern

`one/` and `three/` both wire Langfuse and emit per-run session IDs the user submits with their entry. When editing tracing code, preserve the session-id format `{TEAM_NAME}-{ulid}` (see `demo_lanfuse.py` at the repo root for the canonical reference) and remember to `langfuse_client.flush()` before process exit or traces will be lost.
