"""Run the fraud detection pipeline on train and validation splits.

Pipeline (per split):
  1. Load transactions, users, locations, SMS, mails, audio.
  2. Build evidence bundles (audio transcribed locally with faster-whisper).
  3. Rule-based screening → auto-fraud / shortlist / auto-clean.
  4. Cheap LLM (Haiku 4.5) reviews shortlist in parallel batches.
  5. Stronger LLM (Sonnet 4.6) re-checks only borderline verdicts.
  6. Write fraud IDs + audit + Langfuse session.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    OUTPUT_DIR,
    EVIDENCE_DIR,
    WHISPER_MODEL_SIZE,
    RISK_THRESHOLD_LLM,
    RISK_THRESHOLD_AUTO_FRAUD,
    PRIMARY_MODEL,
    REVIEWER_MODEL,
)
from src.data_loader import load_all
from src.preprocess import build_all_evidence, screen_transactions
from src.agent import batch_review, escalate_borderline
from src.tracing import RunTracer


def run_pipeline(data_dir: Path, run_name: str, output_file: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  FRAUD DETECTION PIPELINE: {run_name.upper()}")
    print(f"{'='*60}")

    tracer = RunTracer(run_name)

    # ── 1) Load data ──────────────────────────────────────────────
    print(f"\n[1/5] Loading data from {data_dir.name}...")
    t0 = time.time()
    data = load_all(data_dir)
    n_txn = len(data["transactions"])
    n_audio = len(data["audio_files"])
    print(f"  Loaded: {n_txn} transactions, {len(data['users'])} users, "
          f"{len(data['sms'])} SMS, {len(data['mails'])} mails, "
          f"{len(data['locations'])} locations, {n_audio} audio files")

    # ── 2) Build evidence (cached) ────────────────────────────────
    print(f"\n[2/5] Building evidence bundles...")
    evidence_cache = EVIDENCE_DIR / f"{data_dir.name}_evidence.json"
    evidence = build_all_evidence(data, evidence_cache, WHISPER_MODEL_SIZE)
    preprocess_time = time.time() - t0
    tracer.log_preprocessing(n_txn, n_audio, preprocess_time)
    print(f"  Preprocessing took {preprocess_time:.1f}s")

    # ── 3) Rule-based screening ───────────────────────────────────
    print(f"\n[3/5] Screening with rule-based risk score...")
    shortlist, auto_fraud, auto_clean = screen_transactions(
        evidence,
        threshold_llm=RISK_THRESHOLD_LLM,
        threshold_auto=RISK_THRESHOLD_AUTO_FRAUD,
    )
    tracer.log_screening(len(auto_fraud), len(shortlist), len(auto_clean))
    saved_calls_pct = (1 - len(shortlist) / max(n_txn, 1)) * 100
    print(f"  Skipped {saved_calls_pct:.1f}% of transactions before any LLM call")

    # ── 4) Cheap-model review on shortlist ────────────────────────
    print(f"\n[4/5] Reviewing {len(shortlist)} shortlisted txns with primary model "
          f"({PRIMARY_MODEL})...")
    t2 = time.time()
    primary_verdicts = batch_review(
        shortlist,
        trace_fn=lambda p, r, b, input_tokens=0, output_tokens=0:
            tracer.log_llm_batch(p, r, f"primary-{b}", input_tokens, output_tokens),
    )
    primary_time = time.time() - t2
    print(f"  Primary review took {primary_time:.1f}s")

    # ── 5) Reviewer escalation on borderline confidence ───────────
    print(f"\n[5/5] Escalating borderline verdicts to reviewer model ({REVIEWER_MODEL})...")
    t3 = time.time()
    evidence_by_id = {ev["transaction_id"]: ev for ev in shortlist}
    final_verdicts = escalate_borderline(
        primary_verdicts,
        evidence_by_id,
        trace_fn=lambda p, r, b, input_tokens=0, output_tokens=0:
            tracer.log_llm_batch(p, r, f"reviewer-{b}", input_tokens, output_tokens),
    )
    reviewer_time = time.time() - t3
    print(f"  Reviewer escalation took {reviewer_time:.1f}s")

    # ── Compile fraud IDs ─────────────────────────────────────────
    fraud_ids = set(auto_fraud)
    for v in final_verdicts:
        if v.get("verdict") == "FRAUD":
            fraud_ids.add(v["transaction_id"])
    fraud_ids = sorted(fraud_ids)

    # ── Write output ──────────────────────────────────────────────
    output_path = OUTPUT_DIR / output_file
    output_path.write_text("\n".join(fraud_ids) + ("\n" if fraud_ids else ""), encoding="utf-8")
    print(f"\n  Wrote {len(fraud_ids)} fraud IDs to {output_path.name}")
    print(f"  Fraud rate: {len(fraud_ids)}/{n_txn} = {len(fraud_ids)/n_txn*100:.1f}%")

    audit = {
        "run_name": run_name,
        "n_transactions": n_txn,
        "n_auto_fraud": len(auto_fraud),
        "n_auto_clean": len(auto_clean),
        "n_llm_reviewed": len(shortlist),
        "n_escalated_to_reviewer": sum(1 for v in final_verdicts
                                       if v.get("reason", "").startswith("[reviewer]")),
        "n_fraud_detected": len(fraud_ids),
        "preprocess_time_s": round(preprocess_time, 2),
        "primary_llm_time_s": round(primary_time, 2),
        "reviewer_llm_time_s": round(reviewer_time, 2),
        "fraud_ids": fraud_ids,
        "verdicts": final_verdicts,
    }
    audit_path = OUTPUT_DIR / f"{run_name}_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    tracer.log_final_results(len(fraud_ids), n_txn)
    session_id = tracer.finish()
    print(f"  Langfuse session: {session_id}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    return {"session_id": session_id, "fraud_ids": fraud_ids, "audit": audit}


def main():
    sessions = {}
    result_train = run_pipeline(TRAIN_DIR, "train", "train.txt")
    sessions["train"] = result_train["session_id"]
    result_val = run_pipeline(VAL_DIR, "val", "val_fraud.txt")
    sessions["val"] = result_val["session_id"]

    session_path = OUTPUT_DIR / "langfuse_sessions.txt"
    session_path.write_text(
        f"Train session: {sessions['train']}\nVal session: {sessions['val']}\n",
        encoding="utf-8",
    )

    print(f"\n{'='*60}\n  ALL DONE\n{'='*60}")
    print(f"  Train frauds: {len(result_train['fraud_ids'])}")
    print(f"  Val frauds:   {len(result_val['fraud_ids'])}")
    print(f"  Langfuse sessions saved to {session_path.name}")


if __name__ == "__main__":
    main()
