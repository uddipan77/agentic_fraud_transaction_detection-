"""Configuration for fraud detection system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = BASE_DIR / "train_data"
VAL_DIR = BASE_DIR / "val_data"
OUTPUT_DIR = BASE_DIR / "output"
EVIDENCE_DIR = BASE_DIR / "evidence"

OUTPUT_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR.mkdir(exist_ok=True)

# ── API ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Two-tier model strategy:
#   PRIMARY  – cheap & fast, reviews every shortlisted transaction
#   REVIEWER – stronger model, only re-checks borderline verdicts
PRIMARY_MODEL = "anthropic/claude-haiku-4.5"
REVIEWER_MODEL = "anthropic/claude-sonnet-4.6"

# ── Langfuse ───────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
TEAM_NAME = os.getenv("TEAM_NAME", "Fanatics").replace(" ", "-")

# ── Screening thresholds (rule-based gate before any LLM call) ─────────
RISK_THRESHOLD_LLM = 0.30        # below this → auto-clean (no LLM)
RISK_THRESHOLD_AUTO_FRAUD = 0.85 # above this → auto-fraud (no LLM)

# ── LLM batching & concurrency ────────────────────────────────────────
BATCH_SIZE = 12                  # transactions per LLM batch call
MAX_PARALLEL_BATCHES = 4         # threadpool size for batch dispatch
ESCALATION_CONFIDENCE = 0.70     # primary verdict below this → escalate to reviewer

# ── Audio ──────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"      # faster-whisper model size
