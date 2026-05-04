"""Configuration for the fraud detection system."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env from project root
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
_env_vars: dict[str, str] = {}
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            _env_vars[k.strip()] = v.strip()
            os.environ[k.strip()] = v.strip()  # Force override


@dataclass
class Config:
    # Models
    primary_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    reviewer_model: str = "qwen/qwen3-32b"

    # API
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # LLM parameters
    primary_temperature: float = 0.0
    reviewer_temperature: float = 0.0
    max_tokens_primary: int = 700
    max_tokens_reviewer: int = 500

    # Thresholds
    review_confidence_threshold: float = 0.75
    high_value_review_threshold: float = 0.80
    fraud_decision_threshold: float = 0.50

    # Paths
    base_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parent.parent))

    # Processing
    batch_size: int = 10
    max_retries: int = 5
    retry_delay: float = 5.0

    # Location matching
    location_window_hours: float = 24.0
    location_distance_km: float = 50.0

    # Message lookback
    message_lookback_days: int = 7

    def dataset_path(self, split: str) -> Path:
        return Path(self.base_dir) / f"{split}_data"

    def output_dir(self) -> Path:
        return Path(self.base_dir) / "output"

    def evidence_dir(self, split: str) -> Path:
        d = Path(self.base_dir) / "evidence" / split
        d.mkdir(parents=True, exist_ok=True)
        return d
