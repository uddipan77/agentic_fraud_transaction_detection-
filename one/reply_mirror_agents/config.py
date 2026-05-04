from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DATA_ROOT = WORKSPACE_ROOT / "one"
TRAIN_DATASET = DATA_ROOT / "The Truman Show - train"
VALIDATION_DATASET = DATA_ROOT / "The Truman Show - validation"
ENV_FILE = WORKSPACE_ROOT / ".env"

load_dotenv(ENV_FILE)


@dataclass(slots=True)
class AppConfig:
    primary_model: str = os.getenv(
        "PRIMARY_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
    )
    reviewer_model: str = os.getenv(
        "REVIEWER_MODEL", "google/gemma-4-31b-it:free"
    )
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    langfuse_public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host: str = os.getenv(
        "LANGFUSE_HOST", "https://challenges.reply.com/langfuse"
    )
    team_name: str = os.getenv("TEAM_NAME", "reply-mirror")
    review_threshold: float = float(os.getenv("REVIEW_THRESHOLD", "0.75"))
    high_value_review_probability: float = float(
        os.getenv("HIGH_VALUE_REVIEW_PROBABILITY", "0.45")
    )
    disagreement_threshold: float = float(
        os.getenv("DISAGREEMENT_THRESHOLD", "0.62")
    )
    location_window_hours: int = int(os.getenv("LOCATION_WINDOW_HOURS", "36"))
    message_lookback_days: int = int(os.getenv("MESSAGE_LOOKBACK_DAYS", "21"))
    amount_high_impact_floor: float = float(
        os.getenv("AMOUNT_HIGH_IMPACT_FLOOR", "2000")
    )
    amount_high_impact_monthly_salary_factor: float = float(
        os.getenv("AMOUNT_HIGH_IMPACT_MONTHLY_SALARY_FACTOR", "0.75")
    )
    primary_temperature: float = float(os.getenv("PRIMARY_TEMPERATURE", "0.0"))
    reviewer_temperature: float = float(os.getenv("REVIEWER_TEMPERATURE", "0.0"))
    max_tokens_primary: int = int(os.getenv("MAX_TOKENS_PRIMARY", "650"))
    max_tokens_reviewer: int = int(os.getenv("MAX_TOKENS_REVIEWER", "450"))
    output_dir: Path = PROJECT_ROOT / "output"
    train_dataset_path: Path = TRAIN_DATASET
    validation_dataset_path: Path = VALIDATION_DATASET

    def validate(self) -> None:
        missing = []
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if not self.langfuse_public_key:
            missing.append("LANGFUSE_PUBLIC_KEY")
        if not self.langfuse_secret_key:
            missing.append("LANGFUSE_SECRET_KEY")
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required environment variables: {joined}")

    def dataset_path(self, split: str) -> Path:
        normalized = split.lower()
        if normalized == "train":
            return self.train_dataset_path
        if normalized == "validation":
            return self.validation_dataset_path
        raise ValueError(f"Unsupported dataset split: {split}")


def load_config() -> AppConfig:
    config = AppConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config