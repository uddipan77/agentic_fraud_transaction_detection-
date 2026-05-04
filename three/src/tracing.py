"""Langfuse tracing integration (v3 API)."""
import ulid
from langfuse import Langfuse
from .config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST, TEAM_NAME


def get_langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )


def generate_session_id(suffix: str = "") -> str:
    tag = f"{TEAM_NAME}-{suffix}" if suffix else TEAM_NAME
    return f"{tag}-{ulid.new().str}"


class RunTracer:
    """Manages Langfuse tracing for a full dataset run using v3 span API."""

    def __init__(self, run_name: str):
        self.client = get_langfuse_client()
        self.session_id = generate_session_id(run_name)
        # Root span for the entire run
        self.root_span = self.client.start_span(
            name=f"fraud-detection-{run_name}",
            metadata={"run_type": run_name, "team": TEAM_NAME,
                       "session_id": self.session_id},
        )
        # Update trace-level session
        self.root_span.update_trace(
            session_id=self.session_id,
            name=f"fraud-detection-{run_name}",
        )

    def log_preprocessing(self, n_transactions: int, n_audio: int, duration_s: float):
        span = self.root_span.start_span(
            name="preprocessing",
            metadata={
                "n_transactions": n_transactions,
                "n_audio": n_audio,
                "duration_s": round(duration_s, 2),
            },
        )
        span.end()

    def log_screening(self, n_auto_fraud: int, n_shortlist: int, n_auto_clean: int):
        span = self.root_span.start_span(
            name="fast-screening",
            metadata={
                "n_auto_fraud": n_auto_fraud,
                "n_shortlist": n_shortlist,
                "n_auto_clean": n_auto_clean,
            },
        )
        span.end()

    def log_llm_batch(self, prompt: str, response: str, batch_num: int,
                      input_tokens: int = 0, output_tokens: int = 0):
        gen = self.root_span.start_generation(
            name=f"agent-batch-{batch_num}",
            model="anthropic/claude-sonnet-4.6",
            input=prompt[:5000],
            output=response[:3000],
            metadata={"batch_num": batch_num},
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
            },
        )
        gen.end()

    def log_final_results(self, n_fraud: int, n_total: int):
        span = self.root_span.start_span(
            name="final-output",
            metadata={
                "n_fraud_detected": n_fraud,
                "n_total_transactions": n_total,
                "fraud_rate": round(n_fraud / n_total * 100, 2) if n_total > 0 else 0,
            },
        )
        span.end()

    def finish(self) -> str:
        self.root_span.end()
        self.client.flush()
        return self.session_id
