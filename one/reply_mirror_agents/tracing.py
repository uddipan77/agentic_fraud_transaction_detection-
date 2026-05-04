from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator

from langfuse import Langfuse

from config import AppConfig
from utils import make_session_id


@dataclass(slots=True)
class RunTrace:
    dataset_name: str
    session_id: str
    trace_id: str | None


class TraceManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
            tracing_enabled=bool(
                config.langfuse_public_key and config.langfuse_secret_key
            ),
        )

    @contextmanager
    def start_run(
        self,
        dataset_name: str,
        *,
        input_payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[RunTrace]:
        session_id = make_session_id(self.config.team_name, dataset_name)
        trace_id = self.client.create_trace_id(seed=session_id)
        trace_context = {"trace_id": trace_id, "session_id": session_id}
        run_metadata = {"dataset": dataset_name, **(metadata or {})}
        with self.client.start_as_current_observation(
            trace_context=trace_context,
            name=f"{dataset_name}_run",
            as_type="chain",
            input=input_payload,
            metadata=run_metadata,
        ):
            yield RunTrace(dataset_name=dataset_name, session_id=session_id, trace_id=trace_id)
        self.client.flush()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
        input_payload: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        context = self.client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input_payload,
            metadata=metadata,
        )
        with context:
            yield

    def event(
        self,
        name: str,
        *,
        input_payload: Any = None,
        output_payload: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.client.create_event(
            name=name,
            input=input_payload,
            output=output_payload,
            metadata=metadata,
        )

    def trace_url(self, trace_id: str | None) -> str | None:
        if not trace_id:
            return None
        try:
            return self.client.get_trace_url(trace_id=trace_id)
        except Exception:
            return None

    def flush(self) -> None:
        self.client.flush()