from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

from config import AppConfig
from tracing import TraceManager


class LLMClient:
    def __init__(self, config: AppConfig, tracer: TraceManager) -> None:
        self.config = config
        self.tracer = tracer
        self.primary_model = ChatOpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            model=config.primary_model,
            temperature=config.primary_temperature,
            max_tokens=config.max_tokens_primary,
        )
        self.reviewer_model = ChatOpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            model=config.reviewer_model,
            temperature=config.reviewer_temperature,
            max_tokens=config.max_tokens_reviewer,
        )

    def _invoke(
        self,
        *,
        model: ChatOpenAI,
        system_prompt: str,
        user_prompt: str,
        session_id: str,
        dataset_name: str,
        transaction_id: str,
        agent_name: str,
    ) -> str:
        metadata = {
            "langfuse_session_id": session_id,
            "langfuse_trace_name": f"{dataset_name}_{agent_name}",
            "dataset": dataset_name,
            "transaction_id": transaction_id,
            "agent_name": agent_name,
        }
        with self.tracer.span(
            f"{agent_name}_call",
            as_type="agent",
            input_payload={"transaction_id": transaction_id, "dataset": dataset_name},
            metadata={"model": model.model_name},
        ):
            response = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
                config={
                    "callbacks": [CallbackHandler()],
                    "metadata": metadata,
                },
            )
        return str(response.content)

    def invoke_primary(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        session_id: str,
        dataset_name: str,
        transaction_id: str,
    ) -> str:
        return self._invoke(
            model=self.primary_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_id=session_id,
            dataset_name=dataset_name,
            transaction_id=transaction_id,
            agent_name="primary_investigator",
        )

    def invoke_reviewer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        session_id: str,
        dataset_name: str,
        transaction_id: str,
    ) -> str:
        return self._invoke(
            model=self.reviewer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_id=session_id,
            dataset_name=dataset_name,
            transaction_id=transaction_id,
            agent_name="reviewer",
        )