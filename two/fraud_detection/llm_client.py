"""LLM client for calling OpenRouter API."""

import re
import time
import requests
from typing import Any

from .config import Config


class LLMClient:
    """Simple HTTP-based LLM client for Groq API."""

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.groq_base_url
        self.api_key = config.groq_api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make a single LLM API call with retries."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.config.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )
                if resp.status_code == 429:
                    # Exponential backoff: 10s, 20s, 40s, 60s, 60s
                    wait = min(60, 10 * (2 ** attempt))
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt+1})...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    # Strip <think>...</think> tags from models like qwen
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    return content
                elif "error" in data:
                    err_msg = data["error"].get("message", str(data["error"]))
                    print(f"  API error: {err_msg}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                        continue
                    return ""
                else:
                    return ""

            except requests.exceptions.Timeout:
                print(f"  Timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
            except requests.exceptions.RequestException as e:
                print(f"  Request error: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue

        return ""

    def invoke_primary(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call the primary agent model."""
        return self._call(
            model=self.config.primary_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.primary_temperature,
            max_tokens=self.config.max_tokens_primary,
        )

    def invoke_reviewer(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call the reviewer agent model."""
        return self._call(
            model=self.config.reviewer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.reviewer_temperature,
            max_tokens=self.config.max_tokens_reviewer,
        )
