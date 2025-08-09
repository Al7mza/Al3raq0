from __future__ import annotations

import os
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("openai package is required. Please install dependencies.") from exc


class LLMClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Please configure your environment.")

        base_url = os.getenv("OPENAI_BASE_URL") or None
        # If using a custom base URL (e.g., OpenRouter), you may also need to set additional headers
        # You can set them via environment variables in the future if needed.
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, system_prompt: str, history_messages: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": system_prompt}] + history_messages

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            presence_penalty=0.2,
            frequency_penalty=0.2,
        )
        text = response.choices[0].message.content or ""
        return text.strip()