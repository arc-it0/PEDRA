"""Structured LLM calls compatible with the Syntax LiteLLM proxy."""
import json
import re
from .config import settings

class LLMUnavailable(RuntimeError):
    pass

class LLMClient:
    def __init__(self):
        self.enabled = bool(settings.api_key and settings.model)
        self._client = None

    def complete(self, system: str, user: str, max_tokens: int = 1600) -> str:
        if not self.enabled:
            raise LLMUnavailable("OPENAI_API_KEY and OPENAI_MODEL must be configured for LLM responses")
        if self._client is None:
            options = {"api_key": settings.api_key, "timeout": 35.0, "max_retries": 1}
            if settings.base_url:
                options["base_url"] = settings.base_url
            from openai import OpenAI
            self._client = OpenAI(**options)
        try:
            response = self._client.chat.completions.create(
                model=settings.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0, max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMUnavailable(f"Chat completion failed: {type(exc).__name__}") from exc

    def json(self, system: str, user: str) -> dict:
        raw = self.complete(system + "\nReturn strictly one JSON object; no Markdown fences.", user)
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise ValueError("LLM did not return JSON")
        obj = json.loads(match.group())
        if not isinstance(obj, dict):
            raise ValueError("LLM JSON must be an object")
        return obj
