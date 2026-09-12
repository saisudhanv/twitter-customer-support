"""Small provider client for generating support replies."""

from __future__ import annotations

import requests

from src.config import Config, get_config


class LLMError(RuntimeError):
    """Raised when the configured LLM cannot generate a response."""


class GeminiClient:
    """Generate text with Google's Gemini generateContent REST endpoint."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        if not self.config.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not configured")

    def generate(self, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent"
        )
        response = requests.post(
            url,
            params={"key": self.config.gemini_api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        if not response.ok:
            raise LLMError(f"Gemini request failed with HTTP {response.status_code}")

        payload = response.json()
        try:
            return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Gemini returned no text") from exc