from __future__ import annotations

import logging
from typing import Any

import httpx

from ai_supervisor.llm_base import LLMClient

logger = logging.getLogger(__name__)

YANDEX_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


class YandexGPTClient(LLMClient):
    """YandexGPT Foundation Models API (REST). IAM-токен или API-ключ Yandex Cloud."""

    def __init__(
        self,
        *,
        folder_id: str,
        model_uri: str,
        iam_token: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.15,
        max_tokens: int = 1800,
    ) -> None:
        if bool(iam_token) == bool(api_key):
            raise ValueError("Укажите ровно один из: YANDEX_IAM_TOKEN или YANDEX_API_KEY")
        self._iam_token = (iam_token or "").strip()
        self._api_key = (api_key or "").strip()
        self._folder_id = folder_id
        self._model_uri = model_uri
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _auth_headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Api-Key {self._api_key}"}
        return {"Authorization": f"Bearer {self._iam_token}"}

    async def complete(self, *, system: str, user: str) -> str:
        headers = {
            **self._auth_headers(),
            "x-folder-id": self._folder_id,
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self._temperature,
                "maxTokens": self._max_tokens,
            },
            "messages": [
                {"role": "system", "text": system},
                {"role": "user", "text": user},
            ],
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(YANDEX_COMPLETION_URL, headers=headers, json=body)
            if r.status_code >= 400:
                logger.error("YandexGPT error %s: %s", r.status_code, r.text)
            r.raise_for_status()
            data = r.json()
        return _extract_yandex_text(data)


def _extract_yandex_text(data: dict[str, Any]) -> str:
    result = data.get("result") or {}
    alts = result.get("alternatives") or []
    if not alts:
        raise ValueError(f"YandexGPT: пустой ответ: {data!r}")
    first = alts[0]
    msg = first.get("message") or {}
    text = msg.get("text")
    if text is None:
        text = first.get("text")
    if text is None:
        raise ValueError(f"YandexGPT: нет текста: {data!r}")
    return str(text)
