from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from ai_supervisor.llm_base import LLMClient

logger = logging.getLogger(__name__)

_last_403_detail_log_mono: float = 0.0
_403_LOG_INTERVAL_SEC = 600.0

YANDEX_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


class YandexGPTClient(LLMClient):
    """YandexGPT Foundation Models API (REST). IAM-токен или API-ключ Yandex Cloud."""

    def __init__(
        self,
        *,
        folder_id: str,
        model_uri: str,
        iam_token: Optional[str] = None,
        api_key: Optional[str] = None,
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
        global _last_403_detail_log_mono
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
                now = time.monotonic()
                if r.status_code == 403 and (
                    now - _last_403_detail_log_mono >= _403_LOG_INTERVAL_SEC
                ):
                    _last_403_detail_log_mono = now
                    logger.error("YandexGPT error %s: %s", r.status_code, r.text[:800])
                elif r.status_code != 403:
                    logger.error("YandexGPT error %s: %s", r.status_code, r.text[:800])
                else:
                    logger.debug("YandexGPT 403 (детали в логе не чаще чем раз в %s с)", int(_403_LOG_INTERVAL_SEC))
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
