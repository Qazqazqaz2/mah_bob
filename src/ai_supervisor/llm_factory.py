from __future__ import annotations

import logging

from ai_supervisor.config import Settings
from ai_supervisor.llm_base import LLMClient
from ai_supervisor.llm_gigachat import GigaChatLLMClient
from ai_supervisor.llm_yandex import YandexGPTClient

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "yandex":
        if not settings.yandex_folder_id:
            raise ValueError("YandexGPT: задайте YANDEX_FOLDER_ID")
        has_iam = bool((settings.yandex_iam_token or "").strip())
        has_key = bool((settings.yandex_api_key or "").strip())
        if has_iam == has_key:
            raise ValueError("YandexGPT: укажите ровно один из YANDEX_IAM_TOKEN или YANDEX_API_KEY")
        model_uri = settings.yandex_model_uri.strip()
        if not model_uri:
            model_uri = f"gpt://{settings.yandex_folder_id}/yandexgpt/latest"
        logger.info("LLM: YandexGPT, modelUri=%s", model_uri)
        return YandexGPTClient(
            iam_token=settings.yandex_iam_token if has_iam else None,
            api_key=settings.yandex_api_key if has_key else None,
            folder_id=settings.yandex_folder_id,
            model_uri=model_uri,
            temperature=settings.yandex_temperature,
            max_tokens=settings.yandex_max_tokens,
        )
    if not settings.gigachat_credentials:
        raise ValueError("GigaChat: задайте GIGACHAT_CREDENTIALS")
    logger.info("LLM: GigaChat")
    return GigaChatLLMClient(credentials=settings.gigachat_credentials)
