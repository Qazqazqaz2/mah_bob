from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional, Set

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_access_token: str

    # Однократный импорт в SQLite при первом запуске БД (см. seed_runtime_from_env).
    # Дальше настройки в боте / в таблицах duty_users, monitored_chats, kv.
    monitored_chat_ids: str = ""

    manager_chat_id: Optional[int] = Field(default=None)
    duty_user_ids: str = ""

    # Кто может менять настройки в ЛС (через запятую). Пусто = любой пользователь в ЛС.
    bot_admin_user_ids: str = ""

    llm_provider: Literal["yandex", "gigachat"] = "yandex"

    yandex_iam_token: str = ""
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_model_uri: str = ""
    yandex_temperature: float = Field(default=0.15, ge=0.0, le=2.0)
    yandex_max_tokens: int = Field(default=1800, ge=64, le=8000)

    gigachat_credentials: str = ""

    sqlite_path: str = Field(
        default="./data/supervisor.sqlite",
        validation_alias=AliasChoices("SQLITE_PATH", "SUPERVISOR_DB_PATH"),
    )
    context_window_messages: int = Field(
        default=40,
        ge=5,
        le=200,
        validation_alias=AliasChoices("CONTEXT_WINDOW_MESSAGES", "SUPERVISOR_CONTEXT_MESSAGES"),
    )
    analysis_debounce_seconds: float = Field(
        default=0.0,
        ge=0,
        validation_alias=AliasChoices("ANALYSIS_DEBOUNCE_SECONDS", "SUPERVISOR_ANALYZE_DEBOUNCE_SECONDS"),
    )
    analysis_min_interval_seconds: float = Field(
        default=0.0,
        ge=0,
        validation_alias=AliasChoices(
            "ANALYSIS_MIN_INTERVAL_SECONDS",
            "SUPERVISOR_ANALYZE_MIN_INTERVAL_SECONDS",
        ),
    )
    analysis_max_wait_seconds: float = Field(
        default=0.0,
        ge=0,
        validation_alias=AliasChoices(
            "ANALYSIS_MAX_WAIT_SECONDS",
            "SUPERVISOR_ANALYZE_MAX_WAIT_SECONDS",
        ),
    )

    long_poll_timeout: int = Field(default=45, ge=0, le=90)
    long_poll_limit: int = Field(default=100, ge=1, le=1000)

    log_level: str = "INFO"

    @field_validator("manager_chat_id", mode="before")
    @classmethod
    def empty_manager_chat(cls, v: object) -> object:
        if v is None or v == "":
            return None
        return v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_llm_provider(cls, v: object) -> str:
        s = str(v).strip().lower() if v is not None else "yandex"
        if s in {"yandexgpt", "yandex-gpt"}:
            return "yandex"
        return s

    @field_validator("monitored_chat_ids", "duty_user_ids", "bot_admin_user_ids", mode="before")
    @classmethod
    def strip_str(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v).strip()

    def monitored_chat_id_set(self) -> Optional[Set[int]]:
        raw = self.monitored_chat_ids.replace(" ", "")
        if not raw:
            return None
        return {int(x) for x in raw.split(",") if x}

    def duty_user_id_list(self) -> list[int]:
        raw = self.duty_user_ids.replace(" ", "")
        if not raw:
            return []
        return [int(x) for x in raw.split(",") if x]

    def bot_admin_id_set(self) -> Optional[Set[int]]:
        raw = self.bot_admin_user_ids.replace(" ", "")
        if not raw:
            return None
        return {int(x) for x in raw.split(",") if x}

@lru_cache
def get_settings() -> Settings:
    return Settings()
