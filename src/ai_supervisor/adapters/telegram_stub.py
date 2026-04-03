from __future__ import annotations

from typing import Any

from ai_supervisor.adapters.base import MessengerAdapter


class TelegramAdapterStub(MessengerAdapter):
    """Заглушка под будущую интеграцию Telegram Bot API."""

    name = "telegram"

    async def send_text(self, *, chat_id: int | None, user_id: int | None, text: str) -> None:
        raise NotImplementedError("Telegram: не реализовано в рамках текущего ТЗ")

    def normalize_incoming(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        return None
