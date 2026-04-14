from __future__ import annotations

from typing import Any, Optional

from ai_supervisor.adapters.base import MessengerAdapter


class TelegramAdapterStub(MessengerAdapter):
    """Заглушка под будущую интеграцию Telegram Bot API."""

    name = "telegram"

    async def send_text(
        self, *, chat_id: Optional[int], user_id: Optional[int], text: str
    ) -> None:
        raise NotImplementedError("Telegram: не реализовано в рамках текущего ТЗ")

    def normalize_incoming(self, raw: dict[str, Any]) -> Optional[dict[str, Any]]:
        return None
