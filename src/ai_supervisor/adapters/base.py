from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class OutgoingMessage(Protocol):
    chat_id: int | None
    user_id: int | None
    text: str


class MessengerAdapter(ABC):
    """Будущая унификация для Telegram и других каналов."""

    name: str = "abstract"

    @abstractmethod
    async def send_text(self, *, chat_id: int | None, user_id: int | None, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def normalize_incoming(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Приводит входящее событие к виду, похожему на MAX Update, или None если не поддерживается."""
        raise NotImplementedError
