from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol


class OutgoingMessage(Protocol):
    chat_id: Optional[int]
    user_id: Optional[int]
    text: str


class MessengerAdapter(ABC):
    """Будущая унификация для Telegram и других каналов."""

    name: str = "abstract"

    @abstractmethod
    async def send_text(
        self, *, chat_id: Optional[int], user_id: Optional[int], text: str
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def normalize_incoming(self, raw: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Приводит входящее событие к виду, похожему на MAX Update, или None если не поддерживается."""
        raise NotImplementedError
