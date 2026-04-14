from __future__ import annotations

import logging
from typing import Any, Optional

from aiomax import Bot

from ai_supervisor.storage import SupervisorStorage

logger = logging.getLogger(__name__)


class SupervisorBot(Bot):
    """
    Расширение aiomax.Bot: long polling с timeout, сохранение marker в SQLite,
    те же параметры, что и у REST GET /updates.
    """

    def __init__(
        self,
        storage: SupervisorStorage,
        *,
        poll_timeout: int,
        poll_limit: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._storage = storage
        self._poll_timeout = poll_timeout
        self._poll_limit = poll_limit

    def restore_marker(self) -> None:
        m = self._storage.get_marker()
        if m is not None:
            self.marker = m
            logger.info("Восстановлен marker long polling: %s", m)

    async def get_updates(self, limit: Optional[int] = None) -> dict:
        lim = self._poll_limit if limit is None else limit
        payload: dict[str, Any] = {"limit": lim, "marker": self.marker}
        if self._poll_timeout and self._poll_timeout > 0:
            payload["timeout"] = self._poll_timeout
        payload = {k: v for k, v in payload.items() if v is not None}
        logger.info(
            "MAX: poll cycle -> marker=%s limit=%s timeout=%s",
            payload.get("marker"),
            payload.get("limit"),
            payload.get("timeout", 0),
        )
        response = await self.get(
            "https://platform-api.max.ru/updates", params=payload
        )
        data = await response.json()
        try:
            n = len(data.get("updates") or [])
        except Exception:
            n = -1
        if "marker" in data:
            self.marker = data["marker"]
        logger.info("MAX: poll cycle <- updates=%s marker=%s", n, self.marker)
        if self.marker is not None:
            self._storage.set_marker(self.marker)
        return data
