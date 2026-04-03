from __future__ import annotations

import logging
from datetime import datetime, timezone

from aiomax import Bot

from ai_supervisor.analysis import AnalysisResult
from ai_supervisor.storage import SupervisorStorage

logger = logging.getLogger(__name__)


def _msk_ts(ts: int | None) -> str:
    if ts is None:
        return "—"
    try:
        sec = float(ts) / 1000.0 if ts > 1_000_000_000_000 else float(ts)
        dt = datetime.fromtimestamp(sec, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (OSError, OverflowError, ValueError):
        return str(ts)


def build_alert_text(
    *,
    chat_id: int,
    chat_title: str,
    message_ts: int | None,
    result: AnalysisResult,
) -> str:
    sev = result.severity.upper()
    ev = "\n".join(f"• {e}" for e in (result.evidence or [])[:5]) or "—"
    lines = [
        "**AI Supervisor — алерт**",
        f"**Чат:** {chat_title} (`{chat_id}`)",
        f"**Время сообщения:** {_msk_ts(message_ts)}",
        f"**Категория:** {result.category}  |  **Важность:** {sev}",
        f"**Заголовок:** {result.title}",
        "",
        f"**Суть:** {result.summary}",
        "",
        f"**Участники / роли:** {result.who or '—'}",
        "",
        "**Опора на переписку:**",
        ev,
    ]
    return "\n".join(lines)


class NotificationDispatcher:
    def __init__(self, storage: SupervisorStorage, bot: Bot) -> None:
        self._storage = storage
        self._bot = bot

    async def dispatch(self, *, chat_id: int, chat_title: str, message_ts: int | None, result: AnalysisResult) -> None:
        text = build_alert_text(
            chat_id=chat_id,
            chat_title=chat_title,
            message_ts=message_ts,
            result=result,
        )
        mc = self._storage.get_manager_chat_id()
        if mc is not None:
            try:
                await self._bot.send_message(text=text, chat_id=mc, format="markdown")
            except Exception:
                logger.exception("Не удалось отправить алерт в чат менеджеров")

        for uid in self._storage.list_duty_users():
            try:
                await self._bot.send_message(text=text, user_id=uid, format="markdown")
            except Exception:
                logger.exception("Не удалось отправить личное уведомление user_id=%s", uid)
