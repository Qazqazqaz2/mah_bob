from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from aiomax import Bot
from aiomax.types import Message as MaxMessage

from ai_supervisor import bot_ui
from ai_supervisor.analysis import AnalysisResult, analyze_transcript, format_transcript
from ai_supervisor.config import Settings
from ai_supervisor.llm_base import LLMClient
from ai_supervisor.notifier import NotificationDispatcher
from ai_supervisor.storage import StoredLine, SupervisorStorage

logger = logging.getLogger(__name__)


def _is_probably_group_chat(chat: dict[str, Any]) -> bool:
    return chat.get("type") != "dialog"


def _sender_display_name(sender: Any) -> str:
    if sender is None:
        return "unknown"
    name = getattr(sender, "name", None) or getattr(sender, "first_name", None)
    if name:
        return str(name)
    if getattr(sender, "username", None):
        return f"@{sender.username}"
    return f"user_{sender.user_id}"


class ChatSupervisor:
    """Обработка входящих сообщений (aiomax): контекст → LLM → уведомления."""

    def __init__(
        self,
        *,
        settings: Settings,
        storage: SupervisorStorage,
        llm: LLMClient,
        bot: Bot,
        notifier: NotificationDispatcher,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._llm = llm
        self._bot = bot
        self._notifier = notifier
        self._chat_cache: dict[int, dict[str, Any]] = {}
        self._debounce_tasks: dict[int, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._dm_hint_shown: set[int] = set()
        self._dm_menu_throttle: dict[int, float] = {}

    def note_user_greeted_on_start(self, user_id: int) -> None:
        self._dm_hint_shown.add(user_id)

    async def send_dm_main_menu(
        self,
        chat_id: int,
        user_id: int,
        *,
        force: bool = False,
        min_interval_sec: float = 14.0,
    ) -> None:
        now = time.monotonic()
        if not force:
            last = self._dm_menu_throttle.get(user_id, 0.0)
            if now - last < min_interval_sec:
                return
        self._dm_menu_throttle[user_id] = now
        try:
            await self._bot.send_message(
                text=bot_ui.TEXT_MAIN,
                chat_id=chat_id,
                format="markdown",
                keyboard=bot_ui.keyboard_main(),
            )
        except Exception:
            logger.exception("Не удалось отправить главное меню в chat_id=%s", chat_id)

    async def _handle_dialog_message(self, message: MaxMessage, chat_id: int) -> None:
        sender = message.sender
        if sender is not None and sender.is_bot:
            return
        uid = sender.user_id if sender else None
        if uid is None:
            return

        body_text = ""
        if message.body and message.body.text:
            body_text = message.body.text.strip()
        low = body_text.lower()
        explicit = low in {"/start", "/help", "start", "help", "/начать"} or low.startswith("/help")

        if explicit:
            try:
                await self._bot.send_message(
                    text=bot_ui.TEXT_ABOUT,
                    chat_id=chat_id,
                    format="markdown",
                    keyboard=bot_ui.keyboard_with_back(),
                )
            except Exception:
                logger.exception("Не удалось отправить раздел «О боте»")
            return

        if not body_text:
            if uid not in self._dm_hint_shown:
                self._dm_hint_shown.add(uid)
                await self.send_dm_main_menu(chat_id, uid, force=True)
            return

        if uid not in self._dm_hint_shown:
            self._dm_hint_shown.add(uid)
            await self.send_dm_main_menu(chat_id, uid, force=True)
            return

        await self.send_dm_main_menu(chat_id, uid, force=False)

    def _label_for_chat(self, chat_id: int) -> str:
        c = self._chat_cache.get(chat_id)
        if not c:
            return f"Чат {chat_id}"
        t = c.get("title")
        if t:
            return str(t)
        return f"Чат {chat_id}"

    async def _ensure_chat(self, chat_id: int) -> dict[str, Any]:
        if chat_id in self._chat_cache:
            return self._chat_cache[chat_id]
        try:
            chat = await self._bot.get_chat(chat_id)
            info = {"chat_id": chat.chat_id, "title": chat.title, "type": chat.type}
        except Exception:
            logger.exception("Не удалось получить чат %s", chat_id)
            info = {"chat_id": chat_id, "title": None, "type": "chat"}
        self._chat_cache[chat_id] = info
        return info

    async def handle_aiomax_message(self, message: MaxMessage) -> None:
        rec = message.recipient
        if rec is None or rec.chat_id is None:
            return
        chat_id = rec.chat_id

        chat_type = getattr(rec, "chat_type", None)
        if chat_type == "dialog":
            await self._handle_dialog_message(message, chat_id)
            return

        chat = await self._ensure_chat(chat_id)
        if not _is_probably_group_chat(chat):
            await self._handle_dialog_message(message, chat_id)
            return

        self._storage.touch_known_chat(chat_id, chat.get("title"))

        monitored = self._storage.monitored_chat_filter()
        if monitored is not None and chat_id not in monitored:
            return

        sender = message.sender
        if sender is not None and sender.is_bot:
            return

        if message.body is None or not message.body.text:
            return
        text = message.body.text.strip()
        if not text:
            return

        ts_raw = message.timestamp
        ts_ms = int((ts_raw or 0) * 1000) if ts_raw else 0
        sid = sender.user_id if sender else None
        sname = _sender_display_name(sender)

        self._storage.append_message(
            chat_id=chat_id,
            ts=ts_ms,
            sender_id=sid,
            sender_name=sname,
            text=text,
            keep_last=max(self._settings.context_window_messages * 3, 200),
        )

        deb = float(self._settings.analysis_debounce_seconds or 0.0)
        if deb <= 0:
            await self._run_analysis(chat_id)
            return

        async with self._lock:
            prev = self._debounce_tasks.pop(chat_id, None)
            if prev is not None and not prev.done():
                prev.cancel()

            async def _delayed() -> None:
                try:
                    await asyncio.sleep(deb)
                    await self._run_analysis(chat_id)
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("Отложенный анализ chat_id=%s", chat_id)

            self._debounce_tasks[chat_id] = asyncio.create_task(_delayed())

    async def _run_analysis(self, chat_id: int) -> None:
        lines = self._storage.recent_context(chat_id, self._settings.context_window_messages)
        if not lines:
            return
        title = self._label_for_chat(chat_id)
        transcript = format_transcript(lines)
        result = await analyze_transcript(
            self._llm,
            chat_label=f"{title} ({chat_id})",
            transcript=transcript,
        )
        if not result.alert:
            return
        last_ts = lines[-1].ts if lines else None
        await self._notifier.dispatch(
            chat_id=chat_id,
            chat_title=title,
            message_ts=last_ts,
            result=result,
        )

    async def analyze_lines(self, chat_id: int, lines: list[StoredLine]) -> AnalysisResult:
        await self._ensure_chat(chat_id)
        title = self._label_for_chat(chat_id)
        transcript = format_transcript(lines)
        return await analyze_transcript(
            self._llm,
            chat_label=f"{title} ({chat_id})",
            transcript=transcript,
        )
