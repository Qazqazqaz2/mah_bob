from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Set, Tuple

from aiomax import Bot
from aiomax.types import Message as MaxMessage

from ai_supervisor import bot_ui
from ai_supervisor.analysis import AnalysisResult, analyze_transcript, format_transcript
from ai_supervisor.config import Settings
from ai_supervisor.llm_base import LLMClient
from ai_supervisor.llm_factory import build_llm
from ai_supervisor.notifier import NotificationDispatcher
from ai_supervisor.storage import StoredLine, SupervisorStorage

logger = logging.getLogger(__name__)

def _is_dialog_suspended_error(e: Exception) -> bool:
    s = repr(e)
    return "error.dialog.suspended" in s or "dialog.suspended" in s


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
        llm: Optional[LLMClient],
        bot: Bot,
        notifier: NotificationDispatcher,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._llm_override = llm
        self._llm_cache: Optional[Tuple[str, LLMClient]] = None
        self._bot = bot
        self._notifier = notifier
        self._chat_cache: Dict[int, Dict[str, Any]] = {}
        self._debounce_tasks: dict[int, asyncio.Task[None]] = {}
        self._interval_wait_tasks: dict[int, asyncio.Task[None]] = {}
        self._max_wait_tasks: dict[int, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._chat_analysis_locks: Dict[int, asyncio.Lock] = {}
        self._dm_hint_shown: Set[int] = set()
        self._dm_menu_throttle: Dict[int, float] = {}

    def _active_llm(self) -> LLMClient:
        if self._llm_override is not None:
            return self._llm_override
        want = self._storage.get_llm_provider(self._settings.llm_provider)
        if self._llm_cache is None or self._llm_cache[0] != want:
            logger.info("LLM: провайдер %s", want)
            self._llm_cache = (want, build_llm(self._settings, want))
        return self._llm_cache[1]

    def _analysis_debounce(self) -> float:
        return self._storage.get_analysis_debounce_seconds(
            self._settings.analysis_debounce_seconds
        )

    def _analysis_max_wait(self) -> float:
        return self._storage.get_analysis_max_wait_seconds(
            self._settings.analysis_max_wait_seconds
        )

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
        if self._storage.is_dialog_suspended(user_id):
            return
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
        except Exception as e:
            if _is_dialog_suspended_error(e):
                self._storage.mark_dialog_suspended(user_id, True)
                logger.warning(
                    "ЛС недоступны (dialog.suspended), user_id=%s — отключаю меню",
                    user_id,
                )
                return
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
                try:
                    raise
                except Exception as e:
                    if uid is not None and _is_dialog_suspended_error(e):
                        self._storage.mark_dialog_suspended(uid, True)
                        logger.warning(
                            "ЛС недоступны (dialog.suspended), user_id=%s — отключаю ответы",
                            uid,
                        )
                        return
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

    async def _ensure_chat(self, chat_id: int) -> Dict[str, Any]:
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

        max_wait = float(self._analysis_max_wait() or 0.0)
        if max_wait > 0:
            async with self._lock:
                # Планируем "принудительный" анализ не реже чем раз в max_wait,
                # даже если debounce постоянно отменяется из-за потока сообщений.
                mw = self._max_wait_tasks.get(chat_id)
                if mw is None or mw.done():
                    async def _force_after() -> None:
                        try:
                            await asyncio.sleep(max_wait)
                            await self._run_analysis(chat_id)
                        except asyncio.CancelledError:
                            return
                        except Exception:
                            logger.exception("Отложенный анализ (max_wait) chat_id=%s", chat_id)

                    self._max_wait_tasks[chat_id] = asyncio.create_task(_force_after())

        deb = float(self._analysis_debounce() or 0.0)
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

    async def _chat_analysis_lock(self, chat_id: int) -> asyncio.Lock:
        async with self._lock:
            lk = self._chat_analysis_locks.get(chat_id)
            if lk is None:
                lk = asyncio.Lock()
                self._chat_analysis_locks[chat_id] = lk
            return lk

    async def _run_analysis(
        self, chat_id: int, *, bypass_interval_gate: bool = False
    ) -> None:
        # Если анализ всё же запускается — max_wait можно сбросить,
        # иначе он будет лишний раз дергать анализ после уже выполненного прогона.
        async with self._lock:
            t = self._max_wait_tasks.pop(chat_id, None)
            if t is not None and not t.done():
                t.cancel()

        lk = await self._chat_analysis_lock(chat_id)

        async with lk:
            lines = self._storage.recent_context(
                chat_id, self._settings.context_window_messages
            )
            if not lines:
                return
            newest_ts = lines[-1].ts
            hwm = self._storage.get_chat_analyze_hwm(chat_id)
            if hwm is not None and newest_ts <= hwm:
                logger.debug(
                    "Пропуск анализа: тот же хвост переписки chat_id=%s newest=%s hwm=%s",
                    chat_id,
                    newest_ts,
                    hwm,
                )
                return

            min_int = self._storage.get_analysis_min_interval_seconds(
                self._settings.analysis_min_interval_seconds
            )
            delay_for: Optional[float] = None
            if not bypass_interval_gate and min_int > 0:
                last_wall = self._storage.get_chat_last_analysis_wall(chat_id)
                now = time.time()
                if last_wall is not None and (now - last_wall) < min_int:
                    delay_for = min_int - (now - last_wall)

        if delay_for is not None and delay_for > 0:
            async with self._lock:
                prev = self._interval_wait_tasks.pop(chat_id, None)
                if prev is not None and not prev.done():
                    prev.cancel()

                async def _after_interval() -> None:
                    try:
                        await asyncio.sleep(delay_for)
                        await self._run_analysis(
                            chat_id, bypass_interval_gate=True
                        )
                    except asyncio.CancelledError:
                        return
                    except Exception:
                        logger.exception(
                            "Отложенный анализ (интервал) chat_id=%s", chat_id
                        )

                self._interval_wait_tasks[chat_id] = asyncio.create_task(
                    _after_interval()
                )
            return

        async with lk:
            lines = self._storage.recent_context(
                chat_id, self._settings.context_window_messages
            )
            if not lines:
                return
            newest_ts = lines[-1].ts
            hwm = self._storage.get_chat_analyze_hwm(chat_id)
            if hwm is not None and newest_ts <= hwm:
                return

            title = self._label_for_chat(chat_id)
            transcript = format_transcript(lines)
            result = await analyze_transcript(
                self._active_llm(),
                chat_label=f"{title} ({chat_id})",
                transcript=transcript,
            )
            self._storage.set_chat_analyze_hwm(chat_id, newest_ts)
            self._storage.set_chat_last_analysis_wall(chat_id, time.time())
            if not result.alert:
                return
            last_ts = lines[-1].ts if lines else None
            max_age = float(self._settings.alert_max_age_seconds or 0.0)
            if max_age > 0 and last_ts is not None:
                now_ms = int(time.time() * 1000)
                if now_ms - int(last_ts) > int(max_age * 1000):
                    logger.info(
                        "Алерт не отправлен: контекст слишком старый (age_sec=%s > max_age_sec=%s) chat_id=%s",
                        round((now_ms - int(last_ts)) / 1000.0, 1),
                        max_age,
                        chat_id,
                    )
                    return
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
            self._active_llm(),
            chat_label=f"{title} ({chat_id})",
            transcript=transcript,
        )
