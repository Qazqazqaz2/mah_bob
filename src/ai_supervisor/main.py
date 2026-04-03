from __future__ import annotations

import asyncio
import logging
import sys

from ai_supervisor.aiomax_bot import SupervisorBot
from ai_supervisor.config import Settings, get_settings
from ai_supervisor.llm_factory import build_llm
from ai_supervisor.notifier import NotificationDispatcher
from ai_supervisor.storage import SupervisorStorage
from ai_supervisor.bot_ui import handle_ui_callback, keyboard_main, TEXT_MAIN
from ai_supervisor.supervisor import ChatSupervisor

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _validate_notifications(storage: SupervisorStorage) -> None:
    if storage.get_manager_chat_id() is None and not storage.list_duty_users():
        logging.warning(
            "В SQLite не заданы чат алертов и дежурные — настройте в боте «Настройки» или задайте .env для первого импорта."
        )


async def run_aiomax() -> None:
    s = get_settings()
    _setup_logging(s.log_level)

    storage = SupervisorStorage(s.sqlite_path)
    storage.seed_runtime_from_env(s)
    _validate_notifications(storage)
    llm = build_llm(s)
    bot = SupervisorBot(
        storage,
        poll_timeout=s.long_poll_timeout,
        poll_limit=s.long_poll_limit,
        access_token=s.max_access_token,
        mention_prefix=True,
        default_format="markdown",
    )
    bot.restore_marker()

    notifier = NotificationDispatcher(storage, bot)
    sup = ChatSupervisor(
        settings=s,
        storage=storage,
        llm=llm,
        bot=bot,
        notifier=notifier,
    )

    @bot.on_bot_start()
    async def _on_bot_start(payload):  # noqa: ANN001
        """Начало диалога — сразу кнопочное меню."""
        try:
            await payload.send(
                text=TEXT_MAIN,
                format="markdown",
                keyboard=keyboard_main(),
            )
            sup.note_user_greeted_on_start(payload.user.user_id)
        except Exception:
            logger.exception("on_bot_start: не удалось отправить меню")

    @bot.on_ready()
    async def _on_ready() -> None:
        logger.info("MAX: polling активен (бот @%s, id=%s)", bot.username, bot.id)

    @bot.on_button_callback()
    async def _on_button_callback(callback):  # noqa: ANN001
        await handle_ui_callback(callback, settings=s, storage=storage)

    @bot.on_message(detect_commands=False)
    async def _on_message(message):  # noqa: ANN001
        await sup.handle_aiomax_message(message)

    await bot.start_polling()


def run_sync() -> None:
    try:
        asyncio.run(run_aiomax())
    except KeyboardInterrupt:
        logging.info("Остановка по Ctrl+C")
        sys.exit(0)


def main() -> None:
    run_sync()


if __name__ == "__main__":
    main()
