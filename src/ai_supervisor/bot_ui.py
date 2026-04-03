"""
Кнопочное управление в личке с ботом (inline keyboard + callback).
Настройки алертов / мониторинга — в SQLite, не в .env.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiomax.buttons import CallbackButton, LinkButton

from ai_supervisor.storage import SupervisorStorage

if TYPE_CHECKING:
    from aiomax.types import Callback

    from ai_supervisor.config import Settings

logger = logging.getLogger(__name__)

CB_MAIN = "ui:main"
CB_ABOUT = "ui:about"
CB_SETUP = "ui:setup"
CB_STATUS = "ui:status"
CB_ALERTS = "ui:alerts"
CB_SETTINGS = "ui:settings"

TEXT_MAIN = (
    "**AI Supervisor**\n\n"
    "Мониторинг **групповых** чатов MAX и алерты менеджерам.\n\n"
    "Выберите раздел кнопкой ниже — команды вводить не нужно."
)

TEXT_ABOUT = (
    "**Что делает бот**\n\n"
    "• Читает новые сообщения в рабочих **группах**, куда его добавили.\n"
    "• Держит контекст переписки и отправляет фрагмент в **YandexGPT / GigaChat**.\n"
    "• Если модель видит риск (конфликт, срыв, игнор задач и т.д.) — приходит "
    "**сводка** в чат менеджеров и/или в личку дежурным.\n\n"
    "Личные переписки **не анализируются**."
)

TEXT_SETUP = (
    "**Как подключить**\n\n"
    "1. Добавьте бота в **групповой** рабочий чат.\n"
    "2. Назначьте бота **администратором** (иначе события группы часто не приходят).\n"
    "3. Откройте бота в ЛС → **Настройки бота**: задайте чат для алертов, дежурных и "
    "список мониторинга (или «все группы»).\n"
    "4. При первом запуске пустая БД может один раз подтянуть старые значения из `.env` "
    "(импорт), дальше всё в SQLite.\n\n"
    "_Токен MAX и оплата LLM — в `.env` на сервере._"
)

TEXT_ALERTS = (
    "**Куда уходят алерты**\n\n"
    "Задаётся в **Настройки бота** (кнопки в главном меню):\n\n"
    "• **Чат для алертов** — общий чат менеджеров.\n"
    "• **Дежурные** — кому дублировать в личку.\n\n"
    "Пока ничего не выбрано — алерты только в лог."
)

TEXT_SETTINGS_HUB = (
    "**Настройки** (хранятся в SQLite на сервере)\n\n"
    "Выберите раздел. Список «недавних групп» появляется после того, как "
    "в этих чатах было сообщение, пока бот был в них."
)


def _can_configure(user_id: int, settings: Settings) -> bool:
    admins = settings.bot_admin_id_set()
    if admins is None:
        return True
    return user_id in admins


def _deny_kb() -> list[list[CallbackButton]]:
    return [[CallbackButton("◀ Главное меню", CB_MAIN)]]


def keyboard_main() -> list[list[CallbackButton | LinkButton]]:
    return [
        [
            CallbackButton("Настройки бота", CB_SETTINGS),
        ],
        [
            CallbackButton("Что делает бот", CB_ABOUT),
            CallbackButton("Как подключить", CB_SETUP),
        ],
        [
            CallbackButton("Статус", CB_STATUS),
            CallbackButton("Куда алерты", CB_ALERTS),
        ],
        [
            LinkButton("Сайт API MAX", "https://dev.max.ru/docs-api"),
        ],
    ]


def _row_back_main() -> list[CallbackButton]:
    return [CallbackButton("◀ Главное меню", CB_MAIN)]


def keyboard_with_back() -> list[list[CallbackButton]]:
    return [_row_back_main()]


def keyboard_settings_hub() -> list[list[CallbackButton]]:
    return [
        [CallbackButton("Чат для алертов", "cfg:page:mgr")],
        [CallbackButton("Дежурные (личка)", "cfg:page:duty")],
        [CallbackButton("Какие чаты мониторить", "cfg:page:mon")],
        _row_back_main(),
    ]


def _btn_label_chat(cid: int, title: str | None) -> str:
    base = (title.strip() if title else "") or str(cid)
    if len(base) > 22:
        base = base[:20] + "…"
    return f"{base}"


def keyboard_pick_manager(storage: SupervisorStorage) -> list[list[CallbackButton]]:
    known = storage.list_known_chats(10)
    rows: list[list[CallbackButton]] = []
    for cid, title in known:
        rows.append(
            [CallbackButton(_btn_label_chat(cid, title), f"cfg:mgr:set:{cid}")]
        )
    if not rows:
        rows.append(
            [CallbackButton("Обновить список (после сообщений в группах)", "cfg:page:mgr")]
        )
    rows.append([CallbackButton("Сбросить чат алертов", "cfg:mgr:clr")])
    rows.append([CallbackButton("◀ Настройки", CB_SETTINGS)])
    return rows


def keyboard_duty_menu() -> list[list[CallbackButton]]:
    return [
        [CallbackButton("Я дежурный — добавить меня", "cfg:duty:add")],
        [CallbackButton("Убрать себя из дежурных", "cfg:duty:rm")],
        [CallbackButton("Кто в списке", "cfg:duty:list")],
        [CallbackButton("◀ Настройки", CB_SETTINGS)],
    ]


def keyboard_mon_menu(storage: SupervisorStorage) -> list[list[CallbackButton]]:
    mode = storage.get_monitor_mode()
    rows: list[list[CallbackButton]] = [
        [
            CallbackButton(
                "✓ Все группы" if mode == "all" else "Все группы (куда добавлен бот)",
                "cfg:mon:all",
            ),
        ],
        [CallbackButton("Только выбранные чаты…", "cfg:mon:pick")],
        [CallbackButton("◀ Настройки", CB_SETTINGS)],
    ]
    return rows


def keyboard_mon_pick(storage: SupervisorStorage) -> list[list[CallbackButton]]:
    mode = storage.get_monitor_mode()
    selected = set(storage.list_monitored_chat_ids()) if mode == "list" else set()
    rows: list[list[CallbackButton]] = []
    for cid, title in storage.list_known_chats(10):
        mark = "✓ " if cid in selected else "+ "
        label = mark + _btn_label_chat(cid, title)
        rows.append([CallbackButton(label[:36], f"cfg:mon:tog:{cid}")])
    if not rows:
        rows.append(
            [CallbackButton("Обновить (нужна активность в группах)", "cfg:mon:pick")]
        )
    rows.append([CallbackButton("◀ К мониторингу", "cfg:page:mon")])
    rows.append([CallbackButton("◀ Настройки", CB_SETTINGS)])
    return rows


async def handle_ui_callback(
    callback: Callback,
    *,
    settings: Settings,
    storage: SupervisorStorage,
) -> None:
    pl = (callback.payload or "").strip()
    uid = callback.user.user_id

    if pl.startswith("cfg:"):
        if not _can_configure(uid, settings):
            try:
                await callback.send(
                    text="**Нет прав** менять настройки. Админ задаётся в `.env`: `BOT_ADMIN_USER_IDS`.",
                    format="markdown",
                    keyboard=_deny_kb(),
                )
            except Exception:
                logger.exception("cfg: отказ в доступе")
            return
        await _handle_cfg(callback, pl=pl, storage=storage)
        return

    if not pl.startswith("ui:"):
        return

    if pl == CB_MAIN:
        text, kb = TEXT_MAIN, keyboard_main()
    elif pl == CB_SETTINGS:
        text, kb = TEXT_SETTINGS_HUB, keyboard_settings_hub()
    elif pl == CB_ABOUT:
        text, kb = TEXT_ABOUT, keyboard_with_back()
    elif pl == CB_SETUP:
        text, kb = TEXT_SETUP, keyboard_with_back()
    elif pl == CB_STATUS:
        text, kb = _status_text(settings, storage), keyboard_with_back()
    elif pl == CB_ALERTS:
        text, kb = TEXT_ALERTS, keyboard_with_back()
    else:
        return

    try:
        await callback.send(text=text, format="markdown", keyboard=kb)
    except Exception:
        logger.exception("ui callback %s: не удалось отправить ответ", pl)


async def _handle_cfg(callback: Callback, *, pl: str, storage: SupervisorStorage) -> None:
    try:
        if pl == "cfg:page:mgr":
            mid = storage.get_manager_chat_id()
            hint = (
                ""
                if storage.list_known_chats(1)
                else "\n_Пока нет недавних групп — напишите любое сообщение в нужной группе (где есть бот), затем снова откройте этот экран._\n"
            )
            head = (
                f"**Чат для алертов**\n\nСейчас: `{mid if mid is not None else 'не задан'}`\n"
                f"{hint}\nВыберите чат:\n"
            )
            await callback.send(
                text=head,
                format="markdown",
                keyboard=keyboard_pick_manager(storage),
            )
            return
        if pl == "cfg:page:duty":
            await callback.send(
                text="**Дежурные**\n\nКто получает копию алерта в личку:",
                format="markdown",
                keyboard=keyboard_duty_menu(),
            )
            return
        if pl == "cfg:page:mon":
            await callback.send(
                text="**Мониторинг**\n\n• **Все группы** — анализ везде, где есть бот.\n"
                "• **Только выбранные** — отметьте чаты (после активности в них).",
                format="markdown",
                keyboard=keyboard_mon_menu(storage),
            )
            return
        if pl == "cfg:mon:pick":
            await callback.send(
                text="**Чаты в мониторинге**\n\n**+** — в список, **✓** — уже в списке (нажмите, чтобы убрать).",
                format="markdown",
                keyboard=keyboard_mon_pick(storage),
            )
            return
        if pl == "cfg:mon:all":
            storage.set_monitor_mode("all")
            await callback.send(
                text="Режим: **все группы**, куда добавлен бот.",
                format="markdown",
                keyboard=keyboard_mon_menu(storage),
            )
            return
        if pl.startswith("cfg:mon:tog:"):
            rest = pl.removeprefix("cfg:mon:tog:")
            cid = int(rest)
            storage.ensure_monitor_list_mode()
            if cid in storage.list_monitored_chat_ids():
                storage.remove_monitored_chat(cid)
            else:
                storage.add_monitored_chat(cid)
            await callback.send(
                text="Список обновлён.",
                format="markdown",
                keyboard=keyboard_mon_pick(storage),
            )
            return
        if pl.startswith("cfg:mgr:set:"):
            rest = pl.removeprefix("cfg:mgr:set:")
            cid = int(rest)
            storage.set_manager_chat_id(cid)
            await callback.send(
                text=f"Чат для алертов: **`{cid}`**",
                format="markdown",
                keyboard=keyboard_pick_manager(storage),
            )
            return
        if pl == "cfg:mgr:clr":
            storage.set_manager_chat_id(None)
            await callback.send(
                text="Чат алертов **сброшен**.",
                format="markdown",
                keyboard=keyboard_pick_manager(storage),
            )
            return
        if pl == "cfg:duty:add":
            uid_self = callback.user.user_id
            storage.add_duty_user(uid_self)
            await callback.send(
                text=f"Добавлен дежурный: `{uid_self}`",
                format="markdown",
                keyboard=keyboard_duty_menu(),
            )
            return
        if pl == "cfg:duty:rm":
            uid_self = callback.user.user_id
            storage.remove_duty_user(uid_self)
            await callback.send(
                text=f"Удалён из дежурных: `{uid_self}`",
                format="markdown",
                keyboard=keyboard_duty_menu(),
            )
            return
        if pl == "cfg:duty:list":
            lst = storage.list_duty_users()
            txt = "**Дежурные:**\n" + ("\n".join(f"• `{u}`" for u in lst) if lst else "_(пусто)_")
            await callback.send(
                text=txt,
                format="markdown",
                keyboard=keyboard_duty_menu(),
            )
            return
    except Exception:
        logger.exception("ошибка cfg %s", pl)


def _status_text(settings: Settings, storage: SupervisorStorage) -> str:
    snap = storage.export_state_snapshot()
    monitored = storage.monitored_chat_filter()
    mon = (
        "все группы, куда добавлен бот"
        if monitored is None
        else f"только {len(monitored)} чат(ов)"
    )
    mgr = storage.get_manager_chat_id()
    mgr_s = f"`{mgr}`" if mgr is not None else "не задан"
    nd = len(storage.list_duty_users())
    admins = settings.bot_admin_id_set()
    adm = "любой в ЛС" if admins is None else f"{len(admins)} id в BOT_ADMIN_USER_IDS"
    return (
        "**Статус**\n\n"
        f"• **Маркер** long polling: `{snap.get('max_marker')}`\n"
        f"• **Строк в истории** (SQLite): `{snap.get('messages_rows')}`\n"
        f"• **Чат алертов:** {mgr_s}\n"
        f"• **ЛС дежурным:** {nd} чел.\n"
        f"• **Мониторинг:** {mon}\n"
        f"• **Кто меняет настройки:** {adm}\n"
        f"• **LLM:** `{settings.llm_provider}`\n"
        f"• **База:** `{settings.sqlite_path}`"
    )
