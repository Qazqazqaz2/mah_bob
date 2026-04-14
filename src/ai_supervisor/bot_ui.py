"""
Кнопочное управление в личке с ботом (inline keyboard + callback).
Настройки алертов / мониторинга — в SQLite, не в .env.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Union

from aiomax.buttons import CallbackButton, LinkButton

from ai_supervisor.config import Settings
from ai_supervisor.storage import SupervisorStorage

if TYPE_CHECKING:
    from aiomax.types import Callback

logger = logging.getLogger(__name__)

def _is_dialog_suspended_error(e: Exception) -> bool:
    s = repr(e)
    return "error.dialog.suspended" in s or "dialog.suspended" in s

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

TEXT_LLM = (
    "**Модель для анализа**\n\n"
    "Какой ИИ разбирает переписку: **YandexGPT** или **GigaChat**. "
    "Ключи и креды задаются в `.env` на сервере (`YANDEX_*` или `GIGACHAT_CREDENTIALS`). "
    "Переключение применяется к следующим проверкам чатов."
)

TEXT_FREQ = (
    "**Частота анализа чатов**\n\n"
    "**Пауза перед проверкой** — после последнего сообщения в чате бот ждёт N секунд и один раз "
    "запускает модель (реже дергает API при быстрой переписке).\n\n"
    "**Мин. интервал** — не чаще одного вызова модели по **одному** чату раз в N секунд. "
    "Если сообщения приходят раньше, проверка откладывается до конца интервала "
    "(повторно один и тот же хвост без новых сообщений не анализируется).\n\n"
    "**Макс. ожидание** — если переписка **не замолкает** и пауза постоянно сбрасывается, "
    "бот всё равно запустит анализ не позже чем через N секунд после начала активности.\n\n"
    "_Значения хранятся в SQLite; при первом создании БД подставляются из `.env`._"
)

_FREQ_DEBOUNCE_OPTIONS = (0.0, 15.0, 30.0, 60.0, 120.0)
_FREQ_INTERVAL_OPTIONS = (0.0, 60.0, 120.0, 300.0, 600.0)
_FREQ_MAXWAIT_OPTIONS = (0.0, 30.0, 60.0, 120.0, 300.0)


def _can_configure(user_id: int, settings: Settings) -> bool:
    admins = settings.bot_admin_id_set()
    if admins is None:
        return True
    return user_id in admins


def _deny_kb() -> list[list[CallbackButton]]:
    return [[CallbackButton("◀ Главное меню", CB_MAIN)]]


def keyboard_main() -> List[List[Union[CallbackButton, LinkButton]]]:
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
        [CallbackButton("Модель ИИ (Yandex / GigaChat)", "cfg:page:llm")],
        [CallbackButton("Частота анализа чатов", "cfg:page:freq")],
        _row_back_main(),
    ]


def keyboard_freq_pick(storage: SupervisorStorage, settings: Settings) -> list[list[CallbackButton]]:
    deb = storage.get_analysis_debounce_seconds(settings.analysis_debounce_seconds)
    interval = storage.get_analysis_min_interval_seconds(
        settings.analysis_min_interval_seconds
    )
    maxw = storage.get_analysis_max_wait_seconds(settings.analysis_max_wait_seconds)
    rows: list[list[CallbackButton]] = []
    for sec in _FREQ_DEBOUNCE_OPTIONS:
        mark = "✓ " if abs(deb - sec) < 0.01 else ""
        label = f"{mark}Пауза: {int(sec)} с" if sec > 0 else f"{mark}Пауза: сразу (0)"
        rows.append([CallbackButton(label[:40], f"cfg:freq:deb:{int(sec)}")])
    for sec in _FREQ_INTERVAL_OPTIONS:
        mark = "✓ " if abs(interval - sec) < 0.01 else ""
        if sec <= 0:
            label = f"{mark}Интервал: выкл"
        else:
            label = f"{mark}Интервал: {int(sec)} с"
        rows.append([CallbackButton(label[:40], f"cfg:freq:int:{int(sec)}")])
    for sec in _FREQ_MAXWAIT_OPTIONS:
        mark = "✓ " if abs(maxw - sec) < 0.01 else ""
        if sec <= 0:
            label = f"{mark}Макс. ожидание: выкл"
        else:
            label = f"{mark}Макс. ожидание: {int(sec)} с"
        rows.append([CallbackButton(label[:40], f"cfg:freq:max:{int(sec)}")])
    rows.append([CallbackButton("◀ Настройки", CB_SETTINGS)])
    return rows


def keyboard_llm_pick(storage: SupervisorStorage, settings: Settings) -> list[list[CallbackButton]]:
    cur = storage.get_llm_provider(settings.llm_provider)
    return [
        [
            CallbackButton(
                "✓ YandexGPT" if cur == "yandex" else "YandexGPT",
                "cfg:llm:yandex",
            ),
        ],
        [
            CallbackButton(
                "✓ GigaChat" if cur == "gigachat" else "GigaChat",
                "cfg:llm:gigachat",
            ),
        ],
        [CallbackButton("◀ Настройки", CB_SETTINGS)],
    ]


def _btn_label_chat(cid: int, title: Optional[str]) -> str:
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
        await _handle_cfg(callback, pl=pl, storage=storage, settings=settings)
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
    except Exception as e:
        if _is_dialog_suspended_error(e):
            storage.mark_dialog_suspended(uid, True)
            logger.warning(
                "ЛС недоступны (dialog.suspended), user_id=%s — отключаю UI-ответы",
                uid,
            )
            return
        logger.exception("ui callback %s: не удалось отправить ответ", pl)


async def _handle_cfg(
    callback: Callback,
    *,
    pl: str,
    storage: SupervisorStorage,
    settings: Settings,
) -> None:
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
        if pl == "cfg:page:llm":
            cur = storage.get_llm_provider(settings.llm_provider)
            hint = f"\n\n_Сейчас выбрано:_ **`{cur}`**"
            await callback.send(
                text=TEXT_LLM + hint,
                format="markdown",
                keyboard=keyboard_llm_pick(storage, settings),
            )
            return
        if pl == "cfg:page:freq":
            d = storage.get_analysis_debounce_seconds(settings.analysis_debounce_seconds)
            i = storage.get_analysis_min_interval_seconds(
                settings.analysis_min_interval_seconds
            )
            m = storage.get_analysis_max_wait_seconds(settings.analysis_max_wait_seconds)
            hint = f"\n\n_Сейчас:_ пауза **{int(d)}** с, интервал **{int(i)}** с, макс. ожидание **{int(m)}** с"
            await callback.send(
                text=TEXT_FREQ + hint,
                format="markdown",
                keyboard=keyboard_freq_pick(storage, settings),
            )
            return
        if pl.startswith("cfg:freq:deb:"):
            sec = float(pl.removeprefix("cfg:freq:deb:"))
            storage.set_analysis_debounce_seconds(sec)
            await callback.send(
                text=f"Пауза перед анализом: **{int(sec)}** с",
                format="markdown",
                keyboard=keyboard_freq_pick(storage, settings),
            )
            return
        if pl.startswith("cfg:freq:int:"):
            sec = float(pl.removeprefix("cfg:freq:int:"))
            storage.set_analysis_min_interval_seconds(sec)
            await callback.send(
                text=f"Мин. интервал между вызовами модели: **{int(sec)}** с",
                format="markdown",
                keyboard=keyboard_freq_pick(storage, settings),
            )
            return
        if pl.startswith("cfg:freq:max:"):
            sec = float(pl.removeprefix("cfg:freq:max:"))
            storage.set_analysis_max_wait_seconds(sec)
            await callback.send(
                text=f"Макс. ожидание до анализа при активности: **{int(sec)}** с",
                format="markdown",
                keyboard=keyboard_freq_pick(storage, settings),
            )
            return
        if pl == "cfg:llm:yandex":
            storage.set_llm_provider("yandex")
            await callback.send(
                text="Модель анализа: **YandexGPT**",
                format="markdown",
                keyboard=keyboard_llm_pick(storage, settings),
            )
            return
        if pl == "cfg:llm:gigachat":
            storage.set_llm_provider("gigachat")
            await callback.send(
                text="Модель анализа: **GigaChat**",
                format="markdown",
                keyboard=keyboard_llm_pick(storage, settings),
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
        f"• **LLM (SQLite):** `{storage.get_llm_provider(settings.llm_provider)}` "
        f"_(fallback .env: `{settings.llm_provider}`)_\n"
        f"• **Пауза перед анализом:** `{int(storage.get_analysis_debounce_seconds(settings.analysis_debounce_seconds))}` с "
        f"_(.env `ANALYSIS_DEBOUNCE_SECONDS`)_\n"
        f"• **Мин. интервал LLM по чату:** `{int(storage.get_analysis_min_interval_seconds(settings.analysis_min_interval_seconds))}` с "
        f"_(.env `ANALYSIS_MIN_INTERVAL_SECONDS`)_\n"
        f"• **База:** `{settings.sqlite_path}`"
    )
