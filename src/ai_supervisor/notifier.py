from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from aiomax import Bot

from ai_supervisor.analysis import AnalysisResult, ToxicityBreakdown
from ai_supervisor.storage import SupervisorStorage

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_IN_SIGNATURE = 5


def _evidence_signature(evidence: list[str]) -> frozenset[str]:
    """Нормализованные строки для сравнения дублей (регистр, пробелы)."""
    out: list[str] = []
    for e in (evidence or [])[:_MAX_EVIDENCE_IN_SIGNATURE]:
        s = e.strip()
        if not s:
            continue
        out.append(" ".join(s.split()).casefold())
    return frozenset(out)


def _evidence_redundant(
    new_sig: frozenset[str],
    recent_sigs: list[frozenset[str]],
) -> bool:
    """Пропуск, если набор цитат уже был или целиком входит в недавний алерт."""
    if not new_sig:
        return False
    for old in recent_sigs:
        if new_sig == old or new_sig <= old:
            return True
    return False


def _format_toxicity_block(tox: ToxicityBreakdown) -> str:
    labels = [
        ("Оскорбления", tox.insult),
        ("Угрозы", tox.threat),
        ("Мат / нецензурная лексика", tox.profanity),
        ("Давление / травля", tox.harassment),
        ("Ненависть / дискриминация", tox.hate_speech),
    ]
    lines = [f"• {name}: {round(score * 100)}%" for name, score in labels]
    ov = tox.overall or "none"
    lines.append(f"• **Сводно:** {ov}")
    if tox.notes and tox.notes.strip():
        lines.append(f"• _{tox.notes.strip()}_")
    return "\n".join(lines)


def _msk_ts(ts: Optional[int]) -> str:
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
    message_ts: Optional[int],
    result: AnalysisResult,
    toxicity_model_label: str = "ИИ",
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
    if result.toxicity is not None:
        lines.extend(
            [
                "",
                f"**Токсичность ({toxicity_model_label}):**",
                _format_toxicity_block(result.toxicity),
            ]
        )
    return "\n".join(lines)


class NotificationDispatcher:
    def __init__(self, storage: SupervisorStorage, bot: Bot) -> None:
        self._storage = storage
        self._bot = bot
        # chat_id -> [(monotonic_ts, frozenset нормализованных цитат), ...]
        self._alert_evidence_history: dict[int, list[tuple[float, frozenset[str]]]] = {}
        self._fallback_dedupe: dict[
            tuple[int, str, str, tuple[str, ...]], float
        ] = {}
        self._dedupe_ttl_sec = 900.0

    def _should_send_alert(
        self, *, chat_id: int, result: AnalysisResult
    ) -> bool:
        now = time.monotonic()
        ttl = self._dedupe_ttl_sec
        self._fallback_dedupe = {
            k: t for k, t in self._fallback_dedupe.items() if now - t < ttl * 2
        }
        hist = self._alert_evidence_history.setdefault(chat_id, [])
        hist[:] = [(t, s) for t, s in hist if now - t < ttl * 2]
        recent_sigs = [s for t, s in hist if now - t < ttl]

        sig = _evidence_signature(result.evidence or [])
        if sig and _evidence_redundant(sig, recent_sigs):
            logger.debug(
                "Алерт пропущен: те же или подмножество уже отправленных цитат (chat_id=%s)",
                chat_id,
            )
            return False

        if sig:
            hist.append((now, sig))
            if len(hist) > 32:
                hist[:] = hist[-24:]
            return True

        # Нет нормализуемых цитат (например только метки) — дедуп по полному кортежу
        ev_tuple = tuple((result.evidence or [])[:5])
        fk = (chat_id, result.category, result.title, ev_tuple)
        prev = self._fallback_dedupe.get(fk)
        if prev is not None and now - prev < ttl:
            logger.debug("Повторный алерт (fallback) пропущен chat_id=%s", chat_id)
            return False
        self._fallback_dedupe[fk] = now
        return True

    async def dispatch(self, *, chat_id: int, chat_title: str, message_ts: Optional[int], result: AnalysisResult) -> None:
        if not self._should_send_alert(chat_id=chat_id, result=result):
            return
        prov = self._storage.get_llm_provider("yandex")
        tox_label = "GigaChat" if prov == "gigachat" else "YandexGPT"
        text = build_alert_text(
            chat_id=chat_id,
            chat_title=chat_title,
            message_ts=message_ts,
            result=result,
            toxicity_model_label=tox_label,
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
