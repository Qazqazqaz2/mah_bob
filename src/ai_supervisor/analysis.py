from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ai_supervisor.conflict_heuristic import (
    llm_hint_for_forced_context,
    llm_hint_for_soft_context,
    transcript_conflict_message_evidence,
    transcript_conflict_signals,
    transcript_tension_message_evidence,
    transcript_tension_signals,
)
from ai_supervisor.llm_base import LLMClient, strip_code_fence
from ai_supervisor.storage import StoredLine

logger = logging.getLogger(__name__)

_last_llm_unavailable_warn_mono: float = 0.0
_LLM_UNAVAILABLE_WARN_INTERVAL_SEC = 120.0

SYSTEM_PROMPT = """Ты — аналитик рабочих чатов организаторов мероприятий.
Твоя задача — по фрагменту переписки определить, есть ли ПРОБЛЕМНАЯ ситуация для руководства.

## ПОЛНЫЙ СПИСОК ПРОБЛЕМНЫХ СИТУАЦИЙ (alert=true):

### 1. КОНФЛИКТЫ (все степени — от назревающего до открытого)
**КРИТИЧЕСКИЕ маркеры (severity=high, category=conflict):**
- Мат, оскорбления, угрозы: «иди нахуй», «пошёл ты», «дебил», «кретин», «сука»
- Открытые обвинения: «ты виноват», «из-за тебя всё сорвалось», «ты саботируешь»
- Угрозы: «уволюсь», «никому не скажу», «разнесу по всем чатам», «позвоню руководству»
- Переход на личности: «ты всегда так», «вот твоя типичная тупость»

**НАЗРЕВАЮЩИЙ конфликт (severity=medium/low, category=conflict):**
- Пассивная агрессия: «ну конечно», «кто бы сомневался», «опять ты», кавычки вокруг слов («умный», «важный»)
- Сарказм/троллинг: «ахахах/лол/😂😂😂», «гениально», «браво!», язвительные эмодзи 😏😒🙄
- Эмоциональные маркеры: МНОЖЕСТВО !!!!???, капс ВСЁ В ВЕРХНЕМ РЕГИСТРЕ
- Намёки на провал: «если опять не сделаешь», «как обычно всё на мне»

### 2. ФОРС-МАЖОР (severity=high, category=force_majeure)
**Слова-маркеры:**
- «СРОЧНО!», «КАТАСТРОФА», «ВСЁ ПРОПАЛО», «СОРВАНО»
- Проблемы: «поставщик подвёл», «техника сломалась», «гости не едут», «денег нет»
- Угроза сроков: «не успеваем», «deadline горит», «завтра сдача а ничего нет»

### 3. ИГНОРИРОВАНИЕ ЗАДАЧ (severity=medium/high, category=ignored_tasks)
- 3+ напоминания без реакции: «@Имя ещё раз напоминаю», «второй раз пишу»
- Отсутствие ответов на критические сообщения (видно по контексту)
- Саботаж: «не буду делать», «сам делай», «не моя зона ответственности»

### 4. ПОТЕРЯ КОНТРОЛЯ (severity=high, category=control_loss)
**Маркеры хаоса:**
- Противоречивые указания: «сначала делай А, потом НЕ ДЕЛАЙ А»
- Множество отмен/пересмотров: «отменяю предыдущее», «всё меняется»
- Отказ от ответственности: «я не знаю», «решайте сами», «не моя проблема»
- Хаотичная коммуникация: 10+ сообщений подряд без структуры, флуд

### 5. ТОКСИЧНОСТЬ (отдельно оценивается в toxicity)
- Давление/травля: «ты обязан», «докажи что не верблюд», групповое насилие
- Дискриминация: по полу, возрасту, национальности

## НЕ ПРОБЛЕМА (alert=false):
- Обычные вопросы/уточнения
- Вежливые споры: «а можно иначе?», «не согласен, предлагаю»
- Юмор без сарказма
- Технические обсуждения

## ФОРМАТ ОТВЕТА (строго JSON):
{
  "alert": true/false,
  "severity": "low|medium|high|none", 
  "category": "conflict|force_majeure|ignored_tasks|control_loss|toxicity|none",
  "title": "КРАТКИЙ заголовок (макс 10 слов)",
  "summary": "2-5 предложений: ЧТО произошло + ПОЧЕМУ важно для руководства",
  "who": "имена через запятую: Иван, Маша (никнеймы как в чате)",
  "evidence": ["дословная цитата 1", "цитата 2", "перефраз"],
  "toxicity": {
    "insult": 0.0-1.0,
    "threat": 0.0-1.0, 
    "profanity": 0.0-1.0,
    "harassment": 0.0-1.0,
    "hate_speech": 0.0-1.0,
    "overall": "none|low|medium|high",
    "notes": "где именно токсичность в последних 5-10 репликах"
  }
}

**ПРАВИЛА:**
- В evidence — дословные цитаты с именами: «Иван: иди нахуй»
- В who — только активные участники проблемы
- Если несколько проблем — выбери главную (conflict приоритетнее)
- Toxicity оцени только ПОСЛЕДНИЕ 5-10 реплик чата
- При alert=false заполни поля нейтрально"""


class ToxicityBreakdown(BaseModel):
    model_config = ConfigDict(extra="ignore")

    insult: float = Field(0.0, ge=0.0, le=1.0)
    threat: float = Field(0.0, ge=0.0, le=1.0)
    profanity: float = Field(0.0, ge=0.0, le=1.0)
    harassment: float = Field(0.0, ge=0.0, le=1.0)
    hate_speech: float = Field(0.0, ge=0.0, le=1.0)
    overall: str = "none"
    notes: str = ""

    @field_validator(
        "insult", "threat", "profanity", "harassment", "hate_speech", mode="before"
    )
    @classmethod
    def _clamp_score(cls, v: object) -> float:
        try:
            x = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        if x > 1.0 and x <= 100.0:
            x = x / 100.0
        return max(0.0, min(1.0, x))


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    alert: bool
    severity: str = "low"
    category: str = "none"
    title: str = ""
    summary: str = ""
    who: str = ""
    evidence: list[str] = Field(default_factory=list)
    toxicity: Optional["ToxicityBreakdown"] = None

    @field_validator("who", mode="before")
    @classmethod
    def _coerce_who(cls, v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
            return ", ".join(parts)
        return str(v).strip()

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        if isinstance(v, list):
            out: list[str] = []
            for x in v:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    out.append(s)
            return out
        s = str(v).strip()
        return [s] if s else []


def format_transcript(lines: list[StoredLine]) -> str:
    out: list[str] = []
    for ln in lines:
        ts = ln.ts
        name = ln.sender_name
        out.append(f"[{ts}] {name}: {ln.text}")
    return "\n".join(out)


def parse_json_loose(raw: str) -> dict[str, Any]:
    t = strip_code_fence(raw)
    # иногда модель добавляет текст до JSON
    m = re.search(r"\{[\s\S]*\}\s*$", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def parse_analysis_result(data: dict[str, Any]) -> AnalysisResult:
    """Разбор ответа модели: основной JSON + опциональный блок toxicity."""
    d = dict(data)
    tox_raw = d.pop("toxicity", None)
    d["toxicity"] = None
    result = AnalysisResult.model_validate(d)
    if isinstance(tox_raw, dict):
        try:
            return result.model_copy(
                update={"toxicity": ToxicityBreakdown.model_validate(tox_raw)}
            )
        except ValidationError:
            logger.debug("Блок toxicity в ответе LLM не разобран", exc_info=True)
    return result


async def analyze_transcript(
    llm: LLMClient,
    *,
    chat_label: str,
    transcript: str,
) -> AnalysisResult:
    global _last_llm_unavailable_warn_mono
    forced, signal_labels = transcript_conflict_signals(transcript)
    soft_labels = transcript_tension_signals(transcript)
    hint = llm_hint_for_forced_context(signal_labels) + llm_hint_for_soft_context(soft_labels)
    user = f"Чат: {chat_label}\n\nФрагмент переписки:\n{transcript}{hint}"
    llm_failed = False
    try:
        raw = await llm.complete(system=SYSTEM_PROMPT, user=user)
    except Exception as e:
        llm_failed = True
        now = time.monotonic()
        if now - _last_llm_unavailable_warn_mono >= _LLM_UNAVAILABLE_WARN_INTERVAL_SEC:
            _last_llm_unavailable_warn_mono = now
            logger.warning(
                "LLM недоступен (%s): %s. Дальше — только эвристики (при срабатывании будет алерт).",
                type(e).__name__,
                e,
            )
        else:
            logger.debug("LLM снова недоступен (%s)", type(e).__name__)
        result = AnalysisResult(
            alert=False,
            category="none",
            title="LLM недоступен",
            summary=(
                "Вызов модели не удался (403 — нет прав на каталог в Yandex Cloud, неверный ключ, сеть). "
                "Назначьте сервисному аккаунту/API-ключу роль вроде «ai.languageModels.user» на папку "
                "из YANDEX_FOLDER_ID и проверьте modelUri. При явной эскалации в тексте сработает локальная эвристика."
            ),
            who="",
            evidence=[],
            toxicity=None,
        )
    else:
        try:
            data = parse_json_loose(raw)
            result = parse_analysis_result(data)
        except Exception:
            logger.exception("Не удалось разобрать JSON анализа, сырой ответ: %s", raw[:2000])
            result = AnalysisResult(
                alert=False,
                category="none",
                title="Ошибка разбора ответа модели",
                summary="Проверьте формат ответа LLM и логи.",
                who="",
                evidence=[],
                toxicity=None,
            )

    if forced and not result.alert:
        logger.info(
            "Эвристика конфликта сработала, модель не дала алерт — принудительный алерт: %s",
            signal_labels,
        )
        ev_msgs, who_hint = transcript_conflict_message_evidence(transcript)
        evidence = ev_msgs if ev_msgs else [f"({lbl})" for lbl in signal_labels[:5]]
        who = who_hint or "—"
        summary = (
            "Зафиксированы формулировки, типичные для конфликта или оскорблений в рабочем чате. "
            "Рекомендуется вмешательство менеджера."
        )
        if llm_failed:
            summary += (
                " Сейчас модель недоступна (403/сеть) — сработали только ключевые слова без разбора контекста; "
                "перепроверьте диалог глазами."
            )
        preserved_tox = result.toxicity
        return AnalysisResult(
            alert=True,
            severity="high",
            category="conflict",
            title="Эскалация / грубость в переписке",
            summary=summary,
            who=who,
            evidence=evidence,
            toxicity=preserved_tox,
        )

    if (not forced) and (not result.alert) and len(soft_labels) >= 2:
        ev_msgs, who_hint = transcript_tension_message_evidence(transcript)
        if ev_msgs:
            return AnalysisResult(
                alert=True,
                severity="low",
                category="conflict",
                title="Пассивная агрессия / троллинг",
                summary=(
                    "В переписке заметны сарказм и обесценивающие формулировки, которые могут "
                    "эскалировать в конфликт и ухудшить рабочую атмосферу."
                ),
                who=who_hint or "—",
                evidence=ev_msgs,
                toxicity=result.toxicity,
            )

    if forced and result.alert and result.severity == "low":
        return result.model_copy(update={"severity": "medium"})

    return result
