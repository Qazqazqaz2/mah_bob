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
    transcript_delay_message_evidence,
    transcript_delay_signals,
    transcript_tension_message_evidence,
    transcript_tension_signals,
)
from ai_supervisor.llm_base import LLMClient, strip_code_fence
from ai_supervisor.storage import StoredLine

logger = logging.getLogger(__name__)

_last_llm_unavailable_warn_mono: float = 0.0
_LLM_UNAVAILABLE_WARN_INTERVAL_SEC = 120.0

SYSTEM_PROMPT = """Ты — опытный аналитик рабочих чатов в сфере услуг (ивенты, свадьбы, агентства).
Твоя задача — определить, есть ли ПРОБЛЕМНАЯ ситуация для руководства, включая РАННИЕ СИГНАЛЫ недовольства клиента.

Ты анализируешь не только явные конфликты, но и:
- нарастающее недовольство
- потерю доверия
- риск расторжения договора
- скрытые или пассивные конфликты

Важно: люди могут писать по-разному — не опирайся только на конкретные слова, анализируй СМЫСЛ, ТОН и ДИНАМИКУ общения.

Правило для evidence:
- evidence должен содержать ТОЛЬКО фразы, которые напрямую подтверждают выбранную category.
- Не включай в evidence нерелевантные реплики из окна контекста (например бытовые вопросы), даже если они рядом в переписке.

---

## ЧТО СЧИТАЕТСЯ ПРОБЛЕМОЙ (alert=true)

### 1. КОНФЛИКТ / ЭСКАЛАЦИЯ (category=conflict)

**Критический конфликт (severity=high):**
- оскорбления, мат, агрессия
- угрозы (уволиться, оставить отзывы, пожаловаться)
- открытые обвинения: «вы не сделали», «это ваша вина»
- требования извинений, ультиматумы
- «я всем расскажу», «напишу отзывы»

**Средний уровень (severity=medium):**
- длительный спор без решения
- взаимные обвинения
- обесценивание компетенции: «вы не умеете», «у вас нет базы»
- давление: «и что вы скажете?», «почему так?»

**Низкий уровень / ранняя стадия (severity=low):**
- раздражение, недовольство
- повторяющиеся претензии
- фразы типа:
  - «вы не учли»
  - «меня не устраивает»
  - «очень странно»
  - «грустно»
  - «складывается впечатление»

---

### 2. РИСК ПОТЕРИ КЛИЕНТА / РАСТОРЖЕНИЕ (category=client_loss)

**Критический (severity=high):**
- «хочу расторгнуть договор»
- «верните деньги»
- «мы не будем работать»
- отказ от услуги / отмена мероприятия

**Средний (severity=medium):**
- обсуждение условий расторжения
- сомнения в ценности работы
- «я не вижу смысла продолжать»

**Ранние сигналы (severity=low):**
- «мы ожидали другого»
- «это не соответствует нашим запросам»
- сравнение с другими компаниями
- сомнения в результате

---

### 3. ПРЕТЕНЗИИ К КАЧЕСТВУ / ПОТЕРЯ ДОВЕРИЯ (category=quality_issues)

**Сильные маркеры:**
- длинные структурированные жалобы (списки проблем)
- обвинения в непрофессионализме
- «вы не придерживались плана»
- «вы не проверяли»
- «нам пришлось самим»

Это особенно важно, даже БЕЗ агрессии — это сигнал руководству.

---

### 4. ТОКСИЧНОСТЬ (category=toxicity)

- агрессивный тон
- давление
- манипуляции
- переход на личности
- эмоциональные всплески

---

### 5. РАННИЕ “ЗВОНОЧКИ” (очень важно)

Даже если нет конфликта, ставь alert=true при наличии:

- повторяющихся замечаний
- ощущения недовольства
- недоверия к экспертизе
- пассивной агрессии
- фраз:
  - «опять не то»
  - «мы это уже обсуждали»
  - «вы нас не услышали»
  - «почему так долго»
  - «нам это не подходит»

---

### 6. НАРУШЕНИЕ СРОКОВ / ОЖИДАНИЙ (category=quality_issues)

ВАЖНО: даже вежливые сообщения могут быть проблемой.

Считай alert=true (severity=low), если клиент:

- указывает на задержку:
  - «мы уже неделю ждем»
  - «до сих пор нет»
  - «обещали, но не прислали»
  - «еще в понедельник должны были»
  - «Скажите, пожалуйста, мы уже неделю ждем варианты площадок»
  - «Нам обещали прислать еще в понедельник»

- напоминает о договоренностях:
  - «вы говорили, что будет»
  - «нам обещали»
  - «ждали к дате»

- мягко выражает недовольство сроками:
  - «подскажите, пожалуйста»
  - «можете уточнить статус»
  - «когда ждать?»

Даже если тон вежливый → это РАННИЙ СИГНАЛ проблемы.

Это НЕ нейтральный запрос, если:
- есть ожидание во времени (ждем, обещали)
- есть факт задержки

Такие ситуации — начало потери доверия.

---

## НЕ ПРОБЛЕМА (alert=false)

- нейтральные обсуждения
- спокойные уточнения
- конструктив без эмоций
- единичные замечания без негатива

---

## ФОРМАТ ОТВЕТА (строго JSON):

{
  "alert": true/false,
  "severity": "low|medium|high|none",
  "category": "conflict|client_loss|quality_issues|toxicity|none",
  "title": "КРАТКИЙ заголовок (макс 10 слов)",
  "summary": "2-5 предложений: что происходит + стадия конфликта/недовольства + риск для бизнеса",
  "who": "имена через запятую",
  "evidence": ["Иван: цитата", "Мария: цитата", "краткий пересказ"],
  "toxicity": {
    "insult": 0.0-1.0,
    "threat": 0.0-1.0,
    "profanity": 0.0-1.0,
    "harassment": 0.0-1.0,
    "hate_speech": 0.0-1.0,
    "overall": "none|low|medium|high",
    "notes": "анализ последних реплик"
  }
}

---

## ПРАВИЛА АНАЛИЗА:

- Оцени ОБЩУЮ динамику, а не одну фразу
- Если клиент пишет длинную жалобу → это уже сигнал (даже без мата)
- Если есть угроза отзывов → это почти всегда alert=true + high
- Расторжение = всегда high
- Если несколько проблем → выбери главную:
  приоритет:
  1. client_loss
  2. conflict
  3. quality_issues

- В evidence — реальные цитаты
- В who — только участники конфликта
- Toxicity — только последние 5–10 сообщений

---

## ГЛАВНОЕ:

Ты не просто ищешь конфликт — ты выявляешь РИСК ДЛЯ БИЗНЕСА.

Даже вежливый, но разочарованный клиент — это alert=true.
"""

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

    # 1) Частый случай: JSON внутри fenced-блока ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if inner:
            return json.loads(inner)

    # 2) Иногда модель добавляет текст до/после JSON. Достаём первый сбалансированный объект.
    start = t.find("{")
    if start == -1:
        return json.loads(t)  # пусть упадёт с понятной ошибкой

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = t[start : i + 1].strip()
                if candidate:
                    return json.loads(candidate)
                break

    # fallback: старое поведение — последний { ... } в конце
    m2 = re.search(r"\{[\s\S]*\}\s*$", t)
    if m2:
        return json.loads(m2.group(0))
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
    delay_labels = transcript_delay_signals(transcript)
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

    if (not forced) and (not result.alert) and len(delay_labels) >= 2:
        ev_msgs, who_hint = transcript_delay_message_evidence(transcript)
        if ev_msgs:
            return AnalysisResult(
                alert=True,
                severity="low",
                category="quality_issues",
                title="Задержка ответа / нарушены ожидания",
                summary=(
                    "Участник(и) вежливо, но явно фиксируют задержку и несоблюдение обещанного срока. "
                    "Это ранний сигнал потери доверия и риска эскалации претензий."
                ),
                who=who_hint or "—",
                evidence=ev_msgs,
                toxicity=result.toxicity,
            )

    if forced and result.alert and result.severity == "low":
        return result.model_copy(update={"severity": "medium"})

    return result
