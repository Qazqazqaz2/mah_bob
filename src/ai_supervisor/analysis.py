from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from ai_supervisor.conflict_heuristic import (
    llm_hint_for_forced_context,
    transcript_conflict_signals,
)
from ai_supervisor.llm_base import LLMClient, strip_code_fence
from ai_supervisor.storage import StoredLine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — аналитик рабочих чатов организаторов мероприятий.
Твоя задача — по фрагменту переписки определить, есть ли ПРОБЛЕМНАЯ ситуация для руководства.

Проблемными считай в том числе:
- назревающий или открытый конфликт, агрессия, обвинения;
- **мат, оскорбления, «иди нахуй», «пошёл ты», уход на личности** — в рабочем чате это почти всегда эскалация (alert=true, category=conflict);
- форс-мажор, критический сбой, угроза срыва сроков/мероприятия;
- явное игнорирование важных сообщений/задач (несколько напоминаний без реакции);
- признаки того, что организатор теряет контроль: хаос, противоречивые указания, отказ от ответственности;
- токсичность, угрозы, эскалация.

НЕ считай проблемой обычные рабочие вопросы, нейтральные уточнения, вежливые споры без риска.

Ответ строго в формате JSON (без пояснений до или после JSON):
{
  "alert": true/false,
  "severity": "low|medium|high",
  "category": "conflict|force_majeure|ignored_tasks|control_loss|other|none",
  "title": "краткий заголовок ситуации",
  "summary": "2-5 предложений: что произошло и почему это важно",
  "who": "кто замешан (имена из текста или роли)",
  "evidence": ["короткая цитата или перефраз из переписки"]
}
Если alert=false, поля summary/who/evidence всё равно заполни нейтрально (можно пустые строки/массив)."""


class AnalysisResult(BaseModel):
    alert: bool
    severity: str = "low"
    category: str = "none"
    title: str = ""
    summary: str = ""
    who: str = ""
    evidence: list[str] = Field(default_factory=list)


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


async def analyze_transcript(
    llm: LLMClient,
    *,
    chat_label: str,
    transcript: str,
) -> AnalysisResult:
    forced, signal_labels = transcript_conflict_signals(transcript)
    hint = llm_hint_for_forced_context(signal_labels)
    user = f"Чат: {chat_label}\n\nФрагмент переписки:\n{transcript}{hint}"
    raw = await llm.complete(system=SYSTEM_PROMPT, user=user)
    try:
        data = parse_json_loose(raw)
        result = AnalysisResult.model_validate(data)
    except Exception:
        logger.exception("Не удалось разобрать JSON анализа, сырой ответ: %s", raw[:2000])
        result = AnalysisResult(
            alert=False,
            category="none",
            title="Ошибка разбора ответа модели",
            summary="Проверьте формат ответа LLM и логи.",
            who="",
            evidence=[],
        )

    if forced and not result.alert:
        logger.info(
            "Эвристика конфликта сработала, модель не дала алерт — принудительный алерт: %s",
            signal_labels,
        )
        return AnalysisResult(
            alert=True,
            severity="high",
            category="conflict",
            title="Эскалация / грубость в переписке",
            summary=(
                "Зафиксированы формулировки, типичные для конфликта или оскорблений в рабочем чате. "
                "Рекомендуется вмешательство менеджера."
            ),
            who="см. последние реплики",
            evidence=signal_labels[:5],
        )

    if forced and result.alert and result.severity == "low":
        return result.model_copy(update={"severity": "medium"})

    return result
