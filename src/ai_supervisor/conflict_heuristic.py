"""
Локальные признаки эскалации (мат, оскорбления, угрозы) — дополняют LLM,
чтобы не пропускать явные конфликты при консервативном ответе модели.
"""

from __future__ import annotations

import re

# Последние строки переписки — достаточно для короткого всплеска
_LOOKBACK_LINES = 12

_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"иди\s+нахуй|нахуй\s+тебе|пош[ёе]л\s+нахуй|пош[ёе]л\s+ты\b|пошел\s+ты\b|"
            r"съеб\w*|свалил\w*|отвали\s+нахуй|заткнись|завали\s+рот",
            re.I | re.UNICODE,
        ),
        "грубый уход из диалога / оскорбление",
    ),
    (
        re.compile(
            r"\bбля\w*\s+(иди|пош|нахуй|съеб)|бля\w*[,!.]?\s*иди",
            re.I | re.UNICODE,
        ),
        "мат + эскалация (как в «бля иди нахуй»)",
    ),
    (
        re.compile(
            r"\b(мудак|мразь|урод|дебил|дура|козл\w*|сука|пидор|"
            r"хуйло|даун|ублюд\w*)\b",
            re.I | re.UNICODE,
        ),
        "личные оскорбления",
    ),
    (
        re.compile(
            r"убью|побью|врежу|приеду|разбер\w*с\s+тобой|щ\s*тебе\s*покаж",
            re.I | re.UNICODE,
        ),
        "угрозы насилием",
    ),
    (
        re.compile(
            r"форс-?\s*мажор|авари\w*|критическ\w*\s+сбой|срыв\s+срок|"
            r"всё\s+пропало|отойд\w*\s+всё|не\s+успе\w*",
            re.I | re.UNICODE,
        ),
        "маркеры форс-мажора / срыва",
    ),
]


def transcript_conflict_signals(transcript: str) -> tuple[bool, list[str]]:
    if not transcript or not transcript.strip():
        return False, []
    lines = transcript.strip().splitlines()
    tail = "\n".join(lines[-_LOOKBACK_LINES:])
    blob = tail.lower()
    hits: list[str] = []
    for rx, label in _RULES:
        if rx.search(tail) or rx.search(blob):
            if label not in hits:
                hits.append(label)
    return (len(hits) > 0), hits


def llm_hint_for_forced_context(signals: list[str]) -> str:
    if not signals:
        return ""
    return (
        "\n\n[Для модели: в хвосте переписки есть явные признаки проблемы: "
        + "; ".join(signals)
        + ". Оцени как alert=true, category=conflict или force_majeure, "
        "severity не ниже medium.]"
    )
