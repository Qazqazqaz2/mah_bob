"""
Локальные признаки эскалации (мат, оскорбления, угрозы) — дополняют LLM,
чтобы не пропускать явные конфликты при консервативном ответе модели.
"""

from __future__ import annotations

import re

# Последние строки переписки — достаточно для короткого всплеска
_LOOKBACK_LINES = 12

# Высокая точность: без префикса «пош» (ложно на «бля, пошло…»), без «свалил» внутри «освалились»,
# без «пошёл ты» в нейтральных репликах («ты пошёл в магазин»).
_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?:^|[\s,.;:!?'\u00ab\u00bb()\x22])(?:"
            r"иди\s+нахуй|нахуй\s+тебе|"
            r"пош[ёе]л\s+нахуй|пошел\s+нахуй|"
            r"\bсъеб(?:ись|аться|ались|ал[аи]?|нись|нуть|ка|куй)?\b|"
            r"\bсвалил[а-яё]*\b|"
            r"\bотвали\s+нахуй|\bзаткнись\b|\bзавали\s+рот"
            r")",
            re.I | re.UNICODE,
        ),
        "грубый уход из диалога / оскорбление",
    ),
    (
        re.compile(
            r"\bбля\w*[,!.]?\s*(?:иди\s+нахуй|иди\b|нахуй\b|съебись|съеб\b)|"
            r"\bбля\w*\s+иди\s+нахуй",
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

_SOFT_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\bну\s+конечно\b|\bага,\s*конечно\b|\bкто\s+бы\s+сомневал\w*\b",
            re.I | re.UNICODE,
        ),
        "сарказм / пассивная агрессия",
    ),
    (
        re.compile(
            r"\bопять\s+ты\b|\bвот\s+опять\b|\bкак\s+всегда\b|\bвечно\s+ты\b",
            re.I | re.UNICODE,
        ),
        "обесценивание / упрёки",
    ),
    (
        re.compile(
            r"[«\"\']\s*умн\w+\s*[»\"\']|\bумн\w+\s+мнен\w*\b|\bучи\s+матчаст\w*\b",
            re.I | re.UNICODE,
        ),
        "троллинг / насмешка",
    ),
    (
        re.compile(
            r"\bахах+\b|\bаха+\b|\bлол\b|\brofl\b|\bкек\b",
            re.I | re.UNICODE,
        ),
        "насмешливый тон",
    ),
    (
        re.compile(
            r"\bты\s+всегда\s+прав[ао]\b|\bконечно\s+ты\s+прав[ао]\b",
            re.I | re.UNICODE,
        ),
        "саркастическая уступка",
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


def transcript_tension_signals(transcript: str) -> list[str]:
    """
    Мягкие маркеры напряжения (сарказм/пассивная агрессия/троллинг).
    Не используем для «принудительного» алерта, но подсвечиваем модели.
    """
    if not transcript or not transcript.strip():
        return []
    lines = transcript.strip().splitlines()
    tail = "\n".join(lines[-_LOOKBACK_LINES:])
    blob = tail.lower()
    hits: list[str] = []
    for rx, label in _SOFT_RULES:
        if rx.search(tail) or rx.search(blob):
            if label not in hits:
                hits.append(label)
    return hits


# ts в логе — число мс или иной текст в скобках (не только цифры подряд)
_TRANSCRIPT_LINE_RE = re.compile(r"^\[([^\]]+)\]\s+(.+?):\s*(.*)$", re.DOTALL)


def _text_triggers_any_rule(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(rx.search(t) for rx, _ in _RULES)


def _text_triggers_any_soft_rule(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(rx.search(t) for rx, _ in _SOFT_RULES)


def transcript_conflict_message_evidence(
    transcript: str,
    *,
    max_items: int = 5,
    max_text_len: int = 420,
) -> tuple[list[str], str]:
    """
    Реплики из хвоста переписки, где сработала эвристика: для алерта «кто / что написал».
    Формат строк транскрипта: [ts] Имя: текст (как в format_transcript).
    """
    if not transcript or not transcript.strip():
        return [], ""
    tail = transcript.strip().splitlines()[-_LOOKBACK_LINES:]
    evidence: list[str] = []
    senders: list[str] = []
    seen_key: set[tuple[str, str]] = set()
    for raw in tail:
        m = _TRANSCRIPT_LINE_RE.match(raw.strip())
        if not m:
            continue
        name = m.group(2).strip()
        text = (m.group(3) or "").strip()
        if not _text_triggers_any_rule(text):
            continue
        norm_body = " ".join(text.split()).casefold()[:400]
        dedupe = (name.casefold(), norm_body)
        if dedupe in seen_key:
            continue
        seen_key.add(dedupe)
        one_line = " ".join(text.split())
        if len(one_line) > max_text_len:
            one_line = one_line[: max_text_len - 1] + "…"
        evidence.append(f"{name}: {one_line}")
        senders.append(name)
        if len(evidence) >= max_items:
            break
    who = ", ".join(dict.fromkeys(senders))
    return evidence, who


def transcript_tension_message_evidence(
    transcript: str,
    *,
    max_items: int = 5,
    max_text_len: int = 420,
) -> tuple[list[str], str]:
    """Реплики из хвоста, где видны сарказм/пассивная агрессия (мягкие маркеры)."""
    if not transcript or not transcript.strip():
        return [], ""
    tail = transcript.strip().splitlines()[-_LOOKBACK_LINES:]
    evidence: list[str] = []
    senders: list[str] = []
    seen_key: set[tuple[str, str]] = set()
    for raw in tail:
        m = _TRANSCRIPT_LINE_RE.match(raw.strip())
        if not m:
            continue
        name = m.group(2).strip()
        text = (m.group(3) or "").strip()
        if not _text_triggers_any_soft_rule(text):
            continue
        norm_body = " ".join(text.split()).casefold()[:400]
        dedupe = (name.casefold(), norm_body)
        if dedupe in seen_key:
            continue
        seen_key.add(dedupe)
        one_line = " ".join(text.split())
        if len(one_line) > max_text_len:
            one_line = one_line[: max_text_len - 1] + "…"
        evidence.append(f"{name}: {one_line}")
        senders.append(name)
        if len(evidence) >= max_items:
            break
    who = ", ".join(dict.fromkeys(senders))
    return evidence, who


def llm_hint_for_forced_context(signals: list[str]) -> str:
    if not signals:
        return ""
    return (
        "\n\n[Для модели: в хвосте переписки есть явные признаки проблемы: "
        + "; ".join(signals)
        + ". Оцени как alert=true, category=conflict или force_majeure, "
        "severity не ниже medium.]"
    )


def llm_hint_for_soft_context(signals: list[str]) -> str:
    if not signals:
        return ""
    return (
        "\n\n[Для модели: в хвосте переписки есть мягкие маркеры напряжения (сарказм/пассивная агрессия): "
        + "; ".join(signals)
        + ". Если это мешает работе/эскалирует — оцени alert=true, category=conflict, severity low или medium.]"
    )
