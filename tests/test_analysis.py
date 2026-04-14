import asyncio

import pytest

from ai_supervisor.analysis import AnalysisResult, parse_analysis_result, parse_json_loose
from ai_supervisor.llm_base import strip_code_fence


def test_strip_code_fence() -> None:
    raw = """```json
{"alert": false, "severity": "low", "category": "none", "title": "x", "summary": "", "who": "", "evidence": []}
```"""
    t = strip_code_fence(raw)
    assert '"alert"' in t


def test_parse_json_loose() -> None:
    text = 'Пояснение\n{"alert": true, "severity": "high", "category": "conflict", "title": "t", "summary": "s", "who": "a", "evidence": ["e"]}'
    data = parse_json_loose(text)
    r = parse_analysis_result(data)
    assert r.alert is True
    assert r.severity == "high"


def test_parse_analysis_result_who_as_list() -> None:
    from ai_supervisor.analysis import parse_analysis_result, parse_json_loose

    raw = (
        '{"alert": true, "severity": "high", "category": "conflict", '
        '"title": "t", "summary": "s", '
        '"who": ["Artem", "Maria"], "evidence": ["a: x"]}'
    )
    r = parse_analysis_result(parse_json_loose(raw))
    assert r.who == "Artem, Maria"


def test_parse_analysis_result_toxicity() -> None:
    data = parse_json_loose(
        '{"alert": true, "severity": "high", "category": "conflict", '
        '"title": "t", "summary": "s", "who": "u", "evidence": ["e"], '
        '"toxicity": {"insult": 0.9, "threat": 0, "profanity": 80, '
        '"harassment": 0, "hate_speech": 0, "overall": "high", "notes": "мат"}}'
    )
    r = parse_analysis_result(data)
    assert r.toxicity is not None
    assert r.toxicity.insult == 0.9
    assert r.toxicity.profanity == 0.8
    assert r.toxicity.overall == "high"


def test_analyze_transcript_mock_llm() -> None:
    from ai_supervisor.analysis import analyze_transcript
    from ai_supervisor.llm_base import LLMClient

    class Fake(LLMClient):
        async def complete(self, *, system: str, user: str) -> str:
            return (
                '{"alert": false, "severity": "low", "category": "none", '
                '"title": "ok", "summary": "", "who": "", "evidence": []}'
            )

    async def _run() -> None:
        r = await analyze_transcript(Fake(), chat_label="Test", transcript="hello")
        assert r.alert is False

    asyncio.run(_run())


def test_transcript_conflict_signals_user_examples() -> None:
    from ai_supervisor.conflict_heuristic import transcript_conflict_signals

    ok, labels = transcript_conflict_signals(
        "[12:00] a: бля иди нахуй\n[12:01] b: пошёл нахуй"
    )
    assert ok
    assert labels


def test_heuristic_no_false_positive_casual_chat() -> None:
    from ai_supervisor.conflict_heuristic import transcript_conflict_signals

    t = (
        "[1] a: повезло тебе, у нас по тц нельзя ходить с прикрытым лицом\n"
        "[2] b: сидеть в маске свиньи и спартанским шлемом\n"
        "[3] c: Просто у нас ,ходи в чем хочешь))\n"
        "[4] d: Всю зиму в скимаске гоняю"
    )
    ok, labels = transcript_conflict_signals(t)
    assert not ok
    assert not labels


def test_heuristic_no_blya_poshlo_prefix() -> None:
    from ai_supervisor.conflict_heuristic import transcript_conflict_signals

    ok, _ = transcript_conflict_signals("бля, пошло всё не так, но это не конфликт")
    assert not ok


def test_heuristic_no_osvalilis_substring() -> None:
    from ai_supervisor.conflict_heuristic import transcript_conflict_signals

    ok, _ = transcript_conflict_signals("камни освалились со склона")
    assert not ok


def test_transcript_conflict_message_evidence() -> None:
    from ai_supervisor.conflict_heuristic import transcript_conflict_message_evidence

    t = (
        "[100] Вася: привет всем\n"
        "[101] Петя: иди нахуй отсюда\n"
        "[102] Маша: ок, давайте без этого"
    )
    ev, who = transcript_conflict_message_evidence(t)
    assert ev
    assert any("Петя" in line and "иди нахуй" in line for line in ev)
    assert "Петя" in who


def test_analyze_transcript_heuristic_when_llm_fails() -> None:
    from ai_supervisor.analysis import analyze_transcript
    from ai_supervisor.llm_base import LLMClient

    class Broken(LLMClient):
        async def complete(self, *, system: str, user: str) -> str:
            raise RuntimeError("403 Forbidden")

    async def _run() -> None:
        toxic = "\u0438\u0434\u0438 \u043d\u0430\u0445\u0443\u0439"  # «иди нахуй»
        transcript = f"[12:00] u1: {toxic}"
        r = await analyze_transcript(
            Broken(),
            chat_label="test",
            transcript=transcript,
        )
        assert r.alert is True
        assert r.category == "conflict"
        assert r.who == "u1"
        assert r.evidence and "u1:" in r.evidence[0] and toxic in r.evidence[0]

    asyncio.run(_run())


def test_analyze_transcript_forces_alert_when_heuristic_fires() -> None:
    from ai_supervisor.analysis import analyze_transcript
    from ai_supervisor.llm_base import LLMClient

    class Stubborn(LLMClient):
        async def complete(self, *, system: str, user: str) -> str:
            return (
                '{"alert": false, "severity": "low", "category": "none", '
                '"title": "всё норм", "summary": "", "who": "", "evidence": []}'
            )

    async def _run() -> None:
        r = await analyze_transcript(
            Stubborn(),
            chat_label="Группа",
            transcript="[12:00] x: иди нахуй",
        )
        assert r.alert is True
        assert r.category == "conflict"
        assert r.severity == "high"

    asyncio.run(_run())
