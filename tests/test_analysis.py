import asyncio

import pytest

from ai_supervisor.analysis import AnalysisResult, parse_json_loose
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
    r = AnalysisResult.model_validate(data)
    assert r.alert is True
    assert r.severity == "high"


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
        "[12:00] a: бля иди нахуй\n[12:01] b: пошёл ты"
    )
    assert ok
    assert labels


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
