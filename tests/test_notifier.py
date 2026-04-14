from ai_supervisor.analysis import AnalysisResult, ToxicityBreakdown
from ai_supervisor.notifier import _evidence_redundant, _evidence_signature, build_alert_text


def test_evidence_signature_casefold() -> None:
    s = _evidence_signature(["Artem: иДИ НАХУЙ", "  x:y  "])
    assert "artem: иди нахуй" in s
    assert "x:y" in s


def test_evidence_redundant_equal() -> None:
    a = frozenset({"artem: a", "artem: b"})
    assert _evidence_redundant(a, [a]) is True


def test_evidence_redundant_subset() -> None:
    big = frozenset({"artem: line1", "artem: line2"})
    small = frozenset({"artem: line2"})
    assert _evidence_redundant(small, [big]) is True


def test_evidence_not_redundant_when_new_line() -> None:
    old = frozenset({"artem: line1"})
    new = frozenset({"artem: line1", "artem: line2"})
    assert _evidence_redundant(new, [old]) is False


def test_build_alert_text_includes_toxicity() -> None:
    r = AnalysisResult(
        alert=True,
        severity="high",
        category="conflict",
        title="T",
        summary="S",
        who="U",
        evidence=["e"],
        toxicity=ToxicityBreakdown(
            insult=0.5,
            threat=0.0,
            profanity=0.25,
            harassment=0.0,
            hate_speech=0.0,
            overall="medium",
            notes="пример",
        ),
    )
    text = build_alert_text(
        chat_id=1, chat_title="X", message_ts=None, result=r
    )
    assert "Токсичность (ИИ)" in text
    assert "Оскорбления: 50%" in text
    assert "medium" in text
    assert "пример" in text
