from datetime import datetime

from schemas import AnswerResponse, UserIntent


def test_answer_response_fields():
    response = AnswerResponse(
        question="What is the invoice total?",
        answer="INV-001 totals $22,000.",
        sources=["INV-001"],
        confidence=0.9,
    )
    assert response.question
    assert response.answer
    assert response.sources == ["INV-001"]
    assert 0 <= response.confidence <= 1
    assert isinstance(response.timestamp, datetime)


def test_user_intent_fields():
    intent = UserIntent(
        intent_type="calculation",
        confidence=0.8,
        reasoning="User asked to add invoice totals.",
    )
    assert intent.intent_type == "calculation"
    dumped = intent.model_dump()
    assert set(dumped) >= {"intent_type", "confidence", "reasoning"}
