"""
tests/test_clarification_loop.py

Tests for Feature 4: fact extraction & missing-fact clarification loop.

Everything external is faked, so no Qdrant, no embedding model, no network
and no real Gemini key are required: analyze_query / retrieve / _generate_answer
are replaced with doubles where the test wants to pin behaviour.

Run with: pytest tests/test_clarification_loop.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.models.chunk import Chunk, ChunkMetadata, DocumentType, LegalStatus
from app.models.rag_response import ChatTurn, ExtractedFacts, RagResponse
from app.rag import rag_service
from app.rag.fact_extractor import analyze_query


def _chunk(cid="c1"):
    return Chunk(
        id=cid,
        text="Section 35 of BNSS ...",
        metadata=ChunkMetadata(
            document_id="bnss_2023",
            document_title="Bharatiya Nagarik Suraksha Sanhita, 2023",
            document_type=DocumentType.ACT,
            legal_status=LegalStatus.CURRENT,
            section="35",
            section_title="Arrest without warrant",
            page=10,
            source="India Code",
        ),
    )


@pytest.fixture
def chunks():
    return [_chunk()]


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "GOOGLE_API_KEY", "test-key")
    return "test-key"


# ---------------------------------------------------------------------------
# (a) Ambiguous personal scenario -> clarification, no retrieval, no LLM
# ---------------------------------------------------------------------------


def test_ambiguous_scenario_triggers_clarification_and_skips_retrieval(monkeypatch, api_key):
    facts = ExtractedFacts(
        legal_domain="criminal",
        extracted_facts=["friend detained by police"],
        missing_facts=["duration of detention", "whether a warrant was shown"],
        needs_clarification=True,
        clarification_question="How long was he held, and was a warrant shown?",
    )
    monkeypatch.setattr(rag_service, "analyze_query", lambda *a, **kw: facts)

    retrieve_calls = {"n": 0}

    def _retrieve_boom(**kw):
        retrieve_calls["n"] += 1
        raise AssertionError("retrieve must not be called during clarification")

    monkeypatch.setattr(rag_service, "retrieve", _retrieve_boom)
    monkeypatch.setattr(rag_service, "_generate_answer", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("LLM must not be called during clarification")
    ))

    response = rag_service.answer_query("Police detained my friend")

    assert response.needs_clarification is True
    assert response.citations == []
    assert response.retrieved_chunks == []
    assert response.answer == "How long was he held, and was a warrant shown?"
    assert response.extracted_facts is facts
    assert retrieve_calls["n"] == 0


# ---------------------------------------------------------------------------
# (b) Follow-up turn bypasses clarification (MAX_CLARIFICATION_TURNS=1)
# ---------------------------------------------------------------------------


def test_follow_up_turn_bypasses_clarification_and_runs_rag(monkeypatch, api_key, chunks):
    # Real analyze_query: with 1 prior assistant turn it must short-circuit
    # to needs_clarification=False WITHOUT calling the model.
    monkeypatch.setattr(rag_service.settings, "MAX_CLARIFICATION_TURNS", 1)
    monkeypatch.setattr(rag_service.settings, "CLARIFICATION_ENABLED", True)

    class _BoomClient:
        def __init__(self, api_key=None):
            raise AssertionError("Gemini client must not be built when capped")

    monkeypatch.setattr("app.rag.fact_extractor.genai.Client", _BoomClient)

    retrieve_calls = {"n": 0}

    def _retrieve(**kw):
        retrieve_calls["n"] += 1
        return chunks

    monkeypatch.setattr(rag_service, "retrieve", _retrieve)
    monkeypatch.setattr(rag_service, "_generate_answer", lambda prompt: "Grounded answer [1].")

    history = [
        ChatTurn(role="user", content="Police detained my friend"),
        ChatTurn(role="assistant", content="How long was he held, and was a warrant shown?"),
    ]

    response = rag_service.answer_query(
        "About 3 hours, no warrant shown.",
        history=history,
    )

    assert response.needs_clarification is False
    assert retrieve_calls["n"] == 1
    assert len(response.citations) == 1
    assert response.citations[0].article_or_section == "Section 35"
    assert response.extracted_facts is not None
    assert response.extracted_facts.needs_clarification is False


# ---------------------------------------------------------------------------
# (c) Direct statutory question -> no clarification, runs RAG on turn 1
# ---------------------------------------------------------------------------


def test_direct_statutory_question_runs_rag_on_turn_one(monkeypatch, api_key, chunks):
    facts = ExtractedFacts(
        legal_domain="constitutional",
        extracted_facts=[],
        missing_facts=[],
        needs_clarification=False,
        clarification_question=None,
    )
    monkeypatch.setattr(rag_service, "analyze_query", lambda *a, **kw: facts)
    monkeypatch.setattr(rag_service, "retrieve", lambda **kw: chunks)
    monkeypatch.setattr(rag_service, "_generate_answer", lambda prompt: "Article 21 protects life and liberty [1].")

    response = rag_service.answer_query("What is Article 21?")

    assert response.needs_clarification is False
    assert len(response.citations) == 1
    assert "Article 21" in response.answer
    assert response.extracted_facts is facts


# ---------------------------------------------------------------------------
# (d) CLARIFICATION_ENABLED=False bypasses analyze_query entirely
# ---------------------------------------------------------------------------


def test_clarification_disabled_bypasses_analyzer(monkeypatch, api_key, chunks):
    monkeypatch.setattr(rag_service.settings, "CLARIFICATION_ENABLED", False)

    analyze_calls = {"n": 0}

    def _analyze_boom(*a, **kw):
        analyze_calls["n"] += 1
        raise AssertionError("analyze_query must not be called when disabled")

    monkeypatch.setattr(rag_service, "analyze_query", _analyze_boom)
    monkeypatch.setattr(rag_service, "retrieve", lambda **kw: chunks)
    monkeypatch.setattr(rag_service, "_generate_answer", lambda prompt: "Answer [1].")

    response = rag_service.answer_query("Police detained my friend")

    assert analyze_calls["n"] == 0
    assert response.needs_clarification is False
    assert len(response.citations) == 1


# ---------------------------------------------------------------------------
# analyze_query direct behaviour (fake Gemini client)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, parsed=None, text=None):
        self.parsed = parsed
        self.text = text


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.models = self

    def generate_content(self, **kwargs):
        return self._response


def test_analyze_query_direct_question_returns_no_clarification(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "CLARIFICATION_ENABLED", True)
    monkeypatch.setattr(rag_service.settings, "MAX_CLARIFICATION_TURNS", 1)
    monkeypatch.setattr(rag_service.settings, "GOOGLE_API_KEY", "test-key")

    facts = ExtractedFacts(
        legal_domain="constitutional",
        extracted_facts=[],
        missing_facts=[],
        needs_clarification=False,
        clarification_question=None,
    )
    client = _FakeClient(_FakeResponse(parsed=facts))

    result = analyze_query("What is Article 21?", client=client)
    assert result.needs_clarification is False


def test_analyze_query_ambiguous_returns_clarification(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "CLARIFICATION_ENABLED", True)
    monkeypatch.setattr(rag_service.settings, "MAX_CLARIFICATION_TURNS", 1)
    monkeypatch.setattr(rag_service.settings, "GOOGLE_API_KEY", "test-key")

    facts = ExtractedFacts(
        legal_domain="criminal",
        extracted_facts=["friend detained by police"],
        missing_facts=["duration", "warrant"],
        needs_clarification=True,
        clarification_question="How long was he held, and was a warrant shown?",
    )
    client = _FakeClient(_FakeResponse(parsed=facts))

    result = analyze_query("Police detained my friend", client=client)
    assert result.needs_clarification is True
    assert result.clarification_question == "How long was he held, and was a warrant shown?"


def test_analyze_query_empty_clarification_question_falls_back(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "CLARIFICATION_ENABLED", True)
    monkeypatch.setattr(rag_service.settings, "MAX_CLARIFICATION_TURNS", 1)
    monkeypatch.setattr(rag_service.settings, "GOOGLE_API_KEY", "test-key")

    facts = ExtractedFacts(
        legal_domain="criminal",
        extracted_facts=[],
        missing_facts=["duration"],
        needs_clarification=True,
        clarification_question=None,  # empty -> fallback
    )
    client = _FakeClient(_FakeResponse(parsed=facts))

    result = analyze_query("Police detained my friend", client=client)
    assert result.needs_clarification is False


def test_analyze_query_disabled_returns_no_clarification(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "CLARIFICATION_ENABLED", False)
    monkeypatch.setattr(rag_service.settings, "GOOGLE_API_KEY", "test-key")

    called = {"n": 0}

    class _Boom:
        def __init__(self, api_key=None):
            called["n"] += 1

    monkeypatch.setattr("app.rag.fact_extractor.genai.Client", _Boom)
    result = analyze_query("Police detained my friend")
    assert result.needs_clarification is False
    assert called["n"] == 0


def test_analyze_query_capped_by_max_turns_skips_model(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "CLARIFICATION_ENABLED", True)
    monkeypatch.setattr(rag_service.settings, "MAX_CLARIFICATION_TURNS", 1)
    monkeypatch.setattr(rag_service.settings, "GOOGLE_API_KEY", "test-key")

    called = {"n": 0}

    class _Boom:
        def __init__(self, api_key=None):
            called["n"] += 1

    monkeypatch.setattr("app.rag.fact_extractor.genai.Client", _Boom)

    history = [
        ChatTurn(role="user", content="Police detained my friend"),
        ChatTurn(role="assistant", content="How long?"),
    ]
    result = analyze_query("3 hours", history=history)
    assert result.needs_clarification is False
    assert called["n"] == 0


def test_analyze_query_no_api_key_returns_no_clarification(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "CLARIFICATION_ENABLED", True)
    monkeypatch.setattr(rag_service.settings, "GOOGLE_API_KEY", "")
    result = analyze_query("Police detained my friend")
    assert result.needs_clarification is False