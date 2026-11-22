"""
tests/test_rag_service.py

Tests for app/rag/rag_service.py -- the retrieval + LLM + citation-verifier
stage. Everything external is replaced with a test double, so this suite
needs no Qdrant server, no embedding model download, no network, and no real
Gemini API key:

    retrieve()          -> monkeypatched to return hand-built BNS chunks
    genai.Client        -> replaced with a fake that returns fixed text

What is NOT faked is the part that matters most: verify_citations(). These
tests assert that citation objects come from chunk metadata, that invalid
markers are removed from the answer, and that an empty retrieval never
reaches the model at all.

Run with: pytest tests/test_rag_service.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.models.chunk import Chunk, ChunkMetadata, DocumentType, LegalStatus
from app.models.rag_response import RagResponse
from app.rag import rag_service
from app.rag.prompt_builder import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _bns_chunk(*, chunk_id: str, section: str, section_title: str, page: int, text: str) -> Chunk:
    """A realistic BNS chunk, built exactly like the real pipeline builds
    them (ChunkMetadata with section + section_title + page + source)."""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id="bns_2023",
            document_title="Bharatiya Nyaya Sanhita, 2023",
            document_type=DocumentType.ACT,
            legal_status=LegalStatus.CURRENT,
            chapter="Chapter VI",
            chapter_title="OF OFFENCES AFFECTING THE HUMAN BODY",
            section=section,
            section_title=section_title,
            page=page,
            source="India Code",
            source_url="https://www.indiacode.nic.in/handle/123456789/21356",
        ),
    )


@pytest.fixture
def bns_chunks() -> list[Chunk]:
    """The two provisions a 'punishment for murder' query really returns."""
    return [
        _bns_chunk(
            chunk_id="bns_2023__Section__103__0",
            section="103",
            section_title="Punishment for murder",
            page=47,
            text=(
                "103. Punishment for murder.—(1) Whoever commits murder shall be "
                "punished with death or imprisonment for life, and shall also be "
                "liable to fine."
            ),
        ),
        _bns_chunk(
            chunk_id="bns_2023__Section__104__0",
            section="104",
            section_title="Punishment for murder by life-convict",
            page=47,
            text=(
                "104. Punishment for murder by life-convict.—Whoever, being under "
                "sentence of imprisonment for life, commits murder, shall be "
                "punished with death or with imprisonment for life."
            ),
        ),
    ]


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, text: str):
        self._text = text
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._text)


class _FakeClient:
    """Stand-in for google.genai.Client with just the surface we use."""

    def __init__(self, text: str):
        self.models = _FakeModels(text)


@pytest.fixture
def api_key(monkeypatch):
    """Pretend a key is configured so _generate_answer() proceeds to the
    (faked) client instead of short-circuiting on the missing-key guard."""
    monkeypatch.setattr(rag_service.settings, "GOOGLE_API_KEY", "test-key-not-real")


@pytest.fixture
def fake_gemini(monkeypatch):
    """Install a fake Gemini client returning `text`, and hand back the
    client so a test can inspect what prompt was actually sent."""

    def _install(text: str) -> _FakeClient:
        client = _FakeClient(text)
        monkeypatch.setattr(rag_service.genai, "Client", lambda api_key: client)
        return client

    return _install


# ---------------------------------------------------------------------------
# (a) Empty retrieval must skip the LLM entirely
# ---------------------------------------------------------------------------


def test_empty_retrieval_skips_llm_and_returns_no_citations(monkeypatch, api_key):
    """No chunks -> refusal, no citations, and the model is never contacted."""
    calls = {"retrieve": 0}

    def _retrieve(**kwargs):
        calls["retrieve"] += 1
        return []

    def _client_must_not_be_built(*args, **kwargs):
        raise AssertionError(
            "genai.Client must not be constructed when retrieval returns no chunks"
        )

    monkeypatch.setattr(rag_service, "retrieve", _retrieve)
    monkeypatch.setattr(rag_service.genai, "Client", _client_must_not_be_built)

    response = rag_service.answer_query("what is the punishment for murder?")

    assert calls["retrieve"] == 1
    assert response.citations == []
    assert response.retrieved_chunks == []
    assert "no relevant provisions" in response.answer.lower()
    assert response.query == "what is the punishment for murder?"


def test_empty_retrieval_still_passes_filters_through(monkeypatch, api_key):
    """The refusal path must not silently drop the caller's filters."""
    captured: dict = {}

    def _retrieve(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(rag_service, "retrieve", _retrieve)

    rag_service.answer_query(
        "theft", limit=3, document_type="ACT", legal_status="CURRENT"
    )

    assert captured["limit"] == 3
    assert captured["document_type"] is DocumentType.ACT
    assert captured["legal_status"] is LegalStatus.CURRENT


# ---------------------------------------------------------------------------
# (b) Valid markers map to the exact chunk metadata
# ---------------------------------------------------------------------------


def test_valid_markers_map_to_exact_chunk_metadata(monkeypatch, bns_chunks, api_key, fake_gemini):
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)
    fake_gemini(
        "Murder is punishable with death or imprisonment for life [1]. "
        "A life-convict who commits murder is punished the same way [2]."
    )

    response = rag_service.answer_query("punishment for murder", limit=5, legal_status="CURRENT")

    assert [c.article_or_section for c in response.citations] == ["Section 103", "Section 104"]
    assert [c.page for c in response.citations] == [47, 47]
    assert all(c.document_title == "Bharatiya Nyaya Sanhita, 2023" for c in response.citations)
    assert all(c.source == "India Code" for c in response.citations)
    assert response.citations[0].source_url == (
        "https://www.indiacode.nic.in/handle/123456789/21356"
    )

    # Valid markers stay in the text.
    assert "[1]" in response.answer
    assert "[2]" in response.answer

    # The caller also gets the raw chunks back, for debugging.
    assert [c.id for c in response.retrieved_chunks] == [
        "bns_2023__Section__103__0",
        "bns_2023__Section__104__0",
    ]


def test_single_marker_yields_single_citation(monkeypatch, bns_chunks, api_key, fake_gemini):
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)
    fake_gemini("Murder is punishable with death or life imprisonment [1].")

    response = rag_service.answer_query("punishment for murder")

    assert len(response.citations) == 1
    assert response.citations[0].article_or_section == "Section 103"


def test_repeated_marker_is_deduplicated(monkeypatch, bns_chunks, api_key, fake_gemini):
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)
    fake_gemini("Murder carries death or life [1]. It is defined in Section 101 [1].")

    response = rag_service.answer_query("punishment for murder")

    assert len(response.citations) == 1
    assert response.citations[0].article_or_section == "Section 103"


def test_prompt_contains_numbered_references_and_system_prompt(
    monkeypatch, bns_chunks, api_key, fake_gemini
):
    """The model must actually receive the references, in retrieval order,
    and the citation-enforcing system prompt."""
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)
    client = fake_gemini("Answer [1].")

    rag_service.answer_query("punishment for murder")

    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert call["model"] == rag_service.settings.GEMINI_MODEL_NAME
    assert "[1] Bharatiya Nyaya Sanhita, 2023, Section 103" in call["contents"]
    assert "[2] Bharatiya Nyaya Sanhita, 2023, Section 104" in call["contents"]
    assert call["config"].system_instruction == SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# (c) Hallucinated markers are stripped and never become citations
# ---------------------------------------------------------------------------


def test_hallucinated_marker_is_stripped_and_never_becomes_a_citation(
    monkeypatch, bns_chunks, api_key, fake_gemini
):
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)
    fake_gemini(
        "Murder is punishable with death or life imprisonment [1]. "
        "The offence is also described in Section 999 [99]."
    )

    response = rag_service.answer_query("punishment for murder")

    # The invented marker is gone from the user-visible answer...
    assert "[99]" not in response.answer
    # ...and produced no citation.
    assert len(response.citations) == 1
    assert response.citations[0].article_or_section == "Section 103"
    assert all(c.page != 99 for c in response.citations)
    # No dangling " ." left behind by the removed marker.
    assert " ." not in response.answer


def test_zero_index_is_stripped(monkeypatch, bns_chunks, api_key, fake_gemini):
    """[0] is out of range: references are 1-based."""
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)
    fake_gemini("Some claim [0] and a real one [2].")

    response = rag_service.answer_query("punishment for murder")

    assert "[0]" not in response.answer
    assert [c.article_or_section for c in response.citations] == ["Section 104"]


def test_only_invalid_markers_yields_no_citations(monkeypatch, bns_chunks, api_key, fake_gemini):
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)
    fake_gemini("This whole answer is ungrounded [7][42].")

    response = rag_service.answer_query("punishment for murder")

    assert response.citations == []
    assert "[7]" not in response.answer
    assert "[42]" not in response.answer


def test_verify_citations_with_no_chunks_strips_every_marker():
    """Direct unit test of the verifier's defensive branch."""
    answer, citations = rag_service.verify_citations("Claim [1] and [2].", [])

    assert citations == []
    assert "[1]" not in answer
    assert "[2]" not in answer


# ---------------------------------------------------------------------------
# Failure modes -> RagServiceError (main.py maps this to 503)
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_clean_error(monkeypatch, bns_chunks):
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)
    monkeypatch.setattr(rag_service.settings, "GOOGLE_API_KEY", "")

    with pytest.raises(rag_service.RagServiceError, match="GOOGLE_API_KEY"):
        rag_service.answer_query("punishment for murder")


def test_provider_failure_raises_rag_service_error(monkeypatch, bns_chunks, api_key):
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)

    class _ExplodingClient:
        def __init__(self, api_key):
            self.models = self

        def generate_content(self, **kwargs):
            raise RuntimeError("quota exceeded")

    monkeypatch.setattr(rag_service.genai, "Client", _ExplodingClient)

    with pytest.raises(rag_service.RagServiceError, match="Gemini generation failed"):
        rag_service.answer_query("punishment for murder")


def test_empty_completion_raises_rag_service_error(monkeypatch, bns_chunks, api_key, fake_gemini):
    monkeypatch.setattr(rag_service, "retrieve", lambda **kwargs: bns_chunks)
    fake_gemini("   ")

    with pytest.raises(rag_service.RagServiceError, match="empty answer"):
        rag_service.answer_query("punishment for murder")


def test_unknown_document_type_raises_value_error():
    with pytest.raises(ValueError, match="document_type"):
        rag_service.answer_query("theft", document_type="NOT_A_TYPE")


def test_unknown_legal_status_raises_value_error():
    with pytest.raises(ValueError, match="legal_status"):
        rag_service.answer_query("theft", legal_status="SOMETIME")


# ---------------------------------------------------------------------------
# FastAPI wiring: /query is an adapter, and errors map to real status codes
# ---------------------------------------------------------------------------


def _client():
    from fastapi.testclient import TestClient

    import app.main as main_module

    return TestClient(main_module.app), main_module


def test_query_endpoint_delegates_to_rag_service(monkeypatch):
    client, main_module = _client()
    captured: dict = {}

    def _fake_answer_query(**kwargs):
        captured.update(kwargs)
        return RagResponse(
            query=kwargs["query"], answer="ok", citations=[], retrieved_chunks=[]
        )

    monkeypatch.setattr(main_module, "answer_query", _fake_answer_query)

    response = client.post(
        "/query",
        json={
            "query": "punishment for murder",
            "limit": 3,
            "document_type": "ACT",
            "legal_status": "CURRENT",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "punishment for murder",
        "answer": "ok",
        "citations": [],
        "retrieved_chunks": [],
    }
    assert captured == {
        "query": "punishment for murder",
        "limit": 3,
        "document_type": "ACT",
        "legal_status": "CURRENT",
    }


def test_query_endpoint_maps_generation_failure_to_503(monkeypatch):
    client, main_module = _client()

    def _fail(**kwargs):
        raise rag_service.RagServiceError("GOOGLE_API_KEY is not configured")

    monkeypatch.setattr(main_module, "answer_query", _fail)

    response = client.post("/query", json={"query": "punishment for murder"})

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_query_endpoint_maps_bad_filter_to_400(monkeypatch):
    client, main_module = _client()

    def _fail(**kwargs):
        raise ValueError("Unknown document_type 'NOPE'")

    monkeypatch.setattr(main_module, "answer_query", _fail)

    response = client.post(
        "/query", json={"query": "theft", "document_type": "NOPE"}
    )

    assert response.status_code == 400
    assert "document_type" in response.json()["detail"]
