"""
tests/test_reranker.py

Tests for app/retrieval/reranker.py (cross-encoder second stage).

The real BAAI/bge-reranker-base model is ~1 GB and slow to load, so every
test replaces `_get_reranker()` with a fake whose `.predict(pairs)` returns
a scripted list of scores. No network, no model download.

Run with: pytest tests/test_reranker.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.models.chunk import Chunk, ChunkMetadata, DocumentType, LegalStatus
from app.retrieval import reranker


def _chunk(cid, text="some passage"):
    return Chunk(
        id=cid,
        text=text,
        metadata=ChunkMetadata(
            document_id="test_doc",
            document_title="Test Doc",
            document_type=DocumentType.ACT,
            legal_status=LegalStatus.CURRENT,
            section="1",
            section_title="Test",
            page=1,
            source="test",
        ),
    )


class _FakeReranker:
    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return list(self._scores)


@pytest.fixture
def fake_reranker(monkeypatch):
    def _install(scores):
        fake = _FakeReranker(scores)
        monkeypatch.setattr(reranker, "_get_reranker", lambda: fake)
        return fake

    return _install


def test_rerank_reorders_by_cross_encoder_score(fake_reranker):
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    fake = fake_reranker([0.1, 0.9, 0.3])

    result = reranker.rerank("what is theft", [a, b, c], top_k=2)

    # Highest score (b) first, then c, a dropped by top_k.
    assert [r.id for r in result] == ["b", "c"]
    # predict was called once with (query, text) pairs in input order.
    assert len(fake.calls) == 1
    assert fake.calls[0] == [
        ("what is theft", "some passage"),
        ("what is theft", "some passage"),
        ("what is theft", "some passage"),
    ]


def test_rerank_top_k_larger_than_candidates_returns_all(fake_reranker):
    a, b = _chunk("a"), _chunk("b")
    fake_reranker([0.5, 0.2])

    result = reranker.rerank("q", [a, b], top_k=10)
    assert [r.id for r in result] == ["a", "b"]


def test_rerank_empty_input_skips_model_and_returns_empty(monkeypatch):
    called = {"n": 0}

    def _boom():
        called["n"] += 1
        raise AssertionError("reranker must not be loaded for empty input")

    monkeypatch.setattr(reranker, "_get_reranker", _boom)
    assert reranker.rerank("q", [], top_k=5) == []
    assert called["n"] == 0


def test_rerank_predict_failure_falls_back_to_input_order(fake_reranker, monkeypatch):
    a, b = _chunk("a"), _chunk("b")

    class _Broken:
        def predict(self, pairs):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(reranker, "_get_reranker", lambda: _Broken())
    result = reranker.rerank("q", [a, b], top_k=2)
    assert [r.id for r in result] == ["a", "b"]


def test_reranker_disabled_bypasses_reranking(monkeypatch):
    """RERANKER_ENABLED=False must keep the RRF order and never call the
    cross-encoder."""
    from app.config import settings as _s
    from app.retrieval import retriever

    a = _chunk("a", "alpha")
    b = _chunk("b", "beta")
    reranker_calls = {"n": 0}

    def _no_reranker():
        reranker_calls["n"] += 1
        raise AssertionError("reranker must not be invoked when disabled")

    monkeypatch.setattr("app.vectorstore.qdrant_service.search", lambda *x, **kw: [a, b])
    monkeypatch.setattr("app.retrieval.bm25_index.search", lambda *x, **kw: [b])
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(_s.settings, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(_s.settings, "RERANKER_ENABLED", False)
    monkeypatch.setattr("app.retrieval.reranker._get_reranker", _no_reranker)

    results = retriever.retrieve("q", limit=2)
    # RRF order: b (in both) then a.
    assert [r.id for r in results] == ["b", "a"]
    assert reranker_calls["n"] == 0


def test_reranker_enabled_invokes_cross_encoder(monkeypatch):
    """When enabled, retrieve() must hand fused candidates to the reranker."""
    from app.config import settings as _s
    from app.retrieval import retriever

    a = _chunk("a", "alpha")
    b = _chunk("b", "beta")
    captured = {}

    class _Fake:
        def predict(self, pairs):
            captured["pairs"] = list(pairs)
            return [0.1, 0.9]  # b should win

    monkeypatch.setattr("app.vectorstore.qdrant_service.search", lambda *x, **kw: [a, b])
    monkeypatch.setattr("app.retrieval.bm25_index.search", lambda *x, **kw: [a, b])
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(_s.settings, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(_s.settings, "RERANKER_ENABLED", True)
    monkeypatch.setattr(_s.settings, "RRF_K", 60)
    monkeypatch.setattr(_s.settings, "HYBRID_CANDIDATE_MULTIPLIER", 2)
    monkeypatch.setattr(_s.settings, "RERANKER_CANDIDATE_K", 15)
    monkeypatch.setattr("app.retrieval.reranker._get_reranker", lambda: _Fake())

    results = retriever.retrieve("q", limit=1)
    assert [r.id for r in results] == ["b"]
    assert captured["pairs"] == [("q", "alpha"), ("q", "beta")]