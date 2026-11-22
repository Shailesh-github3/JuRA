"""
tests/test_hybrid_retriever.py

Tests for the Phase 4 hybrid retriever (BM25 + dense Qdrant + RRF).

Two kinds of tests:

  1. Controlled (monkeypatched branches) -- pin the RRF math, deduplication,
     candidate limiting and the out-of-domain guard without depending on a
     real embedding model or Qdrant server.
  2. Integration (real in-memory Qdrant + real BM25 index) -- prove that an
     exact section-number lookup surfaces the right chunk and that filters
     apply identically to both branches.

Run with: pytest tests/test_hybrid_retriever.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.models.chunk import Chunk, ChunkMetadata, DocumentType, LegalStatus
from app.retrieval import retriever


def _chunk(cid, text, *, doc_type=DocumentType.ACT, status=LegalStatus.CURRENT,
           section=None, article=None, title="Test Doc"):
    return Chunk(
        id=cid,
        text=text,
        metadata=ChunkMetadata(
            document_id="test_doc",
            document_title=title,
            document_type=doc_type,
            legal_status=status,
            section=section,
            section_title=section,
            article=article,
            article_title=article,
            page=1,
            source="test",
        ),
    )


# ---------------------------------------------------------------------------
# Controlled: RRF math, dedup, guard
# ---------------------------------------------------------------------------


def _patch_branches(monkeypatch, dense, bm25):
    monkeypatch.setattr("app.vectorstore.qdrant_service.search", lambda *a, **kw: list(dense))
    monkeypatch.setattr("app.retrieval.bm25_index.search", lambda *a, **kw: list(bm25))
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(retriever.settings if hasattr(retriever, "settings") else __import__("app.config.settings", fromlist=["settings"]).settings,
                        "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(__import__("app.config.settings", fromlist=["settings"]).settings, "RRF_K", 60)
    monkeypatch.setattr(__import__("app.config.settings", fromlist=["settings"]).settings, "HYBRID_CANDIDATE_MULTIPLIER", 2)


def test_rrf_fuses_branches_and_ranks_bm25_winner_first(monkeypatch):
    """When dense ranks chunk A #1 and BM25 ranks chunk B #1 (and B also
    appears in dense at #2), RRF must put B first: its combined score
    (1/(60+2) + 1/(60+1)) beats A's (1/(60+1))."""
    from app.config import settings as _settings

    a = _chunk("a", "alpha")
    b = _chunk("b", "beta")
    c = _chunk("c", "gamma")

    monkeypatch.setattr("app.vectorstore.qdrant_service.search", lambda *a_, **kw: [a, b])
    monkeypatch.setattr("app.retrieval.bm25_index.search", lambda *a_, **kw: [b, c])
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(_settings.settings, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(_settings.settings, "RRF_K", 60)
    monkeypatch.setattr(_settings.settings, "HYBRID_CANDIDATE_MULTIPLIER", 2)

    results = retriever.retrieve("q", limit=2)
    assert [r.id for r in results] == ["b", "a"]


def test_rrf_deduplicates_chunks_present_in_both_branches(monkeypatch):
    """A chunk appearing in both lists is scored twice but returned once."""
    from app.config import settings as _settings

    a = _chunk("a", "alpha")
    b = _chunk("b", "beta")

    monkeypatch.setattr("app.vectorstore.qdrant_service.search", lambda *a_, **kw: [a, b])
    monkeypatch.setattr("app.retrieval.bm25_index.search", lambda *a_, **kw: [a, b])
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(_settings.settings, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(_settings.settings, "RRF_K", 60)
    monkeypatch.setattr(_settings.settings, "HYBRID_CANDIDATE_MULTIPLIER", 2)

    results = retriever.retrieve("q", limit=5)
    assert [r.id for r in results] == ["a", "b"]
    assert len(results) == len({r.id for r in results})


def test_out_of_domain_guard_returns_empty_when_both_branches_empty(monkeypatch):
    """Both branches empty -> [] so rag_service refuses instead of generating."""
    from app.config import settings as _settings

    monkeypatch.setattr("app.vectorstore.qdrant_service.search", lambda *a_, **kw: [])
    monkeypatch.setattr("app.retrieval.bm25_index.search", lambda *a_, **kw: [])
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(_settings.settings, "HYBRID_SEARCH_ENABLED", True)

    assert retriever.retrieve("quantum chromodynamics gluon lattice") == []


def test_hybrid_disabled_falls_back_to_dense_only(monkeypatch):
    """HYBRID_SEARCH_ENABLED=false must skip BM25 entirely (ablation mode)."""
    from app.config import settings as _settings

    a = _chunk("a", "alpha")
    bm25_calls = {"n": 0}

    def _bm25(*a_, **kw):
        bm25_calls["n"] += 1
        return [a]

    monkeypatch.setattr("app.vectorstore.qdrant_service.search", lambda *a_, **kw: [a])
    monkeypatch.setattr("app.retrieval.bm25_index.search", _bm25)
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(_settings.settings, "HYBRID_SEARCH_ENABLED", False)

    results = retriever.retrieve("q", limit=2)
    assert [r.id for r in results] == ["a"]
    assert bm25_calls["n"] == 0


def test_empty_query_returns_empty_list(monkeypatch):
    from app.config import settings as _settings

    monkeypatch.setattr(_settings.settings, "HYBRID_SEARCH_ENABLED", True)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


# ---------------------------------------------------------------------------
# Integration: real in-memory Qdrant + real BM25 index
# ---------------------------------------------------------------------------


@pytest.fixture
def hybrid_corpus(monkeypatch):
    """Two chunks in an in-memory Qdrant, with the dense floor disabled so
    the fake-model tests isolate the BM25/RRF behaviour rather than the
    threshold. Uses the real BM25 index end to end."""
    from qdrant_client import QdrantClient
    from app.vectorstore import qdrant_service
    from app.embeddings import embedding_service

    class _FakeTokenizer:
        def __call__(self, text, truncation=False):
            return {"input_ids": text.split() * 2}

    class _FakeModel:
        max_seq_length = 300
        tokenizer = _FakeTokenizer()

        def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
            import numpy as np
            from app.config.settings import settings as s
            dim = s.EMBEDDING_DIMENSION
            return np.array([[float(len(t)), 1.0] + [0.0] * (dim - 2) for t in texts])

    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_service, "_client", client)
    monkeypatch.setattr(embedding_service, "_model", _FakeModel())
    monkeypatch.setattr(qdrant_service.settings, "MIN_SIMILARITY_SCORE", -1.0)

    chunks = [
        _chunk("bns_103", "Section 103 Punishment for murder. Whoever commits murder shall be punished with death.",
               section="103", title="Bharatiya Nyaya Sanhita, 2023"),
        _chunk("bns_104", "Section 104 Punishment for murder by life-convict.",
               section="104", title="Bharatiya Nyaya Sanhita, 2023"),
        _chunk("bns_105", "Section 105 Punishment for culpable homicide not amounting to murder.",
               section="105", title="Bharatiya Nyaya Sanhita, 2023"),
        _chunk("bns_106", "Section 106 Causing death by negligence.",
               section="106", title="Bharatiya Nyaya Sanhita, 2023"),
        _chunk("bns_107", "Section 107 Abetment of a thing.",
               section="107", title="Bharatiya Nyaya Sanhita, 2023"),
        _chunk("bns_108", "Section 108 Abettor when liable for a different act.",
               section="108", title="Bharatiya Nyaya Sanhita, 2023"),
        _chunk("bns_109", "Section 109 Attempt to murder.",
               section="109", title="Bharatiya Nyaya Sanhita, 2023"),
        _chunk("bns_110", "Section 110 Attempt to commit culpable homicide.",
               section="110", title="Bharatiya Nyaya Sanhita, 2023"),
        _chunk("const_14", "Article 14 Equality before law. The State shall not deny to any person equality before the law.",
               doc_type=DocumentType.CONSTITUTION, article="14", section=None, title="Constitution of India"),
        _chunk("const_21", "Article 21 Protection of life and personal liberty.",
               doc_type=DocumentType.CONSTITUTION, article="21", section=None, title="Constitution of India"),
    ]
    # fix metadata document_id to match expected
    for c in chunks:
        if c.id.startswith("bns"):
            c.metadata.document_id = "bns_2023"
        else:
            c.metadata.document_id = "constitution_2026"
    qdrant_service.ensure_collection()
    from app.embeddings import embedding_service
    embedding_service.embed_chunks(chunks)  # upsert_chunks requires vectors
    qdrant_service.upsert_chunks(chunks)  # invalidates BM25 cache

    from app.config import settings as _s
    monkeypatch.setattr(_s.settings, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(_s.settings, "RRF_K", 60)
    monkeypatch.setattr(_s.settings, "HYBRID_CANDIDATE_MULTIPLIER", 4)
    return chunks


def test_exact_lookup_surfaces_right_chunk_via_bm25(hybrid_corpus):
    """'Section 103' / 'Article 14' must resolve to the right provision.

    Uses the real in-memory BM25 index end to end: the exact numeral query
    must surface the chunk carrying that numeral. The controlled RRF-flip
    behaviour is pinned separately in
    test_rrf_fuses_branches_and_ranks_bm25_winner_first.
    """
    results = retriever.retrieve("Section 103", limit=5)
    assert any(r.id == "bns_103" for r in results), results

    results = retriever.retrieve("Article 14", limit=5)
    assert any(r.id == "const_14" for r in results), results


def test_filters_apply_to_both_branches(hybrid_corpus):
    """document_type / legal_status must restrict BOTH dense and BM25."""
    # Constitution-only search must not return the BNS chunk.
    results = retriever.retrieve("Section 103", limit=5, document_type="CONSTITUTION")
    assert all(r.metadata.document_type == DocumentType.CONSTITUTION for r in results)
    assert not any(r.id == "bns_103" for r in results)

    # A legal_status that matches nothing returns [].
    results = retriever.retrieve("Section 103", limit=5, legal_status="REPEALED")
    assert results == []


def test_unrelated_query_with_real_threshold_returns_empty(monkeypatch):
    """End-to-end guard: with the real threshold active, an unrelated query
    returns [] so rag_service refuses instead of generating."""
    # Use the real corpus via the live Qdrant server (skipped if unavailable).
    from app.vectorstore import qdrant_service
    try:
        qdrant_service._get_client().collection_exists(qdrant_service.settings.QDRANT_COLLECTION_NAME)
    except Exception:
        pytest.skip("Qdrant not reachable")
    if not qdrant_service._get_client().collection_exists(qdrant_service.settings.QDRANT_COLLECTION_NAME):
        pytest.skip("collection not ingested")

    from app.config import settings as _s
    monkeypatch.setattr(_s.settings, "MIN_SIMILARITY_SCORE", 0.35)
    monkeypatch.setattr(_s.settings, "HYBRID_SEARCH_ENABLED", True)
    assert retriever.retrieve("quantum chromodynamics gluon lattice") == []
