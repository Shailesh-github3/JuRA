"""
app/retrieval/reranker.py

Second-stage cross-encoder reranker for the hybrid retriever.

WHY THIS EXISTS:
  RRF (Phase 4) fuses two independent ranked lists, but it never reads the
  (query, passage) pair jointly -- it only knows each branch's rank. A
  cross-encoder (BAAI/bge-reranker-base) reads the pair together and is far
  more accurate at deciding which of the RRF candidates actually answers the
  question. We run it over the top RERANKER_CANDIDATE_K fused candidates
  and keep the top `limit`. This is what lifts Recall@1 / MRR on top of the
  RRF baseline.

DESIGN:
  - Lazy singleton CrossEncoder, imported from sentence_transformers. The
    model is downloaded on first use and cached for the process lifetime, so
    tests can monkeypatch `_get_reranker()` with a fake.
  - rerank() is a pure function of (query, chunks, top_k): score every
    (query, chunk.text) pair, sort descending, slice. Empty input -> [].
  - Nothing here touches Qdrant or the LLM; the reranker is a drop-in
    post-processing step over whatever candidate list retrieve() hands it.
"""

import logging
from typing import Optional

from app.config.settings import settings
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    """Lazy singleton. Tests monkeypatch this to return a fake with a
    .predict(pairs) method."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
        logger.info("Loaded cross-encoder reranker: %s", settings.RERANKER_MODEL_NAME)
    return _reranker


def invalidate_cache() -> None:
    """Drop the cached reranker (tests / model swap)."""
    global _reranker
    _reranker = None


def rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """Re-score `chunks` against `query` with the cross-encoder and return
    the best `top_k`, highest score first.

    Returns [] for an empty candidate list without touching the model, so
    the out-of-domain guard in retriever.py never pays for a reranker call
    on a query that retrieved nothing.
    """
    if not chunks:
        return []

    top_k = max(1, int(top_k or len(chunks)))

    reranker = _get_reranker()
    pairs = [(query, chunk.text) for chunk in chunks]

    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        # A reranker failure must never take retrieval down: fall back to
        # the RRF order we already had.
        logger.warning("Cross-encoder rerank failed (%s); keeping RRF order.", exc)
        return list(chunks[:top_k])

    scored = sorted(zip(scores, chunks), key=lambda pair: float(pair[0]), reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
