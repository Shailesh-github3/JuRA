"""
app/retrieval/retriever.py

Step 10 of the pipeline: the boundary between retrieval and everything else.
Takes a plain text query and returns real Chunk objects; nothing here ever
imports or calls an LLM.

From Phase 4 this is a HYBRID retriever: dense Qdrant search (cosine, with
the MIN_SIMILARITY_SCORE noise gate) is fused with a lexical BM25 search
via Reciprocal Rank Fusion. The reason is empirical (see
scripts/evaluate_retrieval.py): dense embeddings match concepts well but
lose exact numerals, so "Article 14" retrieved Articles 228/21/137. BM25
finds the numeral; RRF keeps both signals without either one dominating.

Signature is unchanged from the dense-only version, so rag_service.py,
scripts/search_documents.py and scripts/evaluate_retrieval.py keep working
without edits. Set HYBRID_SEARCH_ENABLED=false in .env to fall back to
dense-only for ablation.
"""

from app.config.settings import settings
from app.embeddings.embedding_service import embed_query
from app.models.chunk import Chunk, DocumentType, LegalStatus
from app.retrieval import bm25_index, reranker
from app.vectorstore import qdrant_service


def _coerce(value, cls):
    """Accept an enum, its string value, or None (the shape callers pass
    over JSON). Returns the enum qdrant_service.search() and bm25_index
    both expect -- they call .value on it."""
    if value is None or isinstance(value, cls):
        return value
    try:
        return cls(str(value).strip().upper())
    except ValueError:
        return value


def retrieve(
    query: str,
    limit: int = 5,
    document_type: DocumentType | None = None,
    legal_status: LegalStatus | None = None,
) -> list[Chunk]:
    """Return up to `limit` chunks for `query`.

    document_type / legal_status are optional query-time filters, passed to
    BOTH the dense and the lexical branch so a filtered search only ever
    scores eligible chunks.

    Returns [] for an empty/whitespace query, and [] when BOTH branches come
    back empty (the out-of-domain guard), so rag_service.py's refusal path
    runs instead of a wasted generation call.
    """
    query = (query or "").strip()
    if not query:
        return []

    document_type = _coerce(document_type, DocumentType)
    legal_status = _coerce(legal_status, LegalStatus)

    limit = max(1, int(limit or 5))

    # Dense-only fallback (ablation / debugging).
    if not settings.HYBRID_SEARCH_ENABLED:
        query_vector = embed_query(query)
        return qdrant_service.search(
            query_vector,
            limit=limit,
            document_type=document_type,
            legal_status=legal_status,
        )

    candidate_limit = limit * settings.HYBRID_CANDIDATE_MULTIPLIER

    # Branch 1: dense (cosine, gated by MIN_SIMILARITY_SCORE inside search).
    query_vector = embed_query(query)
    dense_results = qdrant_service.search(
        query_vector,
        limit=candidate_limit,
        document_type=document_type,
        legal_status=legal_status,
    )

    # Branch 2: lexical BM25 (only score > 0 hits).
    bm25_results = bm25_index.search(
        query,
        limit=candidate_limit,
        document_type=document_type,
        legal_status=legal_status,
    )

    # Out-of-domain guard: both branches empty -> nothing relevant.
    if not dense_results and not bm25_results:
        return []

    # Reciprocal Rank Fusion over chunk_id.
    rrf_k = settings.RRF_K
    fused: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
        chunks_by_id[chunk.id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
        chunks_by_id[chunk.id] = chunk

    ordered_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    fused_candidates = [chunks_by_id[cid] for cid in ordered_ids]

    # Optional second stage: cross-encoder rerank over the fused candidates.
    if settings.RERANKER_ENABLED:
        candidate_k = max(limit, settings.RERANKER_CANDIDATE_K)
        return reranker.rerank(query, fused_candidates[:candidate_k], top_k=limit)

    return fused_candidates[:limit]
