"""
app/retrieval/bm25_index.py

Lexical (BM25) index over the chunks stored in Qdrant.

WHY THIS FILE EXISTS:
  Dense embeddings (all-MiniLM-L6-v2) match concepts well but lose exact
  numerals: "Article 14" embeds close to "Article 228", "Article 21", etc.
  The hybrid retriever in retriever.py fuses this lexical branch with the
  dense one via RRF so exact section-number lookups ("Article 14",
  "Section 103 BNS") land on the right provision.

DESIGN:
  - The index is built LAZILY and CACHED: the first query scrolls the whole
    Qdrant collection once and tokenizes every chunk; later queries reuse it.
  - invalidate_cache() drops the cache; qdrant_service.upsert_chunks() calls
    it (via a lazy import, to avoid a circular import) so re-ingestion
    never serves stale BM25 data.
  - Tokenization is deliberately simple and number-preserving: lowercase +
    regex [a-z0-9]+, so "Article 14", "Section 103", "51A" and "BNS"
    survive as tokens. Anything fancier (stemming, stopword removal) would
    help prose but would destroy the exact codes we are here to find.
  - Filtering (document_type / legal_status) is applied at scoring time by
    building a filtered BM25 over the eligible subset, so a filtered search
    only ever scores eligible chunks.
"""

import logging
import re
from typing import Optional

from rank_bm25 import BM25Okapi

from app.config.settings import settings
from app.models.chunk import Chunk, DocumentType, LegalStatus
from app.vectorstore import qdrant_service

logger = logging.getLogger(__name__)

# Abbreviations for the six indexed corpora. Added to each chunk's token list
# at index time (derived from document_id) so a query that uses an
# abbreviation ("BNS", "IPC", ...) can match even though the stored text
# spells the name out ("Bharatiya Nyaya Sanhita"). Keys are document_id
# prefixes; values are the abbreviation token.
_DOC_ABBREV = {
    "constitution_": "constitution",
    "bns_": "bns",
    "bnss_": "bnss",
    "bsa_": "bsa",
    "ipc_": "ipc",
    "crpc_": "crpc",
}


def _abbrev_for(document_id: str) -> Optional[str]:
    for prefix, abbrev in _DOC_ABBREV.items():
        if (document_id or "").startswith(prefix):
            return abbrev
    return None


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + extract alphanumeric tokens, keeping section codes."""
    return _TOKEN_RE.findall((text or "").lower())


# Module-level cache. None means "not loaded yet"; an empty list means "loaded
# but the collection was empty at load time" (so we don't keep re-scrolling).
_cache: dict = {"chunks": None, "tokens": None, "bm25": None}


def _scroll_all_chunks() -> list[Chunk]:
    """Scroll every point out of Qdrant and rebuild Chunk objects."""
    client = qdrant_service._get_client()
    chunks: list[Chunk] = []
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                try:
                    chunks.append(qdrant_service._payload_to_chunk(point.payload))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Skipping unparseable Qdrant point: %s", exc)
            if offset is None:
                break
    except Exception as exc:
        # Collection missing / server down: treat as empty so callers get []
        # instead of a crash. invalidate_cache() + a retry will pick it up
        # once the collection exists.
        logger.warning("BM25 index could not scroll Qdrant: %s", exc)
        return []
    return chunks


def ensure_index() -> None:
    """Load + cache the index on first use. Idempotent."""
    global _cache
    if _cache["chunks"] is not None:
        return
    chunks = _scroll_all_chunks()
    tokens = []
    for c in chunks:
        toks = tokenize(c.text)
        abbrev = _abbrev_for(c.metadata.document_id)
        if abbrev:
            toks = toks + [abbrev]
        tokens.append(toks)
    if chunks and any(tokens):
        try:
            bm25 = BM25Okapi(tokens)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("BM25Okapi construction failed: %s", exc)
            bm25 = None
    else:
        bm25 = None
    _cache["chunks"] = chunks
    _cache["tokens"] = tokens
    _cache["bm25"] = bm25
    logger.info("BM25 index built over %d chunk(s).", len(chunks))


def invalidate_cache() -> None:
    """Drop the cached index. Called by qdrant_service.upsert_chunks() after
    any write, so re-ingestion never serves stale BM25 data."""
    global _cache
    _cache = {"chunks": None, "tokens": None, "bm25": None}


def _coerce(value, cls):
    if value is None or isinstance(value, cls):
        return value
    try:
        return cls(str(value).strip().upper())
    except ValueError:
        return value


def search(
    query: str,
    limit: int = 5,
    document_type: DocumentType | None = None,
    legal_status: LegalStatus | None = None,
) -> list[Chunk]:
    """Return up to `limit` chunks ranked by BM25, honouring the same filters
    as qdrant_service.search(). Only hits with score > 0 are returned."""
    ensure_index()

    chunks = _cache["chunks"] or []
    tokens = _cache["tokens"] or []
    if not chunks:
        return []

    document_type = _coerce(document_type, DocumentType)
    legal_status = _coerce(legal_status, LegalStatus)

    # Eligible subset (filters applied before scoring).
    eligible: list[tuple[int, Chunk]] = []
    for i, chunk in enumerate(chunks):
        md = chunk.metadata
        if document_type is not None and md.document_type != document_type:
            continue
        if legal_status is not None and md.legal_status != legal_status:
            continue
        eligible.append((i, chunk))

    if not eligible:
        return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    # Rebuild BM25 over the filtered subset when a filter is active, so
    # scores are computed only among eligible chunks. Unfiltered queries
    # reuse the cached index.
    filtered = document_type is not None or legal_status is not None
    if filtered:
        sub_tokens = [tokens[i] for i, _ in eligible]
        try:
            bm25 = BM25Okapi(sub_tokens)
        except Exception:  # pragma: no cover - defensive
            return []
    else:
        bm25 = _cache["bm25"]
        if bm25 is None:
            return []

    scores = bm25.get_scores(q_tokens)

    scored: list[tuple[float, Chunk]] = []
    for j, (_, chunk) in enumerate(eligible):
        score = float(scores[j])
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]
