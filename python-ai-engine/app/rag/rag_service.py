"""
app/rag/rag_service.py

Step 11 of the pipeline: the ONLY file allowed to call an LLM.

Everything before this point is deterministic and already verified:
  retriever.py      -> real Chunk objects out of Qdrant
  prompt_builder.py -> SYSTEM_PROMPT + a numbered [1]..[N] references block

This module joins them, calls Gemini, and then -- critically -- does NOT
trust the model's prose. The model is told to cite with bracket numbers
only; this file maps each [N] back to chunks[N-1] and builds Citation
objects exclusively from that chunk's metadata. A marker that points at a
reference that was never supplied is stripped out of the answer and logged,
so an invented citation can never reach the user.

The contract with the rest of the system is deliberately narrow:

    answer_query(query, limit, document_type, legal_status) -> RagResponse

Three distinct outcomes, kept separate on purpose:

  1. Retrieval found nothing        -> refusal, NO LLM call, no citations.
  2. Retrieval found chunks         -> LLM call, verified citations.
  3. LLM unavailable / no API key   -> RagServiceError (main.py maps it to 503).

Case 1 must never reach the LLM: prompt_builder.build_prompt() raises on an
empty chunk list precisely so an empty references block can never be sent
(see prompt_builder.py:96-114).
"""

import logging
import re

from google import genai
from google.genai import types

from app.config.settings import settings
from app.models.chunk import Chunk, DocumentType, LegalStatus
from app.models.rag_response import ChatTurn, Citation, ExtractedFacts, RagResponse
from app.rag.fact_extractor import analyze_query
from app.rag.prompt_builder import SYSTEM_PROMPT, build_prompt
from app.retrieval.retriever import retrieve

logger = logging.getLogger(__name__)

# Any bracketed integer the model writes, e.g. "[1]", "[12]". Deliberately
# matches [0] and negative-free garbage too (e.g. "[99]") so that INVALID
# markers are found and removed rather than silently ignored.
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Cosmetic cleanup after markers are stripped, so removing "[99]" from
# "punishable with life [99]." does not leave a floating " ." behind.
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([.,;:!?])")

NO_RESULTS_ANSWER = (
    "No relevant provisions were found in the indexed legal corpus for this "
    "question. JuRA answers only from the legal texts it has indexed (the "
    "Constitution, BNS, BNSS, BSA, CrPC and IPC), so it cannot answer this "
    "from general knowledge. Try rephrasing the question, or widening the "
    "document_type / legal_status filters."
)


class RagServiceError(RuntimeError):
    """Raised when an answer cannot be generated at all.

    Kept as its own type (rather than letting SDK exceptions escape) so
    main.py can map it to a single clean 503 instead of leaking provider
    error strings, and so callers can tell "the model is down" apart from
    "the question has no answer in the corpus".
    """


def _coerce_document_type(value: str | DocumentType | None) -> DocumentType | None:
    """Accept the enum, its string value, or None (the shape main.py gets
    from JSON), and return the enum qdrant_service.search() expects -- it
    calls `.value` on the filter, so a bare string would crash there."""
    if value is None or isinstance(value, DocumentType):
        return value
    try:
        return DocumentType(str(value).strip().upper())
    except ValueError as exc:
        allowed = ", ".join(member.value for member in DocumentType)
        raise ValueError(
            f"Unknown document_type {value!r}. Expected one of: {allowed}."
        ) from exc


def _coerce_legal_status(value: str | LegalStatus | None) -> LegalStatus | None:
    """Same coercion as _coerce_document_type, for legal_status."""
    if value is None or isinstance(value, LegalStatus):
        return value
    try:
        return LegalStatus(str(value).strip().upper())
    except ValueError as exc:
        allowed = ", ".join(member.value for member in LegalStatus)
        raise ValueError(
            f"Unknown legal_status {value!r}. Expected one of: {allowed}."
        ) from exc


def _citation_from_chunk(chunk: Chunk) -> Citation:
    """Build a user-facing Citation from a real chunk's metadata.

    This is the only place a Citation is ever constructed, which is what
    makes the "citations are never invented" guarantee structural rather
    than a matter of trusting the model. The Article/Section label follows
    the same rule prompt_builder._reference_label() uses, so what the model
    saw as "[1]" and what the user sees as a citation describe the same
    provision.
    """
    md = chunk.metadata

    if md.article:
        article_or_section = f"Article {md.article}"
    elif md.section:
        article_or_section = f"Section {md.section}"
    else:
        # Defensive: a chunk with neither an article nor a section number
        # still gets a usable label rather than a bare "None".
        article_or_section = md.document_title

    return Citation(
        document_title=md.document_title,
        article_or_section=article_or_section,
        page=md.page,
        source=md.source,
        source_url=md.source_url,
    )


def verify_citations(
    answer: str,
    chunks: list[Chunk],
) -> tuple[str, list[Citation]]:
    """Deterministic citation verification -- the guardrail the whole design
    hinges on.

    For every [N] in the model's answer:

      * 1 <= N <= len(chunks)  -> keep the marker, and remember
                                  chunks[N-1] as a real source.
      * anything else          -> remove the marker from the answer text and
                                  log a warning. It never becomes a Citation.

    Citations are deduplicated by their visible identity (document, article/
    section, page, source) because two retrieved chunks can belong to the
    same provision -- e.g. Article 366 is split into many clause-level
    chunks, and showing the same citation twelve times helps nobody.

    Returns (cleaned_answer, citations_in_order_of_first_mention).
    """
    if not chunks:
        # No references were supplied, so no bracket number can be valid.
        # Strip every marker rather than letting "[1]" imply a source.
        stripped = _CITATION_RE.sub("", answer)
        logger.warning(
            "Answer contained citation markers but retrieval supplied no chunks; "
            "all markers were stripped."
        )
        return _tidy(stripped), []

    citations: list[Citation] = []
    seen: set[tuple] = set()
    invalid_indices: list[int] = []

    def _replace(match: re.Match) -> str:
        index = int(match.group(1))
        if not 1 <= index <= len(chunks):
            invalid_indices.append(index)
            return ""

        citation = _citation_from_chunk(chunks[index - 1])
        identity = (
            citation.document_title,
            citation.article_or_section,
            citation.page,
            citation.source,
            citation.source_url,
        )
        if identity not in seen:
            seen.add(identity)
            citations.append(citation)
        return match.group(0)

    cleaned = _CITATION_RE.sub(_replace, answer)

    if invalid_indices:
        logger.warning(
            "Stripped %d ungrounded citation marker(s) %s -- only %d reference(s) "
            "were supplied to the model.",
            len(invalid_indices),
            sorted(set(invalid_indices)),
            len(chunks),
        )

    if not citations:
        logger.warning(
            "Answer contained no valid citation markers, so no citation could be "
            "attributed to the %d retrieved chunk(s).",
            len(chunks),
        )

    return _tidy(cleaned), citations


def _tidy(text: str) -> str:
    """Collapse whitespace left behind by removed markers."""
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text.strip()


def _generate_answer(prompt: str) -> str:
    """Send `prompt` (plus SYSTEM_PROMPT) to Gemini and return its text.

    Raises RagServiceError for every failure mode -- missing key, network
    error, provider error, empty completion -- so callers only ever have to
    handle one exception type.
    """
    if not settings.GOOGLE_API_KEY.strip():
        raise RagServiceError(
            "GOOGLE_API_KEY is not configured on the AI engine, so no answer can "
            "be generated. Set GOOGLE_API_KEY in python-ai-engine/.env and restart "
            "the service."
        )

    try:
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                # 0.0: this is a citation-bound extraction task, not creative
                # writing -- sampling variety would only add hallucination risk.
                temperature=0.0,
            ),
        )
    except Exception as exc:  # provider/network/auth errors all land here
        raise RagServiceError(f"Gemini generation failed: {exc}") from exc

    try:
        text = response.text
    except Exception as exc:  # e.g. no candidates / safety-blocked response
        raise RagServiceError(f"Gemini response contained no usable text: {exc}") from exc

    if not text or not text.strip():
        raise RagServiceError("Gemini returned an empty answer.")

    return text.strip()


def answer_query(
    query: str,
    limit: int = 5,
    document_type: str | DocumentType | None = None,
    legal_status: str | LegalStatus | None = None,
    history: list[ChatTurn] | None = None,
) -> RagResponse:
    """Answer `query` from the indexed legal corpus.

    Optional multi-turn `history` (prior user/assistant turns) is used for
    fact extraction / clarification (Feature 4) and to enrich the retrieval
    query on a follow-up turn.

    Outcomes:
      1. Clarification needed  -> clarification question, no retrieval, no LLM.
      2. Nothing retrieved      -> refusal, no LLM.
      3. Chunks retrieved       -> prompt -> Gemini -> verified citations.
    """
    history = history or []

    # Step 1: fact extraction / clarification analysis.
    if settings.CLARIFICATION_ENABLED:
        facts = analyze_query(query, history)
    else:
        facts = ExtractedFacts(needs_clarification=False)

    if facts.needs_clarification and facts.clarification_question:
        logger.info("Query %r needs clarification; skipping retrieval.", query)
        return RagResponse(
            query=query,
            answer=facts.clarification_question,
            citations=[],
            retrieved_chunks=[],
            needs_clarification=True,
            extracted_facts=facts,
        )

    # Step 2: build an enriched search query on follow-up turns so the
    # retrieval sees the original situation plus the user's clarification.
    if history:
        prior_user = [t.content for t in history if t.role == "user"]
        search_query = " ".join(prior_user + [query]).strip()
    else:
        search_query = query
    if not search_query:
        search_query = query

    chunks = retrieve(
        query=search_query,
        limit=limit,
        document_type=_coerce_document_type(document_type),
        legal_status=_coerce_legal_status(legal_status),
    )

    if not chunks:
        # Case 1: nothing retrieved. No LLM call, no citations.
        logger.info("No chunks retrieved for query %r; returning refusal.", search_query)
        return RagResponse(
            query=query,
            answer=NO_RESULTS_ANSWER,
            citations=[],
            retrieved_chunks=[],
            needs_clarification=False,
            extracted_facts=facts,
        )

    logger.info("Retrieved %d chunk(s) for query %r; generating answer.", len(chunks), search_query)

    # build_prompt raises on an empty list, which cannot happen here.
    prompt = build_prompt(search_query, chunks)
    raw_answer = _generate_answer(prompt)

    answer, citations = verify_citations(raw_answer, chunks)

    return RagResponse(
        query=query,
        answer=answer,
        citations=citations,
        retrieved_chunks=chunks,
        needs_clarification=False,
        extracted_facts=facts,
    )
