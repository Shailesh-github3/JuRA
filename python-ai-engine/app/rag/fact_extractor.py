"""
app/rag/fact_extractor.py

Feature 4 of the JuRA design: Fact Extraction & Missing-Fact Clarification.

Given a user's (possibly ambiguous, real-life) query plus the conversation
so far, decide whether the query is specific enough to retrieve on. If it is
a direct statutory question ("What is Article 21?", "Punishment for murder
under BNS"), needs_clarification is False and retrieval proceeds. If it is
an underspecified real-life situation ("Police detained my friend",
"Someone took my money"), extract the known facts, list the 2-3 missing
facts that would change the legal answer, and return a clarification
question instead of running premature retrieval.

This module is the ONLY place that calls the LLM for clarification; it uses
Gemini with structured JSON output (response_schema=ExtractedFacts) so the
result is validated by Pydantic rather than parsed by hand.

Design notes:
  - CLARIFICATION_ENABLED=False short-circuits to needs_clarification=False.
  - MAX_CLARIFICATION_TURNS caps the loop: once the assistant has already
    asked once, the next user turn always proceeds to retrieval.
  - Any parse failure or empty clarification question falls back to
    needs_clarification=False, so retrieval always proceeds rather than
    leaving the user stuck.
"""

import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from app.config.settings import settings
from app.models.rag_response import ChatTurn, ExtractedFacts

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = """You are JuRA's fact-extraction assistant for Indian law.

You receive a user query (and possibly prior conversation turns). Decide
whether the query is specific enough to retrieve a legal answer, or whether
critical facts are missing.

Rules:
1. Direct statutory/constitutional questions are ALWAYS specific enough:
   "What is Article 21?", "Punishment for murder under BNS",
   "Explain Section 35 BNSS", "What does Article 14 guarantee?"
   -> needs_clarification = false. Do not ask anything.
2. Real-life situations that name an incident but omit facts that change the
   legal outcome NEED clarification. Examples:
   - "Police detained my friend" (missing: how long, warrant?, cognizable
     offence?, magistrate informed?) -> needs_clarification = true.
   - "Someone took my money" (missing: force vs. deception vs. breach of
     trust, amount, relationship) -> needs_clarification = true.
   - "My landlord is evicting me" (missing: notice given?, lease terms?,
     rent arrears?) -> needs_clarification = true.
3. When clarifying, list only the 2-3 facts that would actually change the
   applicable section, and ask for them in one polite, concise question.
   Never ask for facts that do not affect the legal outcome.
4. legal_domain: a short label like "criminal", "constitutional",
   "contract", "property", "family". Empty string if unknown.
5. extracted_facts: facts the user HAS stated (short phrases).
6. missing_facts: the 2-3 facts still needed (short phrases).
7. clarification_question: a single polite question combining the missing
   facts. Must be non-empty whenever needs_clarification is true.

Return ONLY valid JSON matching the provided schema. Do not add prose.
"""


def _assistant_turn_count(history: list[ChatTurn]) -> int:
    return sum(1 for turn in history if turn.role == "assistant")


def analyze_query(
    query: str,
    history: Optional[list[ChatTurn]] = None,
    client: Optional[object] = None,
) -> ExtractedFacts:
    """Decide whether `query` needs clarification before retrieval.

    `history` is the conversation so far (user/assistant turns). `client` is
    an optional injected genai.Client (used by tests); when None a client is
    created from settings.GOOGLE_API_KEY.
    """
    history = history or []

    # Hard caps: disabled, or already clarified once -> proceed to retrieval.
    if not settings.CLARIFICATION_ENABLED:
        return ExtractedFacts(needs_clarification=False)
    if _assistant_turn_count(history) >= settings.MAX_CLARIFICATION_TURNS:
        return ExtractedFacts(needs_clarification=False)

    # No API key -> cannot analyze; proceed to retrieval (safe default).
    if not settings.GOOGLE_API_KEY.strip():
        logger.info("GOOGLE_API_KEY not set; skipping clarification analysis.")
        return ExtractedFacts(needs_clarification=False)

    if client is None:
        try:
            client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        except Exception as exc:
            logger.warning("Could not build Gemini client for clarification: %s", exc)
            return ExtractedFacts(needs_clarification=False)

    # Build a compact conversation context for the model.
    context_lines = []
    for turn in history[-6:]:
        context_lines.append(f"{turn.role.upper()}: {turn.content}")
    context_lines.append(f"USER: {query}")
    user_text = "\n".join(context_lines)

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL_NAME,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=ExtractedFacts,
            ),
        )
    except Exception as exc:
        logger.warning("Clarification analysis call failed: %s", exc)
        return ExtractedFacts(needs_clarification=False)

    # Parse: prefer response.parsed (Pydantic) if the SDK provides it,
    # otherwise fall back to JSON text.
    parsed = getattr(response, "parsed", None)
    facts: Optional[ExtractedFacts] = None
    if isinstance(parsed, ExtractedFacts):
        facts = parsed
    else:
        text = getattr(response, "text", None)
        if text:
            try:
                facts = ExtractedFacts.model_validate_json(text)
            except Exception:
                try:
                    facts = ExtractedFacts.model_validate(json.loads(text))
                except Exception as exc:
                    logger.warning("Could not parse clarification JSON: %s", exc)
                    return ExtractedFacts(needs_clarification=False)

    if facts is None:
        return ExtractedFacts(needs_clarification=False)

    # Safety: a clarification with no question is useless -> proceed.
    if facts.needs_clarification and not (facts.clarification_question or "").strip():
        logger.warning("Analyzer requested clarification without a question; proceeding to retrieval.")
        facts.needs_clarification = False
        facts.clarification_question = None

    return facts
