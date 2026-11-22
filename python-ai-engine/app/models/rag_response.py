"""
app/models/rag_response.py

Defines what a final ANSWER looks like -- the thing rag_service.py
returns after retrieval + LLM generation have both finished.

Why this exists as its own model instead of just returning a plain string:

Per Step 12 of the original plan, citations must come from the retrieved
chunk metadata -- NOT from the LLM inventing them. If rag_service.py just
returned a string of text, there'd be nothing forcing that rule. By making
"citations" a required, structured field (a list of real Chunk objects),
it's impossible to accidentally ship an answer with no traceable source,
or with a citation the LLM made up on its own.

This is also the exact shape that main.py (the future FastAPI endpoint)
will send back to Spring Boot as JSON.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.chunk import Chunk


class ChatTurn(BaseModel):
    """One turn of a multi-turn conversation, as sent by the backend
    (Java ChatService) in RagQueryRequest.history. Mirrors the backend's
    ChatTurnDto so the two sides stay in sync without a code generator."""

    role: Literal["user", "assistant"] = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., min_length=1, description="The turn's text")


class ExtractedFacts(BaseModel):
    """Result of the fact-extraction / clarification analyzer.

    The analyzer's job is NOT to answer the question -- it decides whether
    the user's situation is too underspecified to retrieve on, and if so,
    which 1-2 facts would unblock a real answer. When needs_clarification
    is False the rest of the pipeline runs normally.
    """

    legal_domain: str = Field(default="", description="e.g. 'criminal', 'constitutional', 'contract'")
    extracted_facts: list[str] = Field(default_factory=list, description="Facts already stated by the user")
    missing_facts: list[str] = Field(default_factory=list, description="Critical facts still needed")
    needs_clarification: bool = Field(default=False)
    clarification_question: Optional[str] = Field(default=None, description="Polite question asking for the missing facts")


class Citation(BaseModel):
    """
    A single, user-facing citation -- a trimmed-down view of a Chunk,
    showing only what's needed to display "where this came from"
    without dumping the full chunk text/vector into the response.
    """

    document_title: str = Field(..., description="e.g. 'Constitution of India'")
    article_or_section: str = Field(
        ..., description="e.g. 'Article 21' or 'Section 103' -- whichever applies"
    )
    page: int
    source: str = Field(..., description="e.g. 'Legislative Department', 'India Code'")
    source_url: str | None = None


class RagResponse(BaseModel):
    """
    The complete answer returned to whoever asked the question --
    whether that's a CLI script (search.py) during testing, or later
    Spring Boot calling the real FastAPI endpoint.
    """

    query: str = Field(..., description="The original question the user asked.")

    answer: str = Field(
        ..., description="The LLM-generated answer, built ONLY from retrieved chunks, or a clarification question."
    )

    citations: list[Citation] = Field(
        ...,
        description=(
            "Real sources the answer was built from. Built directly from "
            "the Chunk objects retrieval returned -- never invented by the LLM. "
            "Empty during a clarification turn."
        ),
    )

    # Keeping the raw retrieved chunks too (not just the trimmed Citations)
    # is useful for debugging: if an answer looks wrong, you can inspect
    # exactly what text the LLM was given, not just the final citation labels.
    retrieved_chunks: list[Chunk] = Field(
        default_factory=list,
        description="The full Chunk objects used as context, kept for debugging/inspection.",
    )

    # --- clarification loop (Feature 4) ---
    needs_clarification: bool = Field(
        default=False,
        description="True when the analyzer decided the query is too underspecified "
        "to retrieve on and `answer` is a clarification question instead of a "
        "grounded answer.",
    )
    extracted_facts: Optional[ExtractedFacts] = Field(
        default=None,
        description="Facts extracted from the user's situation and the missing "
        "facts that triggered clarification, when available.",
    )