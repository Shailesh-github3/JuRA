"""
app/main.py

FastAPI entry point for the JuRA Python AI/RAG engine.

Exposes exactly two routes:

  GET  /health  -> {"status": "ok"}      (liveness only, no dependencies)
  POST /query   -> RagResponse JSON      (retrieval + Gemini + verified citations)

/query is a thin adapter: all retrieval, prompting, generation and citation
verification live in app/rag/rag_service.py. There is deliberately no answer
text, no keyword matching and no citation literal in this file -- everything
the user sees is derived from chunks retrieved out of Qdrant at request time.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.models.rag_response import ChatTurn, RagResponse
from app.rag.rag_service import RagServiceError, answer_query

logger = logging.getLogger(__name__)

app = FastAPI(
    title="JuRA AI / RAG Engine",
    description="Internal FastAPI AI service for Indian Law research",
    version="2.0.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Legal question to query")
    limit: Optional[int] = Field(default=5, ge=1, le=20, description="Max number of chunks/citations")
    document_type: Optional[str] = Field(default=None, description="CONSTITUTION, ACT, CODE")
    legal_status: Optional[str] = Field(default=None, description="CURRENT, SUPERSEDED, REPEALED")
    history: list[ChatTurn] = Field(default_factory=list, description="Prior conversation turns for multi-turn context")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=RagResponse)
def query_rag(request: QueryRequest):
    """Answer a legal question from the indexed corpus.

    Status codes:
      200 - answer produced (citations may legitimately be empty only when
            retrieval found nothing, in which case the answer says so).
      400 - unknown document_type / legal_status value.
      503 - Gemini unavailable: no GOOGLE_API_KEY configured, or the provider
            call failed. Spring Boot's PythonRagClient turns this into a 502
            for the end user.
    """
    try:
        return answer_query(
            query=request.query,
            limit=request.limit or 5,
            document_type=request.document_type,
            legal_status=request.legal_status,
            history=request.history,
        )
    except ValueError as exc:
        # Invalid filter values are a client error, not an engine failure.
        logger.warning("Rejected /query request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RagServiceError as exc:
        logger.error("RAG generation failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
