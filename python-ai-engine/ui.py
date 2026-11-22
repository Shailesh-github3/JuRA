"""
ui.py — JuRA minimal demo UI (Streamlit).

Single-file app that drives the full pipeline in-process:
  hybrid + cross-encoder retrieval -> Gemini generation -> deterministic
  citation verification -> (optional) missing-fact clarification loop.

Run from python-ai-engine/:
    streamlit run ui.py

No Spring Boot backend required: it calls app.rag.rag_service.answer_query()
directly, so the Qdrant collection must be ingested first
(scripts/ingest_documents.py).
"""

import sys
from pathlib import Path

# Make `app.*` importable when launched via `streamlit run ui.py`.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.models.rag_response import ChatTurn
from app.rag.rag_service import RagServiceError, answer_query

st.set_page_config(
    page_title="JuRA — Indian Legal Research",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ JuRA")
    st.caption("Juristic Retrieval-Augmented Intelligence")

    st.divider()
    st.subheader("Retrieval filters")

    doc_type_label = st.selectbox(
        "Document type",
        ["all", "constitution", "act", "code"],
        index=0,
    )
    doc_type_map = {"all": None, "constitution": "CONSTITUTION", "act": "ACT", "code": "CODE"}
    document_type = doc_type_map[doc_type_label]

    status_label = st.selectbox(
        "Legal status",
        ["all", "current", "superseded", "repealed"],
        index=0,
    )
    status_map = {"all": None, "current": "CURRENT", "superseded": "SUPERSEDED", "repealed": "REPEALED"}
    legal_status = status_map[status_label]

    limit = st.slider("Top-k (limit)", min_value=1, max_value=8, value=5)

    st.divider()
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Pipeline: hybrid (dense + BM25 + RRF) → cross-encoder rerank → "
        "Gemini → deterministic citation verification."
    )

# ---------------------------------------------------------------------------
# Main: disclaimer + chat
# ---------------------------------------------------------------------------
st.warning(
    "⚠️ **Educational use only.** JuRA is a research assistant over Indian "
    "legal texts (Constitution, BNS, BNSS, BSA, CrPC, IPC). It is not legal "
    "advice. Verify every citation against the original source before relying "
    "on it in any proceeding.",
    icon="⚠️",
)

st.title("JuRA — Indian Legal Research Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        rag = msg.get("rag")
        if rag is not None:
            if getattr(rag, "needs_clarification", False):
                with st.container(border=True):
                    st.markdown("#### 🔍 Clarification needed")
                    st.markdown(rag.answer)
                    ef = getattr(rag, "extracted_facts", None)
                    if ef is not None:
                        if getattr(ef, "extracted_facts", None):
                            st.markdown("**Extracted facts:** " + ", ".join(ef.extracted_facts))
                        if getattr(ef, "missing_facts", None):
                            st.markdown("**Missing facts:** " + ", ".join(ef.missing_facts))
            else:
                if getattr(rag, "citations", None):
                    with st.expander("✅ Verified Citations", expanded=False):
                        for c in rag.citations:
                            st.markdown(
                                f"**{c.document_title}** — {c.article_or_section} "
                                f"(p. {c.page})"
                            )
                            st.caption(
                                f"Source: {c.source}"
                                + (f" · {c.source_url}" if c.source_url else "")
                            )
                if getattr(rag, "retrieved_chunks", None):
                    with st.expander("📄 Retrieved Chunks", expanded=False):
                        for ch in rag.retrieved_chunks:
                            st.markdown(f"`{ch.id}`")
                            st.text(ch.text[:600])

# Chat input
if prompt := st.chat_input("Ask a legal question or describe a situation..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "rag": None})

    history = [
        ChatTurn(role=m["role"], content=m["content"])
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]

    with st.chat_message("assistant"):
        with st.spinner("Retrieving & generating..."):
            try:
                rag = answer_query(
                    query=prompt,
                    limit=limit,
                    document_type=document_type,
                    legal_status=legal_status,
                    history=history,
                )
                st.markdown(rag.answer)
                if rag.needs_clarification:
                    with st.container(border=True):
                        st.markdown("#### 🔍 Clarification needed")
                        ef = rag.extracted_facts
                        if ef is not None:
                            if ef.extracted_facts:
                                st.markdown("**Extracted facts:** " + ", ".join(ef.extracted_facts))
                            if ef.missing_facts:
                                st.markdown("**Missing facts:** " + ", ".join(ef.missing_facts))
                else:
                    if rag.citations:
                        with st.expander("✅ Verified Citations", expanded=True):
                            for c in rag.citations:
                                st.markdown(
                                    f"**{c.document_title}** — {c.article_or_section} "
                                    f"(p. {c.page})"
                                )
                                st.caption(
                                    f"Source: {c.source}"
                                    + (f" · {c.source_url}" if c.source_url else "")
                                )
                    if rag.retrieved_chunks:
                        with st.expander("📄 Retrieved Chunks", expanded=False):
                            for ch in rag.retrieved_chunks:
                                st.markdown(f"`{ch.id}`")
                                st.text(ch.text[:600])
                st.session_state.messages.append(
                    {"role": "assistant", "content": rag.answer, "rag": rag}
                )
            except RagServiceError as e:
                st.error(f"AI engine error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {e}", "rag": None}
                )
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {e}", "rag": None}
                )
    st.rerun()
