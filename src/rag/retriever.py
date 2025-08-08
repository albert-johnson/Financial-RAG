from __future__ import annotations

import os
from typing import Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from src.config import OPENAI_MODEL, OPENAI_API_KEY
from src.rag.vectorstore import VectorStoreManager
from src.rag.charting import is_chart_request, build_chart


SYSTEM_PROMPT = (
    "You are a finance RAG assistant. Answer accurately using the provided context. "
    "If tables are relevant, summarize key numbers. Cite filenames and page/sheet when possible."
)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer concisely."),
])


def _format_docs(docs: List[Document]) -> str:
    lines: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "")
        page = d.metadata.get("location") or f"chunk {d.metadata.get('chunk_index', '')}"
        lines.append(f"Source: {os.path.basename(src)} ({page})\n{d.page_content}")
    return "\n\n".join(lines)


def answer_query(query: str, k_docs: int = 6) -> Dict:
    vsm = VectorStoreManager()

    # Retrieve relevant documents and tables
    doc_hits = vsm.similarity_search(query, k=k_docs)
    table_hits = vsm.search_tables(query, k=4)

    context = _format_docs(doc_hits)
    if table_hits:
        table_context = "\n\n".join(
            [
                f"Table: {os.path.basename(t.metadata.get('source',''))} at {t.metadata.get('location','')} "
                f"Cols: {', '.join(t.metadata.get('columns', [])[:12])}"
                for t in table_hits
            ]
        )
        context = context + "\n\n" + table_context

    # Initialize LLM only if available
    if OPENAI_API_KEY:
        llm = init_chat_model(OPENAI_MODEL, model_provider="openai")
        chain = QA_PROMPT | llm
        response = chain.invoke({"question": query, "context": context})
        answer_text = response.content if hasattr(response, "content") else str(response)
    else:
        # Fallback summarization if no API key
        answer_text = (
            "[LLM not configured] Retrieved context shown below. Please configure OPENAI_API_KEY for full answers.\n\n"
            + context[:2000]
        )

    # Collect sources and snapshots (page images if any)
    sources = list({d.metadata.get("source") for d in doc_hits})
    snapshots: List[str] = []
    for s in sources:
        base = os.path.splitext(os.path.basename(s))[0]
        # snapshots saved as <stem>_page_*.png
        # We cannot reliably list files without FS, but try best effort
        try:
            import glob
            from src.config import PAGE_IMAGES_DIR
            pattern = os.path.join(str(PAGE_IMAGES_DIR), f"{base}_page_*.png")
            snapshots.extend(sorted(glob.glob(pattern))[:3])
        except Exception:
            pass

    result: Dict = {
        "answer": answer_text,
        "sources": sources,
        "snapshots": snapshots,
    }

    # Chart generation if requested implicitly
    if is_chart_request(query) and table_hits:
        # Pick the top table and generate a chart
        table_path = table_hits[0].metadata.get("table_path")
        if table_path:
            chart_path = build_chart(query, table_path=table_path)
            if chart_path:
                result["chart_path"] = chart_path

    return result