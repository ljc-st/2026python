from __future__ import annotations

from rag.vector_store import load_or_build_vector_store

vector_db = None


def retrieve_docs(query: str, k: int = 3) -> dict:
    global vector_db
    if vector_db is None:
        vector_db = load_or_build_vector_store()
    docs = vector_db.similarity_search(query, k=k)
    return {
        "ok": True,
        "query": query,
        "matches": [doc.page_content for doc in docs],
    }
