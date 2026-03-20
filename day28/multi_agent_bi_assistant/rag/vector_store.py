from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config.settings import VECTOR_DB_PATH

DOCS = [
    Document(page_content="Refund policy: products can be returned within 7 days after receipt if eligible."),
    Document(page_content="After-sales policy: defective products can be exchanged or repaired."),
    Document(page_content="Analytics guideline: reports should combine sales, refund rate, and regional difference."),
    Document(page_content="Operations rule: recommendations must cite structured evidence or policy support."),
]


def build_vector_store() -> FAISS:
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(DOCS, embeddings)
    return db


def load_or_build_vector_store() -> FAISS:
    path = Path(VECTOR_DB_PATH)
    embeddings = OpenAIEmbeddings()
    if path.exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    db = FAISS.from_documents(DOCS, embeddings)
    db.save_local(str(path))
    return db
