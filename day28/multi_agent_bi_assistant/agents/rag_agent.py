from __future__ import annotations

from tools.rag_tools import retrieve_docs


class RAGAgent:
    def run(self, query: str) -> dict:
        result = retrieve_docs(query)
        result["agent"] = "rag"
        return result
