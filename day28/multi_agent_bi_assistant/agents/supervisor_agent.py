from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.analysis_agent import AnalysisAgent
from agents.memory_agent import MemoryAgent
from agents.rag_agent import RAGAgent
from agents.sql_agent import SQLAgent


class GraphState(TypedDict, total=False):
    user_id: str
    query: str
    route: str
    results: list[dict[str, Any]]
    final_answer: str


class SupervisorAgent:
    def __init__(self) -> None:
        self.sql_agent = SQLAgent()
        self.rag_agent = RAGAgent()
        self.analysis_agent = AnalysisAgent()
        self.memory_agent = MemoryAgent()
        self.graph = self._build_graph()

    def _router(self, state: GraphState) -> GraphState:
        query = state["query"].lower()
        routes: list[str] = []

        if any(x in query for x in ["remember", "memory", "preference", "偏好", "记住", "我之前说过"]):
            routes.append("memory")
        if any(x in query for x in ["sales", "refund", "region", "metric", "销量", "退款率", "销售额", "地区"]):
            routes.append("sql")
        if any(x in query for x in ["policy", "rule", "manual", "document", "政策", "规则", "文档"]):
            routes.append("rag")
        if any(x in query for x in ["trend", "analysis", "chart", "分析", "趋势", "图"]):
            routes.append("analysis")

        if not routes:
            routes.append("rag")

        return {**state, "route": ",".join(routes), "results": []}

    def _memory_node(self, state: GraphState) -> GraphState:
        if "memory" not in state.get("route", ""):
            return state
        result = self.memory_agent.run(state["user_id"], state["query"])
        return {**state, "results": state.get("results", []) + [result]}

    def _sql_node(self, state: GraphState) -> GraphState:
        if "sql" not in state.get("route", ""):
            return state
        result = self.sql_agent.run(state["query"])
        return {**state, "results": state.get("results", []) + [result]}

    def _rag_node(self, state: GraphState) -> GraphState:
        if "rag" not in state.get("route", ""):
            return state
        result = self.rag_agent.run(state["query"])
        return {**state, "results": state.get("results", []) + [result]}

    def _analysis_node(self, state: GraphState) -> GraphState:
        if "analysis" not in state.get("route", ""):
            return state
        result = self.analysis_agent.run(state["query"])
        return {**state, "results": state.get("results", []) + [result]}

    def _finalize(self, state: GraphState) -> GraphState:
        parts = []
        for item in state.get("results", []):
            agent_name = item.get("agent", "unknown")
            parts.append(f"[{agent_name}] {item}")
        final_answer = "\n\n".join(parts) if parts else "No result produced."
        return {**state, "final_answer": final_answer}

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("router", self._router)
        graph.add_node("memory", self._memory_node)
        graph.add_node("sql", self._sql_node)
        graph.add_node("rag", self._rag_node)
        graph.add_node("analysis", self._analysis_node)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("router")
        graph.add_edge("router", "memory")
        graph.add_edge("memory", "sql")
        graph.add_edge("sql", "rag")
        graph.add_edge("rag", "analysis")
        graph.add_edge("analysis", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def run(self, user_id: str, query: str) -> dict:
        state: GraphState = {"user_id": user_id, "query": query, "results": []}
        result = self.graph.invoke(state)
        return result
