from __future__ import annotations

from tools.python_tools import analyze_sales_trend


class AnalysisAgent:
    def run(self, query: str) -> dict:
        result = analyze_sales_trend()
        result["agent"] = "analysis"
        result["query"] = query
        return result
