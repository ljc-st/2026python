from __future__ import annotations

from tools.db_tools import run_sql


class SQLAgent:
    def run(self, query: str) -> dict:
        q = query.lower()
        if "highest" in q or "top" in q or "销量最高" in query or "最高销售额" in query:
            sql = """
            SELECT product_name, SUM(sales_amount) AS total_sales
            FROM sales
            GROUP BY product_name
            ORDER BY total_sales DESC
            LIMIT 1
            """
        elif "refund" in q or "退款率" in query:
            sql = """
            SELECT region,
                   ROUND(SUM(refund_count) * 1.0 / SUM(order_count), 4) AS refund_rate
            FROM sales
            GROUP BY region
            ORDER BY refund_rate DESC
            """
        elif "region" in q or "地区" in query:
            sql = """
            SELECT region, SUM(sales_amount) AS total_sales
            FROM sales
            GROUP BY region
            ORDER BY total_sales DESC
            """
        else:
            sql = """
            SELECT date, product_name, region, sales_amount, order_count, refund_count
            FROM sales
            ORDER BY date DESC
            LIMIT 10
            """
        result = run_sql(sql)
        result["agent"] = "sql"
        return result
