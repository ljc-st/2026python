from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import DB_URI

engine = create_engine(DB_URI)


def run_sql(query: str) -> dict:
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = list(result.keys())
            return {
                "ok": True,
                "query": query,
                "columns": columns,
                "rows": [list(row) for row in rows],
            }
    except SQLAlchemyError as e:
        return {"ok": False, "query": query, "error": str(e)}
