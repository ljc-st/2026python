from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "demo.db"
CHART_DIR = BASE_DIR / "data" / "charts"


def analyze_sales_trend() -> dict:
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(
        "SELECT date, SUM(sales_amount) AS total_sales FROM sales GROUP BY date ORDER BY date",
        conn,
    )
    conn.close()

    if df.empty:
        return {"ok": False, "message": "No data found."}

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = CHART_DIR / "sales_trend.png"

    ax = df.plot(x="date", y="total_sales", kind="line", figsize=(8, 4))
    ax.figure.tight_layout()
    ax.figure.savefig(chart_path)
    ax.figure.clf()

    return {
        "ok": True,
        "summary": df.to_dict(orient="records"),
        "chart_path": str(chart_path),
    }
