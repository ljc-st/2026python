from __future__ import annotations

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "demo.db"
SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"

DEMO_DATA = [
    ("Projector A", "East", 12000, 50, 2, "2026-03-01"),
    ("Projector B", "South", 8000, 35, 1, "2026-03-02"),
    ("Projector A", "North", 9000, 40, 3, "2026-03-03"),
    ("Projector C", "East", 6000, 20, 0, "2026-03-03"),
    ("Projector B", "West", 11000, 48, 4, "2026-03-04"),
    ("Projector A", "East", 12500, 52, 2, "2026-03-05"),
    ("Projector C", "South", 7200, 27, 1, "2026-03-06"),
    ("Projector B", "North", 9700, 41, 2, "2026-03-07"),
    ("Projector A", "West", 11300, 45, 3, "2026-03-08"),
    ("Projector C", "East", 6900, 23, 1, "2026-03-09"),
]


def init_db() -> None:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        cur.executescript(f.read())

    cur.execute("SELECT COUNT(*) FROM sales")
    count = cur.fetchone()[0]
    if count == 0:
        cur.executemany(
            """
            INSERT INTO sales (product_name, region, sales_amount, order_count, refund_count, date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            DEMO_DATA,
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_FILE}")
