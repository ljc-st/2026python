CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    region TEXT NOT NULL,
    sales_amount REAL NOT NULL,
    order_count INTEGER NOT NULL,
    refund_count INTEGER NOT NULL,
    date TEXT NOT NULL
);
