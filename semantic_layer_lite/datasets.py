"""A seeded six-table store database, built to contain the fan-out trap.

An order can have several payments, and an order line can ship in several
shipments. Join orders to either one and a plain SUM counts rows twice.
Only two of the six links are declared foreign keys; the others are the
kind of undeclared links real warehouses are full of.
"""
from __future__ import annotations

import datetime as dt
import random
import sqlite3
from pathlib import Path
from typing import Any

DEMO_TABLES = ("customers", "products", "orders", "order_items", "payments", "shipments")

_DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    line_total REAL NOT NULL
);
CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    method TEXT NOT NULL,
    amount REAL NOT NULL,
    paid_date TEXT NOT NULL
);
CREATE TABLE shipments (
    shipment_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    carrier TEXT NOT NULL,
    shipped_date TEXT NOT NULL
);
"""

# (from_table, from_column, to_table, to_column): the links a careful person would approve.
_TRUE = (
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("orders", "customer_id", "customers", "customer_id"),
    ("payments", "order_id", "orders", "order_id"),
    ("shipments", "item_id", "order_items", "item_id"),
    ("shipments", "order_id", "orders", "order_id"),
)

_REGIONS = ("East", "North", "South", "West")
_CATEGORIES = ("Books", "Garden", "Kitchen", "Tools", "Toys")
_CARRIERS = ("DHL", "FedEx", "UPS")
_METHODS = ("bank", "card", "gift_card")


def true_joins() -> list[dict[str, str]]:
    """Return the six real joins of the demo database, sorted."""
    return [
        {"from_table": a, "from_column": b, "to_table": c, "to_column": d, "cardinality": "many_to_one"}
        for a, b, c, d in sorted(_TRUE)
    ]


def make_demo_db(
    path: str | Path,
    seed: int = 7,
    n_customers: int = 80,
    n_orders: int = 400,
) -> dict[str, Any]:
    """Create the demo database at path (replacing any file there).

    Returns {"joins": true joins, "rows": row count per table}. The same
    seed always builds the same database.
    """
    rng = random.Random(seed)
    p = Path(path)
    if p.exists():
        p.unlink()

    customers = []
    for i in range(1, n_customers + 1):
        signup = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randrange(365))
        customers.append((i, f"Customer {i:03d}", rng.choice(_REGIONS), signup.isoformat()))

    products = []
    for i in range(1, 31):
        products.append((i, f"Product {i:02d}", rng.choice(_CATEGORIES), round(rng.uniform(5, 120), 2)))
    price = {row[0]: row[3] for row in products}

    start = dt.datetime(2026, 1, 1)
    orders, items, payments, shipments = [], [], [], []
    item_id = payment_id = shipment_id = 0
    for order_id in range(1, n_orders + 1):
        when = start + dt.timedelta(days=rng.randrange(181), seconds=rng.randrange(86400))
        status = rng.choices(["complete", "pending", "cancelled"], weights=[8, 2, 1])[0]
        orders.append((order_id, rng.randint(1, n_customers), when.strftime("%Y-%m-%d %H:%M:%S"), status))

        order_total = 0.0
        order_items = []
        for _ in range(rng.randint(1, 4)):
            item_id += 1
            pid = rng.randint(1, 30)
            qty = rng.randint(1, 5)
            total = round(qty * price[pid], 2)
            order_total += total
            order_items.append(item_id)
            items.append((item_id, order_id, pid, qty, price[pid], total))

        parts = rng.choice([1, 1, 2, 3])
        share = round(order_total / parts, 2)
        for k in range(parts):
            payment_id += 1
            amount = share if k < parts - 1 else round(order_total - share * (parts - 1), 2)
            paid = (when + dt.timedelta(days=rng.randint(0, 3))).date().isoformat()
            payments.append((payment_id, order_id, rng.choice(_METHODS), amount, paid))

        for iid in order_items:
            if rng.random() < 0.85:
                for _ in range(2 if rng.random() < 0.25 else 1):
                    shipment_id += 1
                    sent = (when + dt.timedelta(days=rng.randint(1, 10))).date().isoformat()
                    shipments.append((shipment_id, order_id, iid, rng.choice(_CARRIERS), sent))

    conn = sqlite3.connect(str(p))
    try:
        conn.executescript(_DDL)
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?)", items)
        conn.executemany("INSERT INTO payments VALUES (?,?,?,?,?)", payments)
        conn.executemany("INSERT INTO shipments VALUES (?,?,?,?,?)", shipments)
        conn.commit()
    finally:
        conn.close()
    rows = {
        "customers": len(customers), "products": len(products), "orders": len(orders),
        "order_items": len(items), "payments": len(payments), "shipments": len(shipments),
    }
    return {"joins": true_joins(), "rows": rows}
