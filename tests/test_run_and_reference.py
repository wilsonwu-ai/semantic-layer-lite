import pathlib
import sqlite3
import tempfile
import unittest

from sll_testutil import demo, example_model, fetch
from semantic_layer_lite.compile import compile_query
from semantic_layer_lite.run import (ReadOnlyError, connect_readonly, run_plan, run_query, run_sql,
                                     same_rows)


class ReadOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, _ = demo()

    def test_run_sql_refuses_writes(self):
        before = fetch(self.db, "SELECT COUNT(*) FROM orders")[0][0]
        for stmt in ("DELETE FROM orders", "DROP TABLE orders", "UPDATE orders SET status = 'x'",
                     "INSERT INTO orders VALUES (1, 1, 'd', 's')", "PRAGMA query_only = OFF"):
            with self.subTest(stmt):
                with self.assertRaises(ReadOnlyError):
                    run_sql(self.db, stmt)
        self.assertEqual(fetch(self.db, "SELECT COUNT(*) FROM orders")[0][0], before)

    def test_connection_itself_is_read_only(self):
        before = fetch(self.db, "SELECT COUNT(*) FROM orders")[0][0]
        conn = connect_readonly(self.db)
        try:
            for stmt in ("DELETE FROM orders", "DROP TABLE orders", "CREATE TABLE t (x)"):
                with self.subTest(stmt):
                    with self.assertRaises(sqlite3.OperationalError):
                        conn.execute(stmt)
        finally:
            conn.close()
        self.assertEqual(fetch(self.db, "SELECT COUNT(*) FROM orders")[0][0], before)

    def test_query_only_pragma_blocks_writes_even_if_the_file_were_writable(self):
        conn = connect_readonly(self.db)
        try:
            self.assertEqual(conn.execute("PRAGMA query_only").fetchone()[0], 1)
        finally:
            conn.close()

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "missing.sqlite"
            with self.assertRaises(FileNotFoundError):
                connect_readonly(p)
            self.assertFalse(p.exists())

    def test_with_statement_is_allowed(self):
        cols, rows = run_sql(self.db, "WITH t AS (SELECT 1 AS one) SELECT one FROM t")
        self.assertEqual((cols, rows), (["one"], [(1,)]))

    def test_same_rows(self):
        self.assertTrue(same_rows([(1, 2.0)], [(1, 2.0000000001)]))
        self.assertFalse(same_rows([(1, 2.0)], [(1, 2.1)]))
        self.assertFalse(same_rows([(1,)], [(1,), (2,)]))
        self.assertFalse(same_rows([("a",)], [("b",)]))
        self.assertFalse(same_rows([(None,)], [(1.0,)]))


REFERENCE = [
    ("total revenue",
     {"metrics": ["revenue"]},
     "SELECT SUM(line_total) FROM order_items", ()),
    ("revenue by region",
     {"metrics": ["revenue"], "dimensions": ["region"]},
     "SELECT c.region, SUM(i.line_total) FROM order_items i "
     "JOIN orders o ON o.order_id = i.order_id JOIN customers c ON c.customer_id = o.customer_id "
     "GROUP BY c.region ORDER BY c.region", ()),
    ("revenue by category",
     {"metrics": ["revenue"], "dimensions": ["category"]},
     "SELECT p.category, SUM(i.line_total) FROM order_items i "
     "JOIN products p ON p.product_id = i.product_id GROUP BY p.category ORDER BY p.category", ()),
    ("orders by status",
     {"metrics": ["order_count"], "dimensions": ["status"]},
     "SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY status", ()),
    ("orders by carrier counted once each",
     {"metrics": ["order_count"], "dimensions": ["carrier"]},
     "SELECT s.carrier, COUNT(DISTINCT o.order_id) FROM orders o "
     "LEFT JOIN shipments s ON s.order_id = o.order_id GROUP BY s.carrier ORDER BY s.carrier", ()),
    ("cash by payment method",
     {"metrics": ["cash_collected"], "dimensions": ["method"]},
     "SELECT method, SUM(amount) FROM payments GROUP BY method ORDER BY method", ()),
    ("revenue by month for one region",
     {"metrics": ["revenue"], "dimensions": ["month"],
      "filters": [{"dimension": "region", "op": "=", "value": "East"}]},
     "SELECT substr(o.order_date, 1, 7), SUM(i.line_total) FROM order_items i "
     "JOIN orders o ON o.order_id = i.order_id JOIN customers c ON c.customer_id = o.customer_id "
     "WHERE c.region = ? GROUP BY 1 ORDER BY 1", ("East",)),
    ("units by category and status",
     {"metrics": ["units_sold"], "dimensions": ["category", "status"]},
     "SELECT p.category, o.status, SUM(i.quantity) FROM order_items i "
     "JOIN orders o ON o.order_id = i.order_id JOIN products p ON p.product_id = i.product_id "
     "GROUP BY 1, 2 ORDER BY 1, 2", ()),
]


class AgainstReferenceSql(unittest.TestCase):
    def test_compiled_results_equal_hand_written_sql(self):
        db, _ = demo()
        model = example_model()
        self.assertGreaterEqual(len(REFERENCE), 5)
        for label, query, ref_sql, ref_params in REFERENCE:
            with self.subTest(label):
                got = run_query(db, model, query)
                _cols, want = run_sql(db, ref_sql, ref_params)
                self.assertTrue(same_rows(got.rows, want), f"{label}: {got.rows[:3]} vs {want[:3]}")
                if query.get("dimensions"):
                    self.assertGreater(len(want), 1)  # grouped queries are not trivially empty

    def test_the_comparison_can_fail(self):
        db, _ = demo()
        got = run_query(db, example_model(), {"metrics": ["revenue"], "dimensions": ["region"]})
        _cols, wrong = run_sql(db, "SELECT region, COUNT(*) FROM customers GROUP BY region ORDER BY region")
        self.assertFalse(same_rows(got.rows, wrong))

    def test_naive_route_through_shipments_gives_a_different_total(self):
        db, _ = demo()
        good = run_query(db, example_model(), {"metrics": ["revenue"]}).rows[0][0]
        _c, naive = run_sql(db, "SELECT SUM(i.line_total) FROM orders o JOIN shipments s ON s.order_id = o.order_id "
                                "JOIN order_items i ON i.item_id = s.item_id")
        self.assertNotAlmostEqual(good, naive[0][0], places=2)

    def test_run_plan_and_run_query_agree(self):
        db, _ = demo()
        q = {"metrics": ["revenue"], "dimensions": ["month"]}
        a = run_plan(db, compile_query(example_model(), q))
        b = run_query(db, example_model(), q)
        self.assertEqual(a.rows, b.rows)
        self.assertEqual(a.columns, ["order_date_month", "revenue"])


if __name__ == "__main__":
    unittest.main()
