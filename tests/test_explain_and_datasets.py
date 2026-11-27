import json
import pathlib
import tempfile
import unittest

from sll_testutil import demo, example_model, fetch
from semantic_layer_lite.compile import compile_query
from semantic_layer_lite.datasets import DEMO_TABLES, make_demo_db, true_joins
from semantic_layer_lite.explain import explain_plan, format_table, render_plan, render_result, result_records
from semantic_layer_lite.live import LiveError, build_prompt, parse_reply
from semantic_layer_lite.run import run_plan


class Explain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, _ = demo()
        cls.model = example_model()
        cls.plan = compile_query(cls.model, {"metrics": ["revenue"], "dimensions": ["region"]})

    def expected_join_ids(self):
        wanted = [{"order_items", "orders"}, {"orders", "customers"}]
        return sorted(j.id for j in self.model.joins if {j.from_table, j.to_table} in wanted)

    def test_explain_names_the_joins_and_the_metric_definition(self):
        info = explain_plan(self.plan)
        self.assertEqual(sorted(info["join_ids"]), self.expected_join_ids())
        self.assertEqual(len(info["join_ids"]), 2)
        m = info["metrics"][0]
        self.assertEqual(m["name"], "revenue")
        self.assertEqual(m["definition"], "SUM(line_total) over order_items")
        self.assertEqual(info["routes"]["customers"][0]["from"], "order_items")
        self.assertEqual(info["routes"]["customers"][-1]["to"], "customers")
        self.assertEqual(info["sql"], self.plan.sql)
        json.dumps(info)  # JSON-ready

    def test_rendered_text_shows_the_path_and_the_sql(self):
        text = render_plan(self.plan)
        for jid in self.expected_join_ids():
            self.assertIn(jid, text)
        self.assertIn("order_items -> orders -> customers", text)
        self.assertIn("SUM(line_total) over order_items", text)
        self.assertIn(self.plan.sql.splitlines()[0], text)

    def test_every_result_row_carries_its_audit(self):
        result = run_plan(self.db, self.plan)
        records = result_records(result)
        self.assertEqual(len(records), len(result.rows))
        self.assertGreater(len(records), 1)
        for rec, row in zip(records, result.rows):
            self.assertEqual(sorted(rec["_join_ids"]), self.expected_join_ids())
            self.assertEqual(rec["_metrics"]["revenue"], "SUM(line_total) over order_items")
            self.assertEqual(rec["region"], row[0])
            self.assertEqual(rec["revenue"], row[1])
        text = render_result(result)
        self.assertIn("How these numbers were made", text)
        self.assertIn(f"({len(result.rows)} rows)", text)

    def test_query_on_one_table_says_no_join_was_needed(self):
        plan = compile_query(self.model, {"metrics": ["order_count"], "dimensions": ["status"]})
        self.assertEqual(explain_plan(plan)["join_ids"], [])
        self.assertIn("none needed", render_plan(plan))

    def test_format_table_aligns_and_handles_null(self):
        text = format_table(["a", "bb"], [("x", None), ("yy", 1.5)])
        lines = text.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("NULL", lines[2])
        self.assertIn("1.50", lines[3])
        # column widths are 2 and 4 (the longest cell in each), joined by two spaces
        self.assertEqual([len(ln) for ln in lines], [8, 8, 8, 8])


class Datasets(unittest.TestCase):
    def test_same_seed_same_database_different_seed_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = (pathlib.Path(tmp) / n for n in ("a.sqlite", "b.sqlite", "c.sqlite"))
            make_demo_db(a, seed=3, n_customers=20, n_orders=60)
            make_demo_db(b, seed=3, n_customers=20, n_orders=60)
            make_demo_db(c, seed=4, n_customers=20, n_orders=60)
            for t in DEMO_TABLES:
                self.assertEqual(fetch(a, f"SELECT * FROM {t} ORDER BY 1"), fetch(b, f"SELECT * FROM {t} ORDER BY 1"))
            self.assertNotEqual(fetch(a, "SELECT * FROM orders ORDER BY 1"), fetch(c, "SELECT * FROM orders ORDER BY 1"))

    def test_row_counts_match_the_report(self):
        db, info = demo()
        for t, n in info["rows"].items():
            self.assertEqual(fetch(db, f"SELECT COUNT(*) FROM {t}")[0][0], n)
        self.assertEqual(sorted(info["rows"]), sorted(DEMO_TABLES))

    def test_true_joins_hold_in_the_data(self):
        db, info = demo()
        self.assertEqual(info["joins"], true_joins())
        for j in info["joins"]:
            missing = fetch(db, f'SELECT COUNT(*) FROM {j["from_table"]} WHERE {j["from_column"]} NOT IN '
                                f'(SELECT {j["to_column"]} FROM {j["to_table"]})')[0][0]
            self.assertEqual(missing, 0, j)

    def test_the_fan_out_trap_is_present(self):
        db, _ = demo()
        multi_pay = fetch(db, "SELECT COUNT(*) FROM (SELECT order_id FROM payments GROUP BY order_id HAVING COUNT(*) > 1)")[0][0]
        multi_ship = fetch(db, "SELECT COUNT(*) FROM (SELECT order_id FROM shipments GROUP BY order_id HAVING COUNT(*) > 1)")[0][0]
        self.assertGreater(multi_pay, 0)
        self.assertGreater(multi_ship, 0)
        correct = fetch(db, "SELECT SUM(line_total) FROM order_items")[0][0]
        via_payments = fetch(db, "SELECT SUM(i.line_total) FROM orders o JOIN payments p ON p.order_id = o.order_id "
                                 "JOIN order_items i ON i.order_id = o.order_id")[0][0]
        self.assertGreater(via_payments, correct)

    def test_payments_add_up_to_order_totals(self):
        db, _ = demo()
        rows = fetch(db, "SELECT o.order_id, (SELECT SUM(line_total) FROM order_items WHERE order_id = o.order_id), "
                         "(SELECT SUM(amount) FROM payments WHERE order_id = o.order_id) FROM orders o")
        self.assertGreater(len(rows), 10)
        for _oid, billed, paid in rows:
            self.assertAlmostEqual(billed, paid, places=2)

    def test_existing_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "x.sqlite"
            p.write_text("junk")
            make_demo_db(p, seed=1, n_customers=5, n_orders=10)
            self.assertEqual(fetch(p, "SELECT COUNT(*) FROM orders")[0][0], 10)


class LiveHelpers(unittest.TestCase):
    """The live mode needs a network and a key; only its pure helpers are tested."""

    def test_prompt_lists_only_approved_metrics(self):
        model = example_model()
        model.metrics["revenue"].status = "proposed"
        prompt = build_prompt(model, "How are sales?")
        self.assertNotIn("- revenue:", prompt)
        self.assertIn("- order_count:", prompt)
        self.assertIn("Question: How are sales?", prompt)

    def test_reply_parsing(self):
        self.assertEqual(parse_reply('Sure! {"metrics": ["revenue"]} done'), {"metrics": ["revenue"]})
        for bad in ("no json here", "{broken", "[1, 2]"):
            with self.subTest(bad):
                with self.assertRaises(LiveError):
                    parse_reply(bad)


if __name__ == "__main__":
    unittest.main()
