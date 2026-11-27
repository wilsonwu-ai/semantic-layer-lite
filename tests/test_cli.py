import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from sll_testutil import EXAMPLE, demo, fetch
from semantic_layer_lite.cli import main, parse_filter
from semantic_layer_lite.compile import CompileError
from semantic_layer_lite.model import load_model

SUBCOMMANDS = ("infer", "approve", "reject", "compile", "query", "explain", "demo-db", "ask")


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class Help(unittest.TestCase):
    def test_help_lists_every_subcommand(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as cm:
                main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        for name in SUBCOMMANDS:
            self.assertIn(name, out.getvalue())

    def test_no_subcommand_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                main([])
        self.assertEqual(cm.exception.code, 2)


class ParseFilter(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(parse_filter("region=West"), {"dimension": "region", "op": "=", "value": "West"})
        self.assertEqual(parse_filter("order_date>=2026-03-01"),
                         {"dimension": "order_date", "op": ">=", "value": "2026-03-01"})
        self.assertEqual(parse_filter("region != East"), {"dimension": "region", "op": "!=", "value": "East"})
        self.assertEqual(parse_filter("region in East, West"),
                         {"dimension": "region", "op": "in", "value": ["East", "West"]})
        self.assertEqual(parse_filter("name=a=b")["value"], "a=b")

    def test_garbage_is_refused(self):
        with self.assertRaises(CompileError):
            parse_filter("just words")


class Workflow(unittest.TestCase):
    def test_infer_approve_reject_compile_query_explain(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            db, sem = tmp / "d.sqlite", tmp / "semantic.json"

            code, out, _ = run_cli("demo-db", str(db), "--seed", "5")
            self.assertEqual(code, 0)
            self.assertTrue(db.exists())
            self.assertIn("orders: 400 rows", out)
            code, _, err = run_cli("demo-db", str(db))
            self.assertEqual(code, 2)  # refuses to overwrite
            self.assertIn("--force", err)
            self.assertEqual(run_cli("demo-db", str(db), "--force")[0], 0)

            code, out, _ = run_cli("infer", str(db), "-o", str(sem))
            self.assertEqual(code, 0)
            self.assertIn("order_items.product_id=products.product_id", out)  # the person reads the proposals
            model = load_model(sem)
            self.assertTrue(all(j.status == "proposed" for j in model.joins))

            # nothing proposed can be compiled yet
            code, _, err = run_cli("compile", str(sem), "--metric", "total_line_total", "--by", "month")
            self.assertEqual(code, 3)
            self.assertIn("metric_not_approved", err)

            self.assertEqual(run_cli("approve", str(sem))[0], 2)  # needs a flag
            code, out, _ = run_cli("approve", str(sem), "--all")
            self.assertEqual(code, 0)
            model = load_model(sem)
            self.assertTrue(all(j.status == "approved" for j in model.joins))
            self.assertTrue(all(m.status == "approved" for m in model.metrics.values()))

            code, out, _ = run_cli("compile", str(sem), "--metric", "total_line_total", "--by", "month")
            self.assertEqual(code, 0)
            self.assertIn("SELECT", out)
            self.assertIn("LIMIT 1000", out)
            self.assertIn('"order_items"."line_total"', out)

            # strike out the products join: the question that needs it is now refused
            bad = "order_items.product_id=products.product_id"
            code, out, _ = run_cli("reject", str(sem), "--join", bad)
            self.assertEqual(code, 0)
            self.assertIn("rejected 1", out)
            code, _, err = run_cli("compile", str(sem), "--metric", "total_line_total", "--by", "category")
            self.assertEqual(code, 3)
            self.assertIn("join_rejected", err)
            # a question that does not need it still compiles
            self.assertEqual(run_cli("compile", str(sem), "--metric", "total_line_total", "--by", "region")[0], 0)
            self.assertEqual(run_cli("reject", str(sem))[0], 2)
            self.assertEqual(run_cli("reject", str(sem), "--join", "no.such=join.id")[0], 2)
            # approving it again by id brings it back
            self.assertEqual(run_cli("approve", str(sem), "--join", bad)[0], 0)
            self.assertEqual(run_cli("compile", str(sem), "--metric", "total_line_total", "--by", "category")[0], 0)

    def test_rejecting_the_direct_join_does_not_route_around_it_through_a_fan_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            db, sem = tmp / "d.sqlite", tmp / "semantic.json"
            run_cli("demo-db", str(db))
            run_cli("infer", str(db), "-o", str(sem))
            run_cli("approve", str(sem), "--all")
            run_cli("reject", str(sem), "--join", "order_items.order_id=orders.order_id")
            code, _, err = run_cli("compile", str(sem), "--metric", "total_line_total", "--by", "month")
            self.assertEqual(code, 3)
            self.assertIn("fan_out", err)

    def test_query_prints_rows_and_the_explanation(self):
        db, _ = demo()
        code, out, _ = run_cli("query", str(db), str(EXAMPLE), "--metric", "revenue", "--by", "region")
        self.assertEqual(code, 0)
        regions = [r[0] for r in fetch(db, "SELECT DISTINCT region FROM customers")]
        self.assertEqual(len(regions), 4)
        for r in regions:
            self.assertIn(r, out)
        self.assertIn("order_items.order_id=orders.order_id", out)
        self.assertIn("orders.customer_id=customers.customer_id", out)
        self.assertIn("SUM(line_total) over order_items", out)

    def test_query_json_carries_audit_fields(self):
        db, _ = demo()
        code, out, _ = run_cli("query", str(db), str(EXAMPLE), "--metric", "revenue", "--by", "category",
                               "--filter", "region in East,West", "--limit", "2", "--json")
        self.assertEqual(code, 0)
        records = json.loads(out)
        self.assertEqual(len(records), 2)
        for rec in records:
            self.assertIn("order_items.product_id=products.product_id", rec["_join_ids"])
            self.assertIn("revenue", rec["_metrics"])

    def test_explain_and_compile_json(self):
        code, out, _ = run_cli("explain", str(EXAMPLE), "--metric", "order_count", "--by", "carrier")
        self.assertEqual(code, 0)
        self.assertIn("shipments.order_id=orders.order_id", out)
        self.assertIn("fans out", out)
        code, out, _ = run_cli("compile", str(EXAMPLE), "--metric", "revenue", "--by", "month",
                               "--filter", "region=West", "--json")
        self.assertEqual(code, 0)
        info = json.loads(out)
        self.assertEqual(info["params"], ["West"])
        self.assertEqual(info["limit"], 1000)

    def test_refusals_exit_three_and_file_errors_exit_two(self):
        code, _, err = run_cli("compile", str(EXAMPLE), "--metric", "revenue", "--by", "carrier")
        self.assertEqual(code, 3)
        self.assertIn("fan_out", err)
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(pathlib.Path(tmp) / "none.json")
            self.assertEqual(run_cli("compile", missing, "--metric", "revenue")[0], 2)
            out_path = pathlib.Path(tmp) / "out.json"
            code, _, err = run_cli("infer", str(pathlib.Path(tmp) / "none.sqlite"), "-o", str(out_path))
            self.assertEqual(code, 2)
            self.assertIn("cannot open", err)
            self.assertFalse(out_path.exists())  # nothing is written when inference fails

    def test_ask_needs_the_live_flag(self):
        db, _ = demo()
        code, _, err = run_cli("ask", str(db), str(EXAMPLE), "how are sales?")
        self.assertEqual(code, 2)
        self.assertIn("--live", err)
        code, _, err = run_cli("ask", str(db), str(EXAMPLE), "how are sales?", "--live")
        self.assertEqual(code, 2)  # --live without an endpoint stops before any network call
        self.assertIn("--endpoint", err)


if __name__ == "__main__":
    unittest.main()
