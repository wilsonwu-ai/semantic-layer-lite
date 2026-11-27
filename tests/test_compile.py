import unittest

from sll_testutil import demo, example_dict, example_model, fetch
from semantic_layer_lite.compile import CompileError, compile_query, find_shortest_paths
from semantic_layer_lite.model import model_from_dict
from semantic_layer_lite.run import run_plan

CUSTOMER_JOIN = "orders.customer_id=customers.customer_id"
PRODUCT_JOIN = "order_items.product_id=products.product_id"


def example_with(join_id=None, status=None, metric=None, metric_status=None):
    data = example_dict()
    for j in data["joins"]:
        if j["id"] == join_id:
            j["status"] = status
    for m in data["metrics"]:
        if m["name"] == metric:
            m["status"] = metric_status
    return model_from_dict(data)


def tiny_model(joins, dim_table="b"):
    """Tables a, b, c, d; metric 'total' sums a.amt; dimension 'label' lives on dim_table."""
    cols = {"id": "INTEGER", "label": "TEXT", "amt": "REAL", "x_id": "INTEGER", "y_id": "INTEGER"}
    return model_from_dict({
        "version": 1,
        "entities": {t: {"primary_key": ["id"], "columns": dict(cols)} for t in "abcd"},
        "joins": [
            {"id": f"{ft}.{fc}={tt}.{tc}", "from_table": ft, "from_column": fc, "to_table": tt,
             "to_column": tc, "cardinality": "many_to_one", "status": st}
            for ft, fc, tt, tc, st in joins
        ],
        "metrics": [{"name": "total", "base_table": "a", "aggregation": "sum", "expression": "amt",
                     "status": "approved"}],
        "dimensions": [{"name": "label", "table": dim_table, "column": "label", "kind": "categorical"}],
    })


class Refusals(unittest.TestCase):
    def test_control_compiles_when_everything_is_approved(self):
        plan = compile_query(example_model(), {"metrics": ["revenue"], "dimensions": ["region"]})
        self.assertIn('LEFT JOIN "customers"', plan.sql)

    def test_proposed_join_is_refused(self):
        m = example_with(CUSTOMER_JOIN, "proposed")
        with self.assertRaises(CompileError) as cm:
            compile_query(m, {"metrics": ["revenue"], "dimensions": ["region"]})
        self.assertEqual(cm.exception.code, "join_not_approved")
        self.assertIn(CUSTOMER_JOIN, str(cm.exception))

    def test_rejected_join_is_refused_and_unrelated_queries_still_work(self):
        m = example_with(CUSTOMER_JOIN, "rejected")
        with self.assertRaises(CompileError) as cm:
            compile_query(m, {"metrics": ["revenue"], "dimensions": ["region"]})
        self.assertEqual(cm.exception.code, "join_rejected")
        plan = compile_query(m, {"metrics": ["revenue"], "dimensions": ["category"]})
        self.assertNotIn("customers", plan.sql)

    def test_unapproved_metric_is_refused(self):
        m = example_with(metric="revenue", metric_status="proposed")
        with self.assertRaises(CompileError) as cm:
            compile_query(m, {"metrics": ["revenue"]})
        self.assertEqual(cm.exception.code, "metric_not_approved")

    def test_no_path_at_all(self):
        m = tiny_model([("a", "x_id", "c", "id", "approved")], dim_table="b")
        with self.assertRaises(CompileError) as cm:
            compile_query(m, {"metrics": ["total"], "dimensions": ["label"]})
        self.assertEqual(cm.exception.code, "no_path")

    def test_two_parallel_joins_are_ambiguous(self):
        both = [("a", "x_id", "b", "id", "approved"), ("a", "y_id", "b", "id", "approved")]
        with self.assertRaises(CompileError) as cm:
            compile_query(tiny_model(both), {"metrics": ["total"], "dimensions": ["label"]})
        self.assertEqual(cm.exception.code, "ambiguous_path")
        self.assertIn("a.x_id=b.id", str(cm.exception))
        self.assertIn("a.y_id=b.id", str(cm.exception))
        # rejecting one of the two removes the ambiguity
        one = [("a", "x_id", "b", "id", "approved"), ("a", "y_id", "b", "id", "rejected")]
        plan = compile_query(tiny_model(one), {"metrics": ["total"], "dimensions": ["label"]})
        self.assertEqual(plan.join_ids(), ["a.x_id=b.id"])

    def test_diamond_is_ambiguous_but_a_shorter_route_wins(self):
        diamond = [("a", "x_id", "b", "id", "approved"), ("a", "y_id", "c", "id", "approved"),
                   ("b", "x_id", "d", "id", "approved"), ("c", "x_id", "d", "id", "approved")]
        with self.assertRaises(CompileError) as cm:
            compile_query(tiny_model(diamond, dim_table="d"), {"metrics": ["total"], "dimensions": ["label"]})
        self.assertEqual(cm.exception.code, "ambiguous_path")
        shortcut = diamond + [("a", "id", "d", "id", "approved")]
        plan = compile_query(tiny_model(shortcut, dim_table="d"), {"metrics": ["total"], "dimensions": ["label"]})
        self.assertEqual(plan.join_ids(), ["a.id=d.id"])

    def test_fan_out_is_refused_for_additive_metrics(self):
        m = example_model()
        for dim in ("carrier", "method"):
            with self.subTest(dim):
                with self.assertRaises(CompileError) as cm:
                    compile_query(m, {"metrics": ["revenue"], "dimensions": [dim]})
                self.assertEqual(cm.exception.code, "fan_out")
        with self.assertRaises(CompileError) as cm:  # a filter can fan out too
            compile_query(m, {"metrics": ["units_sold"], "filters": [
                {"dimension": "carrier", "op": "=", "value": "UPS"}]})
        self.assertEqual(cm.exception.code, "fan_out")

    def test_fan_out_is_allowed_for_distinct_counts_and_for_many_to_one_hops(self):
        m = example_model()
        plan = compile_query(m, {"metrics": ["order_count"], "dimensions": ["carrier"]})
        self.assertTrue(any(s.fan_out for s in plan.steps))
        plan = compile_query(m, {"metrics": ["revenue"], "dimensions": ["region", "category", "month"]})
        self.assertFalse(any(s.fan_out for s in plan.steps))

    def test_bad_queries(self):
        m = example_model()
        bad = [
            ({"metrics": []}, "bad_query"),
            ({"metrics": ["nope"]}, "unknown_metric"),
            ({"metrics": ["revenue"], "dimensions": ["nope"]}, "unknown_dimension"),
            ({"metrics": ["revenue"], "dimensions": ["region; DROP TABLE orders"]}, "unknown_dimension"),
            ({"metrics": ["revenue; DROP TABLE orders"]}, "unknown_metric"),
            ({"metrics": ["revenue"], "dimensions": ["order_date:week"]}, "bad_grain"),
            ({"metrics": ["revenue"], "dimensions": ["region:month"]}, "bad_grain"),
            ({"metrics": ["avg_list_price"], "dimensions": ["month"]}, "bad_grain"),
            ({"metrics": ["revenue", "cash_collected"]}, "mixed_base"),
            ({"metrics": ["revenue", "revenue"]}, "bad_query"),
            ({"metrics": ["revenue"], "limit": 0}, "bad_limit"),
            ({"metrics": ["revenue"], "limit": 10 ** 9}, "bad_limit"),
            ({"metrics": ["revenue"], "limit": "5"}, "bad_limit"),
            ({"metrics": ["revenue"], "extra": 1}, "bad_query"),
            ({"metrics": ["revenue"], "filters": [{"dimension": "region", "op": "= 1 OR 1=1 --", "value": "x"}]},
             "bad_filter"),
            ({"metrics": ["revenue"], "filters": [{"dimension": "region", "op": "="}]}, "bad_filter"),
            ({"metrics": ["revenue"], "filters": [{"dimension": "ghost", "op": "=", "value": 1}]},
             "unknown_dimension"),
            ({"metrics": ["revenue"], "filters": [{"dimension": "region", "op": "in", "value": []}]},
             "bad_filter_value"),
            ({"metrics": ["revenue"], "filters": [{"dimension": "region", "op": "=", "value": ["a"]}]},
             "bad_filter_value"),
        ]
        for query, code in bad:
            with self.subTest(query):
                with self.assertRaises(CompileError) as cm:
                    compile_query(m, query)
                self.assertEqual(cm.exception.code, code)


class Safety(unittest.TestCase):
    def test_filter_value_stays_a_bound_parameter(self):
        db, _ = demo()
        before = fetch(db, "SELECT COUNT(*) FROM orders")[0][0]
        evil = "x'; DROP TABLE orders; --"
        plan = compile_query(example_model(), {"metrics": ["revenue"], "dimensions": ["region"],
                                               "filters": [{"dimension": "region", "op": "=", "value": evil}]})
        self.assertNotIn("DROP", plan.sql)
        self.assertNotIn("x'", plan.sql)
        self.assertEqual(plan.params, [evil])
        self.assertEqual(plan.sql.count("?"), 1)
        result = run_plan(db, plan)
        self.assertEqual(result.rows, [])  # no region has that name
        self.assertEqual(fetch(db, "SELECT COUNT(*) FROM orders")[0][0], before)
        self.assertEqual(fetch(db, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'orders'")[0][0], 1)

    def test_in_filter_binds_every_value(self):
        plan = compile_query(example_model(), {"metrics": ["revenue"], "filters": [
            {"dimension": "region", "op": "in", "value": ["East", "'; --", "West"]}]})
        self.assertEqual(plan.sql.count("?"), 3)
        self.assertEqual(plan.params, ["East", "'; --", "West"])
        self.assertNotIn("--", plan.sql)

    def test_row_limit_always_applies(self):
        m = example_model()
        plan = compile_query(m, {"metrics": ["revenue"]})
        self.assertTrue(plan.sql.rstrip().endswith("LIMIT 1000"))
        plan = compile_query(m, {"metrics": ["revenue"], "dimensions": ["day"], "limit": 3})
        self.assertTrue(plan.sql.rstrip().endswith("LIMIT 3"))
        db, _ = demo()
        rows = run_plan(db, plan).rows
        self.assertEqual(len(rows), 3)
        distinct_days = fetch(db, "SELECT COUNT(DISTINCT substr(order_date, 1, 10)) FROM orders")[0][0]
        self.assertGreater(distinct_days, 3)  # the limit really cut something off

    def test_numeric_text_is_coerced_for_integer_columns(self):
        m = model_from_dict({
            "version": 1,
            "entities": {"t": {"primary_key": ["id"], "columns": {"id": "INTEGER", "n": "INTEGER", "amt": "REAL"}}},
            "joins": [],
            "metrics": [{"name": "total", "base_table": "t", "aggregation": "sum", "expression": "amt * 2",
                         "status": "approved"}],
            "dimensions": [{"name": "n", "table": "t", "column": "n", "kind": "categorical"}],
        })
        plan = compile_query(m, {"metrics": ["total"], "filters": [{"dimension": "n", "op": ">=", "value": "5"}]})
        self.assertEqual(plan.params, [5])
        self.assertIn('"t"."amt" * 2', plan.sql)
        with self.assertRaises(CompileError):
            compile_query(m, {"metrics": ["total"], "filters": [{"dimension": "n", "op": "=", "value": "five"}]})


class TimeGrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, _ = demo()
        cls.model = example_model()
        cls.rows = fetch(cls.db, "SELECT o.order_date, i.line_total FROM order_items i "
                                 "JOIN orders o ON o.order_id = i.order_id")

    def expected(self, width):
        out = {}
        for when, amount in self.rows:
            out[when[:width]] = out.get(when[:width], 0.0) + amount
        return out

    def check(self, grain, width):
        plan = compile_query(self.model, {"metrics": ["revenue"], "dimensions": [grain]})
        result = run_plan(self.db, plan)
        got = dict(result.rows)
        want = self.expected(width)
        self.assertEqual(sorted(got), sorted(want))
        for k in want:
            self.assertAlmostEqual(got[k], want[k], places=6)
        self.assertEqual([r[0] for r in result.rows], sorted(got))  # ordered by the group key
        return got

    def test_month(self):
        got = self.check("month", 7)
        self.assertGreaterEqual(len(got), 5)

    def test_day(self):
        got = self.check("day", 10)
        self.assertGreater(len(got), len(self.expected(7)))

    def test_year(self):
        got = self.check("year", 4)
        self.assertEqual(list(got), ["2026"])

    def test_shorthand_equals_explicit_grain_and_default_is_day(self):
        a = compile_query(self.model, {"metrics": ["revenue"], "dimensions": ["month"]})
        b = compile_query(self.model, {"metrics": ["revenue"], "dimensions": ["order_date:month"]})
        self.assertEqual((a.sql, a.params), (b.sql, b.params))
        c = compile_query(self.model, {"metrics": ["revenue"], "dimensions": ["order_date"]})
        d = compile_query(self.model, {"metrics": ["revenue"], "dimensions": ["day"]})
        self.assertEqual(c.sql, d.sql)


class ShortestPaths(unittest.TestCase):
    def test_finds_all_shortest_and_none_when_disconnected(self):
        m = example_model()
        paths = find_shortest_paths(m.joins, "order_items", "customers")
        self.assertEqual(len(paths), 1)
        self.assertEqual([s.to_table for s in paths[0]], ["orders", "customers"])
        self.assertEqual(find_shortest_paths([], "orders", "customers"), [])
        self.assertEqual(find_shortest_paths(m.joins, "orders", "orders"), [[]])

    def test_fan_out_flag_follows_direction(self):
        m = example_model()
        down = find_shortest_paths(m.joins, "orders", "shipments")[0][0]   # one order, many shipments
        up = find_shortest_paths(m.joins, "shipments", "orders")[0][0]     # many shipments, one order
        self.assertTrue(down.fan_out)
        self.assertFalse(up.fan_out)


if __name__ == "__main__":
    unittest.main()
