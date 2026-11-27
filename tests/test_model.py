import copy
import json
import pathlib
import tempfile
import unittest

from sll_testutil import EXAMPLE, demo, example_dict
from semantic_layer_lite.infer import infer_model
from semantic_layer_lite.model import (ModelError, load_model, model_from_dict, model_to_dict,
                                       render_expression, save_model, set_status, tokenize_expression)


class RoundTrip(unittest.TestCase):
    def test_example_loads_and_round_trips(self):
        data = example_dict()
        model = model_from_dict(data)
        again = model_to_dict(model)
        self.assertEqual(model_from_dict(again).joins, model.joins)
        self.assertEqual(len(model.joins), len(data["joins"]))
        self.assertEqual(sorted(model.metrics), sorted(m["name"] for m in data["metrics"]))

    def test_save_and_load_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "m.json"
            model = load_model(EXAMPLE)
            save_model(model, p)
            self.assertEqual(model_to_dict(load_model(p)), model_to_dict(model))

    def test_invalid_json_is_a_model_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "m.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ModelError):
                load_model(p)


class Validation(unittest.TestCase):
    def check_bad(self, mutate, label):
        data = copy.deepcopy(example_dict())
        mutate(data)
        with self.assertRaises(ModelError, msg=label):
            model_from_dict(data)

    def test_each_rule_can_fail(self):
        def bad_status(d): d["joins"][0]["status"] = "maybe"
        def unknown_table(d): d["joins"][0]["to_table"] = "ghosts"
        def unknown_column(d): d["joins"][0]["from_column"] = "nope"
        def dup_join(d): d["joins"][1]["id"] = d["joins"][0]["id"]
        def bad_cardinality(d): d["joins"][0]["cardinality"] = "many_to_many"
        def bad_agg(d): d["metrics"][0]["aggregation"] = "median"
        def bad_base(d): d["metrics"][0]["base_table"] = "ghosts"
        def injected_expr(d): d["metrics"][0]["expression"] = "unit_price); DROP TABLE orders; --"
        def unknown_col_expr(d): d["metrics"][0]["expression"] = "secret_column"
        def unbalanced(d): d["metrics"][0]["expression"] = "(unit_price"
        def two_values(d): d["metrics"][0]["expression"] = "unit_price unit_price"
        def bad_time_dim(d): d["metrics"][0]["time_dimension"] = "region"
        def bad_kind(d): d["dimensions"][0]["kind"] = "numeric"
        def dup_metric(d): d["metrics"][1]["name"] = d["metrics"][0]["name"]
        def bad_ident(d): d["metrics"][0]["name"] = "bad name"
        def no_joins(d): del d["joins"]
        def bad_version(d): d["version"] = 99
        cases = locals().copy()
        cases.pop("self")
        cases.pop("data", None)
        funcs = {k: v for k, v in cases.items() if callable(v)}
        self.assertGreaterEqual(len(funcs), 15)
        for label, fn in sorted(funcs.items()):
            with self.subTest(label):
                self.check_bad(fn, label)

    def test_metric_order_in_example_is_what_this_test_assumes(self):
        # the mutations above index into the first metric; make sure it is avg_list_price on products
        self.assertEqual(example_dict()["metrics"][0]["base_table"], "products")


class Expressions(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize_expression("quantity * 2.5"),
                         [("id", "quantity"), ("op", "*"), ("num", "2.5")])

    def test_render_quotes_columns(self):
        sql = render_expression("quantity * unit_price", "order_items", {"quantity", "unit_price"})
        self.assertEqual(sql, '"order_items"."quantity" * "order_items"."unit_price"')

    def test_semicolons_and_functions_are_rejected(self):
        for bad in ("quantity; DROP", "load_extension('x')", "quantity -- c", "'a'"):
            with self.subTest(bad):
                with self.assertRaises(ModelError):
                    render_expression(bad, "t", {"quantity"})


class Status(unittest.TestCase):
    def test_set_status_changes_only_what_it_should(self):
        model = model_from_dict(example_dict())
        jid = model.joins[0].id
        self.assertEqual(set_status(model, "rejected", join_ids=[jid]), 1)
        self.assertEqual(model.joins[0].status, "rejected")
        self.assertTrue(all(j.status == "approved" for j in model.joins[1:]))
        self.assertEqual(set_status(model, "rejected", join_ids=[jid]), 0)  # already rejected
        with self.assertRaises(ModelError):
            set_status(model, "approved", join_ids=["no.such=join.id"])

    def test_approve_all_skips_rejected(self):
        db, _ = demo()
        model, _ = infer_model(db)
        victim = model.joins[0].id
        set_status(model, "rejected", join_ids=[victim])
        changed = set_status(model, "approved", all_proposed=True)
        self.assertEqual(changed, len(model.joins) - 1 + len(model.metrics))
        by_id = {j.id: j for j in model.joins}
        self.assertEqual(by_id[victim].status, "rejected")

    def test_striking_out_a_join_is_a_one_word_edit(self):
        db, _ = demo()
        model, _ = infer_model(db)
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "semantic.json"
            save_model(model, p)
            text = p.read_text(encoding="utf-8")
            self.assertEqual(text.count('"status": "proposed"'), len(model.joins) + len(model.metrics))
            p.write_text(text.replace('"status": "proposed"', '"status": "rejected"', 1), encoding="utf-8")
            edited = load_model(p)
        self.assertEqual(sum(1 for j in edited.joins if j.status == "rejected"), 1)
        self.assertEqual(sum(1 for j in edited.joins if j.status == "proposed"), len(model.joins) - 1)
        data = json.loads(text)
        self.assertEqual(data["joins"][0]["status"], "proposed")  # the first status in the file is a join


if __name__ == "__main__":
    unittest.main()
