import pathlib
import sqlite3
import tempfile
import unittest

from sll_testutil import demo, example_dict
from semantic_layer_lite.infer import InferError, infer_model, singular
from semantic_layer_lite.model import model_to_dict


def key(j):
    return (j.from_table, j.from_column, j.to_table, j.to_column)


def build(path, script):
    conn = sqlite3.connect(str(path))
    conn.executescript(script)
    conn.commit()
    conn.close()


class DemoInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, cls.info = demo()
        cls.model, cls.notes = infer_model(cls.db)

    def test_every_declared_foreign_key_is_found(self):
        conn = sqlite3.connect(str(self.db))
        declared = set()
        for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            for row in conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall():
                declared.add((t, row[3], row[2], row[4]))
        conn.close()
        self.assertEqual(len(declared), 2)  # the demo declares exactly two
        by_key = {key(j): j for j in self.model.joins}
        for k in declared:
            self.assertIn(k, by_key)
            self.assertEqual(by_key[k].evidence["source"], "declared_foreign_key")

    def test_undeclared_joins_are_found_by_name_uniqueness_and_containment(self):
        true = {(j["from_table"], j["from_column"], j["to_table"], j["to_column"]) for j in self.info["joins"]}
        found = {key(j) for j in self.model.joins}
        self.assertEqual(len(true), 6)
        self.assertEqual(found, true)
        inferred = [j for j in self.model.joins if j.evidence["source"] == "inferred"]
        self.assertEqual(len(inferred), 4)
        for j in inferred:
            self.assertEqual(j.evidence["values_checked"], j.evidence["values_contained"])
            self.assertTrue(j.evidence["target_is_unique"])

    def test_row_id_columns_take_part_in_no_proposed_join_except_real_keys(self):
        # payment_id and shipment_id are row ids of their own tables; no join may use them.
        cols = {c for j in self.model.joins for c in ((j.from_table, j.from_column), (j.to_table, j.to_column))}
        self.assertNotIn(("payments", "payment_id"), cols)
        self.assertNotIn(("shipments", "shipment_id"), cols)

    def test_cardinality_and_status(self):
        for j in self.model.joins:
            self.assertEqual(j.status, "proposed")
            self.assertEqual(j.cardinality, "many_to_one")
        for m in self.model.metrics.values():
            self.assertEqual(m.status, "proposed")

    def test_proposed_dimensions_match_the_reviewed_example(self):
        ex = example_dict()
        want = {d["name"]: (d["table"], d["column"], d["kind"]) for d in ex["dimensions"]}
        got = {d.name: (d.table, d.column, d.kind) for d in self.model.dimensions.values()}
        self.assertEqual(got, want)
        self.assertNotIn("name", got)  # high-cardinality text is not a dimension
        self.assertNotIn("customer_id", got)

    def test_proposed_measures_and_entities(self):
        m = self.model.metrics
        self.assertEqual(m["total_line_total"].base_table, "order_items")
        self.assertEqual(m["total_line_total"].aggregation, "sum")
        self.assertEqual(m["total_line_total"].time_dimension, "order_date")
        self.assertEqual(m["orders_count"].expression, "*")
        for name, metric in m.items():
            self.assertFalse(metric.expression.endswith("_id"), name)
        ex = example_dict()
        self.assertEqual(model_to_dict(self.model)["entities"], ex["entities"])

    def test_deterministic(self):
        again, _ = infer_model(self.db)
        self.assertEqual(model_to_dict(again), model_to_dict(self.model))

    def test_database_is_not_modified(self):
        before = pathlib.Path(self.db).read_bytes()
        infer_model(self.db)
        self.assertEqual(pathlib.Path(self.db).read_bytes(), before)
        self.assertFalse(pathlib.Path(str(self.db) + "-journal").exists())


class UnrelatedIds(unittest.TestCase):
    def test_no_join_between_unrelated_integer_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = pathlib.Path(tmp) / "t.sqlite"
            build(db, """
                CREATE TABLE tickets (ticket_id INTEGER PRIMARY KEY, legacy_ref INTEGER);
                CREATE TABLE invoices (invoice_id INTEGER PRIMARY KEY, memo TEXT);
                CREATE TABLE lines (line_id INTEGER PRIMARY KEY, invoice_id INTEGER);
                INSERT INTO invoices VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e'),(6,'f');
                INSERT INTO tickets VALUES (1,2),(2,3),(3,5),(4,1);
                INSERT INTO lines VALUES (1,1),(2,1),(3,4),(4,6);
            """)
            model, _ = infer_model(db)
            keys = {key(j) for j in model.joins}
            # positive control: a real, name-matched, contained join is proposed
            self.assertIn(("lines", "invoice_id", "invoices", "invoice_id"), keys)
            # numeric overlap alone is not evidence
            self.assertEqual(keys, {("lines", "invoice_id", "invoices", "invoice_id")})

    def test_name_match_without_containment_is_not_proposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = pathlib.Path(tmp) / "t.sqlite"
            build(db, """
                CREATE TABLE orders (order_id INTEGER PRIMARY KEY);
                CREATE TABLE notes (note_id INTEGER PRIMARY KEY, order_id INTEGER);
                CREATE TABLE refunds (refund_id INTEGER PRIMARY KEY, order_id INTEGER);
                INSERT INTO orders VALUES (1),(2),(3);
                INSERT INTO notes VALUES (1,1),(2,2),(3,999);
                INSERT INTO refunds VALUES (1,1),(2,3);
            """)
            model, _ = infer_model(db)
            keys = {key(j) for j in model.joins}
            self.assertIn(("refunds", "order_id", "orders", "order_id"), keys)
            self.assertNotIn(("notes", "order_id", "orders", "order_id"), keys)  # 999 does not exist

    def test_unique_foreign_key_is_one_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = pathlib.Path(tmp) / "t.sqlite"
            build(db, """
                CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT);
                CREATE TABLE profiles (profile_id INTEGER PRIMARY KEY, customer_id INTEGER UNIQUE);
                INSERT INTO customers VALUES (1,'a'),(2,'b'),(3,'c');
                INSERT INTO profiles VALUES (1,3),(2,1);
            """)
            model, _ = infer_model(db)
            self.assertEqual(len(model.joins), 1)
            self.assertEqual(key(model.joins[0]), ("profiles", "customer_id", "customers", "customer_id"))
            self.assertEqual(model.joins[0].cardinality, "one_to_one")

    def test_plural_table_names_use_singular_id(self):
        self.assertEqual(singular("categories"), "category")
        self.assertEqual(singular("orders"), "order")
        self.assertEqual(singular("address"), "address")


class OpenErrors(unittest.TestCase):
    def test_missing_file_raises_and_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = pathlib.Path(tmp) / "nope.sqlite"
            with self.assertRaises(InferError):
                infer_model(missing)
            self.assertFalse(missing.exists())

    def test_file_that_is_not_a_database_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = pathlib.Path(tmp) / "bad.sqlite"
            bad.write_bytes(b"this is not a sqlite file at all, just some text" * 20)
            with self.assertRaises(InferError):
                infer_model(bad)


if __name__ == "__main__":
    unittest.main()
