import importlib.util
import json
import pathlib
import subprocess
import sys
import unittest

from sll_testutil import ROOT

BENCH = ROOT / "bench" / "run_bench.py"


def load_bench():
    spec = importlib.util.spec_from_file_location("sll_run_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class QuickBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_bench()
        cls.res = cls.bench.measure(quick=True)

    def test_measure_is_deterministic(self):
        self.assertEqual(self.bench.measure(quick=True), self.res)

    def test_every_question_matches_its_reference_and_uses_approved_joins(self):
        q = self.res["questions"]
        file_count = len(json.loads((BENCH.parent / "questions.json").read_text(encoding="utf-8"))["questions"])
        self.assertEqual(q["asked"], file_count)
        self.assertGreaterEqual(q["asked"], 20)
        failed = [qid for qid, ok, _sql in self.res["detail"] if not ok]
        self.assertEqual(failed, [])
        self.assertEqual(q["matches"], q["asked"])
        self.assertEqual(q["approved_only"], q["asked"])

    def test_inference_is_perfect_on_the_demo_schema(self):
        i = self.res["infer"]
        self.assertEqual(i["true"], 6)
        self.assertEqual(i["correct"], i["proposed"])
        self.assertEqual(i["correct"], i["true"])
        self.assertEqual(i["declared_hits"] + i["inferred_hits"], i["correct"])

    def test_raw_schema_offers_several_routes_that_disagree(self):
        r = self.res["routes"]
        self.assertGreater(r["n_routes"], 1)
        self.assertGreaterEqual(r["n_differ"], 1)
        self.assertLess(r["n_differ"], r["n_routes"])  # the direct route is correct, so not all differ
        agreeing = [x for x in r["routes"] if x["agrees"]]
        for x in agreeing:
            self.assertAlmostEqual(x["total"], r["good_total"], places=2)
        for x in r["routes"]:
            if not x["agrees"]:
                self.assertNotAlmostEqual(x["total"], r["good_total"], places=2)

    def test_simple_paths_helper_on_a_square(self):
        from semantic_layer_lite.model import Join
        def j(a, b):
            return Join(f"{a}-{b}", a, "x", b, "x", "many_to_one", "approved")
        joins = [j("a", "b"), j("b", "d"), j("a", "c"), j("c", "d"), j("a", "d")]
        paths = self.bench.simple_paths(joins, "a", "d", 4)
        self.assertEqual(len(paths), 3)  # a-d, a-b-d, a-c-d
        self.assertEqual(sorted(len(p) for p in paths), [1, 2, 2])
        self.assertEqual(len(self.bench.simple_paths(joins, "a", "d", 1)), 1)  # hop limit applies

    def test_render_formats_numbers_the_same_everywhere(self):
        text = self.bench.render(self.res)
        i = self.res["infer"]
        p, r = self.bench.fmt(i["precision"]), self.bench.fmt(i["recall"])
        self.assertIn(f"| {p} | {r} |", text)  # the table cell
        self.assertIn(f"precision {p} and recall {r}", text)  # the sentence, same formatting
        self.assertTrue(text.rstrip().endswith("."))
        self.assertIn("Measured with python3 bench/run_bench.py, Python ", text)


class BenchProcess(unittest.TestCase):
    def test_quick_run_prints_only_markdown(self):
        proc = subprocess.run([sys.executable, str(BENCH), "--quick"], cwd=str(ROOT), capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout
        self.assertEqual(proc.stderr, "")
        self.assertIn("| Join inference |", out)
        self.assertIn("| Questions |", out)
        self.assertIn("distinct simple join paths", out)
        self.assertTrue(out.strip().splitlines()[-1].startswith("Measured with python3 bench/run_bench.py, Python "))
        self.assertNotIn(str(pathlib.Path(ROOT)), out)
        self.assertNotIn("Traceback", out)


if __name__ == "__main__":
    unittest.main()
