"""Bench: join inference quality, compiled SQL vs reference SQL, and the route problem.

Run from the repo root:  python3 bench/run_bench.py   (add --quick for a smaller database)
Prints only a markdown block. Everything is seeded; nothing in the output is a timing.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # noqa: E401,E702

import argparse
import json
import platform
import tempfile
from typing import Any

from semantic_layer_lite.compile import compile_query
from semantic_layer_lite.datasets import make_demo_db
from semantic_layer_lite.infer import infer_model
from semantic_layer_lite.model import Join, load_model, make_join_id
from semantic_layer_lite.run import run_plan, run_sql, same_rows

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 7
MAX_HOPS = 4
REVENUE_TABLE = "order_items"  # the table that holds the revenue column (line_total)


def fmt(x: float) -> str:
    """Format a ratio the same way everywhere it is printed."""
    return f"{x:.3f}"


def simple_paths(joins: list[Join], start: str, goal: str, max_hops: int) -> list[list[tuple[Join, str]]]:
    """Every simple path (no table twice) from start to goal using at most max_hops joins."""
    out: list[list[tuple[Join, str]]] = []

    def walk(table: str, seen: frozenset, path: list[tuple[Join, str]]) -> None:
        if table == goal:
            out.append(list(path))
            return
        if len(path) >= max_hops:
            return
        for j in sorted(joins, key=lambda x: x.id):
            if j.from_table == table:
                nxt = j.to_table
            elif j.to_table == table:
                nxt = j.from_table
            else:
                continue
            if nxt in seen:
                continue
            walk(nxt, seen | {nxt}, path + [(j, nxt)])

    walk(start, frozenset([start]), [])
    return out


def naive_monthly_revenue_sql(path: list[tuple[Join, str]]) -> str:
    """What an agent might write: join along one route, sum the revenue column, group by month."""
    parts = ['SELECT substr("orders"."order_date", 1, 7), SUM("order_items"."line_total") FROM "orders"']
    for j, nxt in path:
        parts.append(f'JOIN "{nxt}" ON "{j.from_table}"."{j.from_column}" = "{j.to_table}"."{j.to_column}"')
    parts.append("GROUP BY 1 ORDER BY 1")
    return " ".join(parts)


def measure(quick: bool = False) -> dict[str, Any]:
    """Run every measurement and return the numbers."""
    n_customers, n_orders = (40, 150) if quick else (80, 400)
    with tempfile.TemporaryDirectory() as tmp:
        db = pathlib.Path(tmp) / "demo.sqlite"
        info = make_demo_db(db, seed=SEED, n_customers=n_customers, n_orders=n_orders)

        # 1. join inference against the generator's true joins
        proposed, _notes = infer_model(db)
        found = {(j.from_table, j.from_column, j.to_table, j.to_column) for j in proposed.joins}
        true = {(j["from_table"], j["from_column"], j["to_table"], j["to_column"]) for j in info["joins"]}
        correct = found & true
        declared_hits = sum(1 for j in proposed.joins
                            if (j.from_table, j.from_column, j.to_table, j.to_column) in true
                            and j.evidence.get("source") == "declared_foreign_key")
        infer_res = {
            "true": len(true), "proposed": len(found), "correct": len(correct),
            "precision": len(correct) / len(found) if found else 0.0,
            "recall": len(correct) / len(true),
            "declared_hits": declared_hits, "inferred_hits": len(correct) - declared_hits,
        }

        # 2. twenty questions: compiled SQL vs hand-written reference SQL
        model = load_model(ROOT / "examples" / "semantic.json")
        questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))["questions"]
        matches = approved_only = 0
        detail = []
        for q in questions:
            plan = compile_query(model, q["query"])
            result = run_plan(db, plan)  # a fresh read-only connection per question
            _cols, ref_rows = run_sql(db, q["reference_sql"], q["reference_params"])
            ok = same_rows(result.rows, ref_rows)
            matches += ok
            approved_only += all(s.join.status == "approved" for s in plan.steps)
            detail.append((q["id"], ok, " ".join(plan.sql.split())))
        q_res = {"asked": len(questions), "matches": matches, "approved_only": approved_only}

        # 3. revenue by month: how many routes does the raw schema offer, and do they agree?
        raw_joins = [Join(make_join_id(j["from_table"], j["from_column"], j["to_table"], j["to_column"]),
                          j["from_table"], j["from_column"], j["to_table"], j["to_column"],
                          j["cardinality"], "approved") for j in info["joins"]]
        good_rows = run_plan(db, compile_query(model, {"metrics": ["revenue"], "dimensions": ["month"]})).rows
        good_total = sum(r[1] for r in good_rows)
        routes = []
        for path in simple_paths(raw_joins, "orders", REVENUE_TABLE, MAX_HOPS):
            _c, rows = run_sql(db, naive_monthly_revenue_sql(path))
            routes.append({
                "route": " -> ".join(["orders"] + [nxt for _, nxt in path]),
                "total": sum(r[1] for r in rows),
                "agrees": same_rows(rows, good_rows),
            })
        route_res = {
            "good_total": good_total, "routes": routes, "n_routes": len(routes),
            "n_differ": sum(1 for r in routes if not r["agrees"]),
        }
    return {"quick": quick, "infer": infer_res, "questions": q_res, "detail": detail, "routes": route_res}


def render(res: dict[str, Any], verbose: bool = False) -> str:
    """Build the markdown block from measured values."""
    i, q, r = res["infer"], res["questions"], res["routes"]
    lines = [
        "| Join inference | True joins | Proposed | Correct | Precision | Recall | Declared found | Inferred found |",
        "|---|---|---|---|---|---|---|---|",
        f"| demo database | {i['true']} | {i['proposed']} | {i['correct']} | {fmt(i['precision'])} "
        f"| {fmt(i['recall'])} | {i['declared_hits']} | {i['inferred_hits']} |",
        "",
        "| Questions | Compiled and run | Exact match with reference SQL | Used approved joins only |",
        "|---|---|---|---|",
        f"| {q['asked']} | {q['asked']} | {q['matches']} | {q['approved_only']} |",
        "",
        "| Route from orders to the revenue column | Grand total | Same monthly answer as the approved route |",
        "|---|---|---|",
        f"| approved route (compiled) | {r['good_total']:.2f} | yes |",
    ]
    for row in r["routes"]:
        lines.append(f"| {row['route']} | {row['total']:.2f} | {'yes' if row['agrees'] else 'no'} |")
    lines += [
        "",
        f"The raw schema offers {r['n_routes']} distinct simple join paths of up to {MAX_HOPS} hops from orders "
        f"to the revenue column, and {r['n_differ']} of them {'gives' if r['n_differ'] == 1 else 'give'} a different monthly revenue than the approved route.",
        f"Join inference proposed {i['proposed']} joins against the generator's {i['true']} true joins, "
        f"with precision {fmt(i['precision'])} and recall {fmt(i['recall'])}.",
        f"Compiled SQL matched hand-written reference SQL on {q['matches']} of {q['asked']} questions, "
        f"and {q['approved_only']} of {q['asked']} used approved joins only.",
    ]
    if verbose:
        lines += ["", "| Question | Match | Compiled SQL |", "|---|---|---|"]
        lines += [f"| {qid} | {'yes' if ok else 'no'} | `{sql}` |" for qid, ok, sql in res["detail"]]
    lines += ["", f"Measured with python3 bench/run_bench.py, Python {platform.python_version()}, {platform.machine()}."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse flags, measure, print the markdown block."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smaller database")
    ap.add_argument("--verbose", action="store_true", help="also print the compiled SQL beside each question")
    args = ap.parse_args(argv)
    print(render(measure(quick=args.quick), verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
