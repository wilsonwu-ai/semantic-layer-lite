"""Command line: infer, approve, reject, compile, query, explain, demo-db, ask."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .compile import CompileError, compile_query
from .datasets import make_demo_db
from .explain import explain_plan, render_plan, render_result, result_records
from .infer import InferError, infer_model
from .live import LiveError, ask_model
from .model import ModelError, load_model, save_model, set_status
from .run import run_plan

_FILTER = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(!=|<=|>=|=|<|>|\s+in\s+)\s*(.*?)\s*$", re.IGNORECASE)


def parse_filter(text: str) -> dict[str, Any]:
    """Parse 'region=West', 'order_date>=2026-03-01' or 'region in East,West' into a filter dict."""
    m = _FILTER.match(text)
    if not m:
        raise CompileError("bad_filter", f"cannot read filter {text!r}; use name=value, name>=value or 'name in a,b'")
    dim, op, val = m.group(1), m.group(2).strip().lower(), m.group(3)
    if op == "in":
        return {"dimension": dim, "op": "in", "value": [v.strip() for v in val.split(",") if v.strip()]}
    return {"dimension": dim, "op": op, "value": val}


def _query_from_args(args: argparse.Namespace) -> dict[str, Any]:
    dims = [d.strip() for item in (args.by or []) for d in item.split(",") if d.strip()]
    query: dict[str, Any] = {
        "metrics": list(args.metric),
        "dimensions": dims,
        "filters": [parse_filter(f) for f in (args.filter or [])],
    }
    if args.limit is not None:
        query["limit"] = args.limit
    return query


def _add_query_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--metric", action="append", required=True, help="metric name (repeatable)")
    p.add_argument("--by", action="append",
                   help="dimension, name:grain, or a bare grain (day, month, year); repeatable or comma separated")
    p.add_argument("--filter", action="append", help="for example region=West or 'region in East,West'")
    p.add_argument("--limit", type=int, help="row limit (default 1000)")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")


def cmd_infer(args: argparse.Namespace) -> int:
    model, notes = infer_model(args.db, containment_cap=args.cap)
    save_model(model, args.output)
    print(f"wrote {args.output}: {len(model.joins)} joins, {len(model.metrics)} metrics, "
          f"{len(model.dimensions)} dimensions, all marked proposed")
    for j in model.joins:
        print(f"  join {j.id}  ({j.cardinality}, {j.evidence.get('source', '?')})")
    for n in notes:
        print(f"note: {n}")
    print("Review the file, set any join you disagree with to \"rejected\" (or run the reject command), "
          "then run approve.")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    if not (args.all or args.join or args.metric):
        print("error: give --all, --join ID or --metric NAME", file=sys.stderr)
        return 2
    model = load_model(args.model)
    n = set_status(model, "approved", all_proposed=args.all,
                   join_ids=args.join or [], metric_names=args.metric or [])
    save_model(model, args.model)
    print(f"approved {n} item(s) in {args.model}")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    if not (args.join or args.metric):
        print("error: give --join ID or --metric NAME", file=sys.stderr)
        return 2
    model = load_model(args.model)
    n = set_status(model, "rejected", join_ids=args.join or [], metric_names=args.metric or [])
    save_model(model, args.model)
    print(f"rejected {n} item(s) in {args.model}")
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    plan = compile_query(load_model(args.model), _query_from_args(args))
    if args.json:
        print(json.dumps(explain_plan(plan), indent=2, default=str))
    else:
        print(plan.sql)
        print(f"-- params: {json.dumps(plan.params)}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    plan = compile_query(load_model(args.model), _query_from_args(args))
    result = run_plan(args.db, plan)
    if args.json:
        print(json.dumps(result_records(result), indent=2, default=str))
    else:
        print(render_result(result))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    plan = compile_query(load_model(args.model), _query_from_args(args))
    if args.json:
        print(json.dumps(explain_plan(plan), indent=2, default=str))
    else:
        print(render_plan(plan))
    return 0


def cmd_demo_db(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"error: {out} already exists; pass --force to replace it", file=sys.stderr)
        return 2
    info = make_demo_db(out, seed=args.seed)
    print(f"wrote {out}")
    for table, n in sorted(info["rows"].items()):
        print(f"  {table}: {n} rows")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if not args.live:
        print("error: ask sends your question to a language model, so it only runs with --live", file=sys.stderr)
        return 2
    if not args.endpoint or not args.llm:
        print("error: --live needs --endpoint and --llm", file=sys.stderr)
        return 2
    model = load_model(args.model)
    query = ask_model(model, args.question, endpoint=args.endpoint, llm=args.llm, key_env=args.key_env)
    print("model suggested this query:")
    print(json.dumps(query, indent=2))
    result = run_plan(args.db, compile_query(model, query))
    print(render_result(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="semantic-layer-lite",
        description="A tiny semantic layer for SQLite: approve joins and metrics once, "
                    "then every question compiles to one safe query.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("infer", help="propose joins, metrics and dimensions from a SQLite file")
    p.add_argument("db")
    p.add_argument("-o", "--output", default="semantic.json")
    p.add_argument("--cap", type=int, default=10000, help="most distinct values to check per containment test")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("approve", help="mark proposed joins and metrics as approved")
    p.add_argument("model")
    p.add_argument("--all", action="store_true", help="approve everything still proposed")
    p.add_argument("--join", action="append", help="join id to approve (repeatable)")
    p.add_argument("--metric", action="append", help="metric name to approve (repeatable)")
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("reject", help="mark joins or metrics as rejected")
    p.add_argument("model")
    p.add_argument("--join", action="append", help="join id to reject (repeatable)")
    p.add_argument("--metric", action="append", help="metric name to reject (repeatable)")
    p.set_defaults(func=cmd_reject)

    p = sub.add_parser("compile", help="print the one SQL statement a metric query compiles to")
    p.add_argument("model")
    _add_query_args(p)
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("query", help="compile, run read-only, and print rows with their explanation")
    p.add_argument("db")
    p.add_argument("model")
    _add_query_args(p)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("explain", help="print the join path and metric definition behind a query")
    p.add_argument("model")
    _add_query_args(p)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("demo-db", help="write the seeded six-table demo database")
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true", help="replace an existing file")
    p.set_defaults(func=cmd_demo_db)

    p = sub.add_parser("ask", help="(optional, needs --live) map a plain question to a metric query")
    p.add_argument("db")
    p.add_argument("model")
    p.add_argument("question")
    p.add_argument("--live", action="store_true", help="allow a network call to your own model")
    p.add_argument("--endpoint", help="URL of a chat-completions style endpoint")
    p.add_argument("--llm", help="model name to send")
    p.add_argument("--key-env", default="SLL_API_KEY", help="environment variable holding the API key")
    p.set_defaults(func=cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 2 on a usage or file error, 3 when a query is refused."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CompileError as exc:
        print(f"refused ({exc.code}): {exc}", file=sys.stderr)
        return 3
    except (ModelError, InferError, LiveError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
