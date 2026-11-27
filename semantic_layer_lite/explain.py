"""Explain every number: the metric definition and the join path behind it.

explain_plan and render_plan describe a compiled Plan. result_records and
render_result attach the same facts (metric definitions, join ids) to the
rows a query returned, so each number travels with its audit trail.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from .compile import Plan
from .model import Metric

if TYPE_CHECKING:  # pragma: no cover
    from .run import QueryResult


def metric_text(m: Metric) -> str:
    """Readable definition of a metric, for example SUM(line_total) over order_items."""
    if m.aggregation == "count_distinct":
        body = f"COUNT(DISTINCT {m.expression})"
    else:
        body = f"{m.aggregation.upper()}({m.expression})"
    return f"{body} over {m.base_table}"


def explain_plan(plan: Plan) -> dict[str, Any]:
    """Return the plan as a JSON-ready dict: metrics, join routes, grain, filters, SQL."""
    return {
        "metrics": [
            {
                "name": m.name,
                "definition": metric_text(m),
                "base_table": m.base_table,
                "status": m.status,
                "description": m.description,
            }
            for m in plan.metrics
        ],
        "base_table": plan.base_table,
        "routes": {
            table: [
                {
                    "join_id": s.join.id,
                    "from": s.from_table,
                    "to": s.to_table,
                    "cardinality": s.join.cardinality,
                    "fan_out": s.fan_out,
                    "status": s.join.status,
                }
                for s in path
            ]
            for table, path in sorted(plan.routes.items())
        },
        "dimensions": [
            {"alias": d.alias, "table": d.table, "column": d.column, "grain": d.grain}
            for d in plan.dimensions
        ],
        "filters": [
            {"dimension": f.dimension, "op": f.op, "value": f.value} for f in plan.filters
        ],
        "limit": plan.limit,
        "join_ids": plan.join_ids(),
        "sql": plan.sql,
        "params": list(plan.params),
    }


def render_plan(plan: Plan) -> str:
    """Return the plan as plain text for a person to read."""
    lines: list[str] = []
    for m in plan.metrics:
        lines.append(f"metric {m.name} = {metric_text(m)} (status {m.status})")
        if m.description:
            lines.append(f"  meaning: {m.description}")
    lines.append("join path:")
    if not plan.routes:
        lines.append(f"  none needed, every column comes from {plan.base_table}")
    for table, path in sorted(plan.routes.items()):
        hops = " -> ".join([plan.base_table] + [s.to_table for s in path])
        lines.append(f"  to reach {table}: {hops}")
        for s in path:
            note = "fans out" if s.fan_out else "no fan-out"
            lines.append(f"    {s.join.id} ({s.join.cardinality}, {note}, status {s.join.status})")
    for d in plan.dimensions:
        grain = f" by {d.grain}" if d.grain else ""
        lines.append(f"group by: {d.table}.{d.column}{grain} as {d.alias}")
    for f in plan.filters:
        lines.append(f"filter: {f.table}.{f.column} {f.op} {f.value!r} (bound as a parameter)")
    lines.append(f"row limit: {plan.limit}")
    lines.append("SQL:")
    lines.extend("  " + ln for ln in plan.sql.splitlines())
    if plan.params:
        lines.append(f"  -- params: {list(plan.params)!r}")
    return "\n".join(lines)


def result_records(result: "QueryResult") -> list[dict[str, Any]]:
    """Return one dict per row, each carrying the join ids and metric definitions behind it."""
    join_ids = result.plan.join_ids()
    defs = {m.name: metric_text(m) for m in result.plan.metrics}
    records = []
    for row in result.rows:
        rec: dict[str, Any] = dict(zip(result.columns, row))
        rec["_join_ids"] = list(join_ids)
        rec["_metrics"] = dict(defs)
        records.append(rec)
    return records


def _fmt(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render rows as an aligned text table."""
    cells = [[_fmt(v) for v in r] for r in rows]
    widths = [max([len(c)] + [len(r[i]) for r in cells]) for i, c in enumerate(columns)]
    head = "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in cells]
    return "\n".join([head, rule] + body)


def render_result(result: "QueryResult") -> str:
    """Return the result table followed by the explanation of how it was made."""
    table = format_table(result.columns, result.rows)
    return f"{table}\n({len(result.rows)} rows)\n\nHow these numbers were made:\n{render_plan(result.plan)}"
