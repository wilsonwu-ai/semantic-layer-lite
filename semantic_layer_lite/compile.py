"""Compile a metric query into one read-only SELECT.

The compiler first builds a Plan (base table, join route for every table it
needs, grain, filters). The SQL string is rendered from that plan and from
nothing else, and explain.py prints the same plan.

It refuses instead of guessing: no approved route, two equally short approved
routes, or a join that repeats base-table rows for a metric that adds up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import GRAINS, NON_ADDITIVE, Join, Metric, Model, affinity

OPS = ("=", "!=", "<", "<=", ">", ">=", "in")
DEFAULT_LIMIT = 1000
MAX_LIMIT = 100000
MAX_IN_VALUES = 100
_STRFTIME = {"day": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}
_QUERY_KEYS = {"metrics", "dimensions", "filters", "limit"}


class CompileError(Exception):
    """A query the compiler refuses. code is a short machine-readable reason."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class Step:
    """One hop of a join route. from_table is already in the query."""

    join: Join
    from_table: str
    to_table: str
    fan_out: bool


@dataclass
class ResolvedDim:
    """A requested dimension, resolved to a column and (for time) a grain."""

    name: str
    alias: str
    table: str
    column: str
    grain: str | None
    expr: str


@dataclass
class ResolvedFilter:
    """A filter on a dimension's column, with its bound value."""

    dimension: str
    table: str
    column: str
    op: str
    value: Any


@dataclass
class Plan:
    """Everything the SQL is rendered from."""

    metrics: list[Metric]
    base_table: str
    dimensions: list[ResolvedDim]
    filters: list[ResolvedFilter]
    routes: dict[str, list[Step]]
    steps: list[Step]
    limit: int
    sql: str = ""
    params: list[Any] = field(default_factory=list)

    def join_ids(self) -> list[str]:
        """Ids of every join the query uses, in the order they are applied."""
        return [s.join.id for s in self.steps]


def _q(table: str, column: str) -> str:
    return f'"{table}"."{column}"'


def _steps_from(joins: list[Join], table: str) -> list[Step]:
    out: list[Step] = []
    for j in sorted(joins, key=lambda x: x.id):
        if j.from_table == j.to_table:
            continue
        if j.from_table == table:
            out.append(Step(j, j.from_table, j.to_table, False))
        elif j.to_table == table:
            out.append(Step(j, j.to_table, j.from_table, j.cardinality == "many_to_one"))
    return out


def find_shortest_paths(joins: list[Join], start: str, goal: str, cap: int = 8) -> list[list[Step]]:
    """Breadth-first search. Return every shortest path from start to goal (up to cap)."""
    if start == goal:
        return [[]]
    dist = {start: 0}
    preds: dict[str, list[tuple[str, Step]]] = {start: []}
    frontier = [start]
    while frontier and goal not in dist:
        nxt: list[str] = []
        for t in frontier:
            for step in _steps_from(joins, t):
                u = step.to_table
                if u not in dist:
                    dist[u] = dist[t] + 1
                    preds[u] = [(t, step)]
                    nxt.append(u)
                elif dist[u] == dist[t] + 1:
                    preds[u].append((t, step))
        frontier = sorted(set(nxt))
    if goal not in dist:
        return []
    out: list[list[Step]] = []

    def walk(table: str, suffix: list[Step]) -> None:
        if len(out) >= cap:
            return
        if table == start:
            out.append(suffix[::-1])
            return
        for prev, step in preds[table]:
            walk(prev, suffix + [step])

    walk(goal, [])
    return out


def describe_path(start: str, path: list[Step]) -> str:
    """Render a path as 'a -> b -> c'."""
    return " -> ".join([start] + [s.to_table for s in path])


def _route(model: Model, base: str, table: str) -> list[Step]:
    approved = [j for j in model.joins if j.status == "approved"]
    found = find_shortest_paths(approved, base, table)
    if len(found) > 1:
        options = "; ".join(describe_path(base, p) + " via " + ", ".join(s.join.id for s in p) for p in found)
        raise CompileError(
            "ambiguous_path",
            f"{len(found)} equally short approved routes from {base} to {table}: {options}. "
            "Reject or fix one of these joins in semantic.json.",
        )
    if found:
        return found[0]
    usable = [j for j in model.joins if j.status != "rejected"]
    alt = find_shortest_paths(usable, base, table)
    if alt:
        pending = [s.join.id for s in alt[0] if s.join.status != "approved"]
        raise CompileError(
            "join_not_approved",
            f"no approved route from {base} to {table}; these joins are still proposed: {', '.join(pending)}",
        )
    alt = find_shortest_paths(model.joins, base, table)
    if alt:
        gone = [s.join.id for s in alt[0] if s.join.status == "rejected"]
        raise CompileError(
            "join_rejected",
            f"no approved route from {base} to {table}; the only route uses rejected joins: {', '.join(gone)}",
        )
    raise CompileError("no_path", f"no join of any status connects {base} to {table}")


def _resolve_dimension(model: Model, spec: Any, metrics: list[Metric]) -> ResolvedDim:
    if not isinstance(spec, str):
        raise CompileError("bad_query", "each dimension must be a string")
    grain: str | None = None
    name = spec
    if spec in GRAINS:
        name = metrics[0].time_dimension or ""
        grain = spec
        if not name:
            raise CompileError("bad_grain", f"metric {metrics[0].name} has no time_dimension, so {spec!r} has no column to group")
    elif ":" in spec:
        name, grain = spec.split(":", 1)
        if grain not in GRAINS:
            raise CompileError("bad_grain", f"grain must be one of {GRAINS}, not {grain!r}")
    dim = model.dimensions.get(name)
    if dim is None:
        raise CompileError("unknown_dimension", f"unknown dimension {name!r}; known: {', '.join(sorted(model.dimensions))}")
    if grain is not None and dim.kind != "time":
        raise CompileError("bad_grain", f"dimension {name} is not a time dimension, so it has no grain")
    if dim.kind == "time" and grain is None:
        grain = "day"
    col = _q(dim.table, dim.column)
    if grain is None:
        return ResolvedDim(name, name, dim.table, dim.column, None, col)
    return ResolvedDim(name, f"{name}_{grain}", dim.table, dim.column, grain,
                       f"strftime('{_STRFTIME[grain]}', {col})")


def _coerce(value: Any, decl: str, where: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise CompileError("bad_filter_value", f"{where}: value must be a string or a number")
    kind = affinity(decl)
    try:
        if isinstance(value, str) and kind == "integer":
            return int(value)
        if isinstance(value, str) and kind == "real":
            return float(value)
    except ValueError:
        raise CompileError("bad_filter_value", f"{where}: {value!r} is not a valid {decl or kind} value") from None
    return value


def _resolve_filter(model: Model, spec: Any) -> ResolvedFilter:
    if not isinstance(spec, dict) or set(spec) != {"dimension", "op", "value"}:
        raise CompileError("bad_filter", "a filter needs exactly dimension, op and value")
    name, op, value = spec["dimension"], spec["op"], spec["value"]
    if op not in OPS:
        raise CompileError("bad_filter", f"filter op must be one of {OPS}, not {op!r}")
    dim = model.dimensions.get(name) if isinstance(name, str) else None
    if dim is None:
        raise CompileError("unknown_dimension", f"unknown filter dimension {name!r}")
    decl = model.entities[dim.table].columns[dim.column]
    where = f"filter on {name}"
    if op == "in":
        if not isinstance(value, list) or not value or len(value) > MAX_IN_VALUES:
            raise CompileError("bad_filter_value", f"{where}: 'in' needs a list of 1 to {MAX_IN_VALUES} values")
        value = [_coerce(v, decl, where) for v in value]
    else:
        value = _coerce(value, decl, where)
    return ResolvedFilter(name, dim.table, dim.column, op, value)


def _render(model: Model, plan: Plan) -> tuple[str, list[Any]]:
    select = [f'{d.expr} AS "{d.alias}"' for d in plan.dimensions]
    select += [f'{model.metric_sql(m.name)} AS "{m.name}"' for m in plan.metrics]
    lines = ["SELECT " + ", ".join(select), f'FROM "{plan.base_table}"']
    for s in plan.steps:
        j = s.join
        lines.append(
            f'LEFT JOIN "{s.to_table}" ON "{j.from_table}"."{j.from_column}" = "{j.to_table}"."{j.to_column}"'
        )
    params: list[Any] = []
    conds: list[str] = []
    for f in plan.filters:
        col = _q(f.table, f.column)
        if f.op == "in":
            conds.append(f"{col} IN ({', '.join('?' for _ in f.value)})")
            params.extend(f.value)
        else:
            conds.append(f"{col} {f.op} ?")
            params.append(f.value)
    if conds:
        lines.append("WHERE " + " AND ".join(conds))
    if plan.dimensions:
        pos = ", ".join(str(i + 1) for i in range(len(plan.dimensions)))
        lines.append("GROUP BY " + pos)
        lines.append("ORDER BY " + pos)
    lines.append(f"LIMIT {int(plan.limit)}")
    return "\n".join(lines), params


def compile_query(model: Model, query: dict[str, Any]) -> Plan:
    """Compile {"metrics", "dimensions", "filters", "limit"} into a Plan with SQL and params.

    Raises CompileError when the query cannot be answered safely.
    """
    if not isinstance(query, dict) or set(query) - _QUERY_KEYS:
        raise CompileError("bad_query", f"a query is an object with keys from {sorted(_QUERY_KEYS)}")
    names = query.get("metrics")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise CompileError("bad_query", "metrics must be a non-empty list of metric names")
    if len(set(names)) != len(names):
        raise CompileError("bad_query", "a metric is listed twice")
    metrics: list[Metric] = []
    for n in names:
        m = model.metrics.get(n)
        if m is None:
            raise CompileError("unknown_metric", f"unknown metric {n!r}; known: {', '.join(sorted(model.metrics))}")
        if m.status != "approved":
            raise CompileError("metric_not_approved", f"metric {n} is {m.status}, not approved")
        metrics.append(m)
    base = metrics[0].base_table
    if any(m.base_table != base for m in metrics):
        raise CompileError("mixed_base", "metrics in one query must share a base table")

    raw_dims = query.get("dimensions", [])
    if not isinstance(raw_dims, list):
        raise CompileError("bad_query", "dimensions must be a list")
    dims = [_resolve_dimension(model, d, metrics) for d in raw_dims]
    aliases = [d.alias for d in dims] + names
    if len(set(aliases)) != len(aliases):
        raise CompileError("bad_query", "a dimension or metric appears twice in the output")
    raw_filters = query.get("filters", [])
    if not isinstance(raw_filters, list):
        raise CompileError("bad_query", "filters must be a list")
    filters = [_resolve_filter(model, f) for f in raw_filters]
    limit = query.get("limit", DEFAULT_LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise CompileError("bad_limit", f"limit must be a whole number from 1 to {MAX_LIMIT}")

    needed = sorted({d.table for d in dims} | {f.table for f in filters})
    routes = {t: _route(model, base, t) for t in needed if t != base}
    additive = [m.name for m in metrics if m.aggregation not in NON_ADDITIVE]
    if additive:
        for table, path in routes.items():
            for s in path:
                if s.fan_out:
                    raise CompileError(
                        "fan_out",
                        f"join {s.join.id} lets one {s.from_table} row match many {s.to_table} rows, "
                        f"so {', '.join(additive)} would be counted more than once when reaching {table}",
                    )
    steps: list[Step] = []
    joined = {base}
    for table in sorted(routes):
        for s in routes[table]:
            if s.to_table not in joined:
                joined.add(s.to_table)
                steps.append(s)
    plan = Plan(metrics, base, dims, filters, routes, steps, limit)
    plan.sql, plan.params = _render(model, plan)
    return plan
