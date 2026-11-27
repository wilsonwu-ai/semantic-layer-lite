"""Load, validate, edit and save the semantic model (semantic.json).

The model is plain JSON that a person can read and edit. Joins and metrics
carry a status: proposed, approved or rejected. Only approved ones are ever
used to build SQL.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STATUSES = ("proposed", "approved", "rejected")
CARDINALITIES = ("many_to_one", "one_to_one")
AGGREGATIONS = ("sum", "count", "avg", "min", "max", "count_distinct")
NON_ADDITIVE = ("count_distinct", "min", "max")
DIM_KINDS = ("time", "categorical")
GRAINS = ("day", "month", "year")
MODEL_VERSION = 1

_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|([-+*/()]))")


class ModelError(ValueError):
    """Raised when semantic.json is unreadable or breaks a rule."""


def affinity(decl: str) -> str:
    """Return SQLite's type affinity for a declared column type."""
    d = (decl or "").upper()
    if "INT" in d:
        return "integer"
    if "CHAR" in d or "CLOB" in d or "TEXT" in d:
        return "text"
    if "BLOB" in d or d == "":
        return "blob"
    if "REAL" in d or "FLOA" in d or "DOUB" in d:
        return "real"
    return "numeric"


@dataclass
class Entity:
    """One table: its primary key columns and its declared column types."""

    name: str
    primary_key: list[str]
    columns: dict[str, str]


@dataclass
class Join:
    """A link between two tables. from_* is the side that holds the key."""

    id: str
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    cardinality: str
    status: str = "proposed"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """A named number: an aggregation of an expression over a base table."""

    name: str
    base_table: str
    aggregation: str
    expression: str
    description: str = ""
    status: str = "proposed"
    time_dimension: str | None = None


@dataclass
class Dimension:
    """A column a metric may be grouped or filtered by."""

    name: str
    table: str
    column: str
    kind: str


@dataclass
class Model:
    """The whole semantic model."""

    entities: dict[str, Entity]
    joins: list[Join]
    metrics: dict[str, Metric]
    dimensions: dict[str, Dimension]
    version: int = MODEL_VERSION

    def metric_sql(self, name: str) -> str:
        """Return the aggregate SQL for a metric, built from declared columns only."""
        m = self.metrics[name]
        return aggregate_sql(m, self.entities[m.base_table].columns)


def make_join_id(from_table: str, from_column: str, to_table: str, to_column: str) -> str:
    """Build the default id of a join, for example orders.customer_id=customers.customer_id."""
    return f"{from_table}.{from_column}={to_table}.{to_column}"


def tokenize_expression(expr: str) -> list[tuple[str, str]]:
    """Split a metric expression into (kind, text) tokens: num, id or op."""
    text = expr.strip()
    if not text:
        raise ModelError("empty expression")
    pos = 0
    out: list[tuple[str, str]] = []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ModelError(f"cannot read expression {expr!r} near position {pos}")
        num, ident, op = m.groups()
        if num is not None:
            out.append(("num", num))
        elif ident is not None:
            out.append(("id", ident))
        else:
            out.append(("op", op))
        pos = m.end()
    return out


def render_expression(expr: str, table: str, columns: Any) -> str:
    """Turn a metric expression into SQL, quoting each column and rejecting anything else."""
    depth = 0
    prev_operand = False
    parts: list[str] = []
    for kind, val in tokenize_expression(expr):
        if kind == "id":
            if val not in columns:
                raise ModelError(f"expression {expr!r} uses {val!r}, which is not a column of {table}")
            if prev_operand:
                raise ModelError(f"expression {expr!r} has two values in a row")
            parts.append(f'"{table}"."{val}"')
            prev_operand = True
        elif kind == "num":
            if prev_operand:
                raise ModelError(f"expression {expr!r} has two values in a row")
            parts.append(val)
            prev_operand = True
        else:
            if val == "(":
                depth += 1
            elif val == ")":
                depth -= 1
                if depth < 0:
                    raise ModelError(f"expression {expr!r} has an unmatched )")
            parts.append(val)
            prev_operand = val == ")"
    if depth != 0:
        raise ModelError(f"expression {expr!r} has an unmatched (")
    return " ".join(parts)


def aggregate_sql(metric: Metric, columns: Any) -> str:
    """Return SUM(...), COUNT(...) and so on for a metric over its base table's columns."""
    if metric.aggregation == "count" and metric.expression.strip() == "*":
        return "COUNT(*)"
    inner = render_expression(metric.expression, metric.base_table, columns)
    if metric.aggregation == "count_distinct":
        return f"COUNT(DISTINCT {inner})"
    return f"{metric.aggregation.upper()}({inner})"


def _need(d: Any, key: str, typ: type, where: str) -> Any:
    if not isinstance(d, dict):
        raise ModelError(f"{where} must be a JSON object")
    val = d.get(key)
    if not isinstance(val, typ):
        raise ModelError(f"{where}: {key!r} must be a {typ.__name__}")
    return val


def _ident(name: Any, what: str) -> str:
    if not isinstance(name, str) or not IDENT.match(name):
        raise ModelError(f"{what} {name!r} must be letters, digits and underscores, not starting with a digit")
    return name


def model_from_dict(data: Any) -> Model:
    """Validate a parsed semantic.json and return a Model. Raises ModelError."""
    if not isinstance(data, dict):
        raise ModelError("semantic model must be a JSON object")
    if data.get("version", MODEL_VERSION) != MODEL_VERSION:
        raise ModelError(f"unsupported model version {data.get('version')!r}")
    entities: dict[str, Entity] = {}
    for name, e in _need(data, "entities", dict, "model").items():
        _ident(name, "table name")
        cols = _need(e, "columns", dict, f"entity {name}")
        for c in cols:
            _ident(c, f"column name in {name}")
        pk = e.get("primary_key", [])
        if not isinstance(pk, list) or any(p not in cols for p in pk):
            raise ModelError(f"entity {name}: primary_key must list columns of the table")
        entities[name] = Entity(name, list(pk), {c: str(t) for c, t in cols.items()})

    joins: list[Join] = []
    seen_ids: set[str] = set()
    for j in _need(data, "joins", list, "model"):
        where = f"join {j.get('id') if isinstance(j, dict) else j!r}"
        jid = _need(j, "id", str, where)
        if not jid or jid in seen_ids:
            raise ModelError(f"{where}: id must be unique and not empty")
        seen_ids.add(jid)
        ft, fc = _need(j, "from_table", str, where), _need(j, "from_column", str, where)
        tt, tc = _need(j, "to_table", str, where), _need(j, "to_column", str, where)
        for t, c in ((ft, fc), (tt, tc)):
            if t not in entities or c not in entities[t].columns:
                raise ModelError(f"{where}: {t}.{c} is not a known column")
        card = _need(j, "cardinality", str, where)
        if card not in CARDINALITIES:
            raise ModelError(f"{where}: cardinality must be one of {CARDINALITIES}")
        status = _need(j, "status", str, where)
        if status not in STATUSES:
            raise ModelError(f"{where}: status must be one of {STATUSES}")
        ev = j.get("evidence", {})
        if not isinstance(ev, dict):
            raise ModelError(f"{where}: evidence must be an object")
        joins.append(Join(jid, ft, fc, tt, tc, card, status, dict(ev)))

    dimensions: dict[str, Dimension] = {}
    for d in _need(data, "dimensions", list, "model"):
        name = _ident(_need(d, "name", str, "dimension"), "dimension name")
        where = f"dimension {name}"
        table, column = _need(d, "table", str, where), _need(d, "column", str, where)
        if table not in entities or column not in entities[table].columns:
            raise ModelError(f"{where}: {table}.{column} is not a known column")
        kind = _need(d, "kind", str, where)
        if kind not in DIM_KINDS:
            raise ModelError(f"{where}: kind must be one of {DIM_KINDS}")
        if name in dimensions:
            raise ModelError(f"duplicate dimension {name}")
        dimensions[name] = Dimension(name, table, column, kind)

    metrics: dict[str, Metric] = {}
    for m in _need(data, "metrics", list, "model"):
        name = _ident(_need(m, "name", str, "metric"), "metric name")
        where = f"metric {name}"
        base = _need(m, "base_table", str, where)
        if base not in entities:
            raise ModelError(f"{where}: unknown base_table {base!r}")
        agg = _need(m, "aggregation", str, where)
        if agg not in AGGREGATIONS:
            raise ModelError(f"{where}: aggregation must be one of {AGGREGATIONS}")
        status = _need(m, "status", str, where)
        if status not in STATUSES:
            raise ModelError(f"{where}: status must be one of {STATUSES}")
        td = m.get("time_dimension")
        if td is not None and (td not in dimensions or dimensions[td].kind != "time"):
            raise ModelError(f"{where}: time_dimension {td!r} must name a time dimension")
        if name in metrics:
            raise ModelError(f"duplicate metric {name}")
        metric = Metric(name, base, agg, _need(m, "expression", str, where),
                        str(m.get("description", "")), status, td)
        aggregate_sql(metric, entities[base].columns)  # raises ModelError on a bad expression
        metrics[name] = metric
    return Model(entities, joins, metrics, dimensions)


def model_to_dict(model: Model) -> dict[str, Any]:
    """Return a JSON-ready dict with joins, metrics and dimensions in sorted order."""
    return {
        "version": model.version,
        "entities": {
            n: {"primary_key": list(e.primary_key), "columns": dict(e.columns)}
            for n, e in sorted(model.entities.items())
        },
        "joins": [asdict(j) for j in sorted(model.joins, key=lambda j: j.id)],
        "metrics": [asdict(m) for _, m in sorted(model.metrics.items())],
        "dimensions": [asdict(d) for _, d in sorted(model.dimensions.items())],
    }


def load_model(path: str | Path) -> Model:
    """Read and validate a semantic.json file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelError(f"{path} is not valid JSON: {exc}") from exc
    return model_from_dict(data)


def save_model(model: Model, path: str | Path) -> None:
    """Write the model as indented JSON."""
    text = json.dumps(model_to_dict(model), indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def set_status(
    model: Model,
    status: str,
    *,
    all_proposed: bool = False,
    join_ids: tuple[str, ...] | list[str] = (),
    metric_names: tuple[str, ...] | list[str] = (),
) -> int:
    """Set the status of the named joins and metrics (and every proposed one if asked).

    Returns how many items changed. An unknown id or name raises ModelError.
    """
    if status not in STATUSES:
        raise ModelError(f"status must be one of {STATUSES}")
    by_id = {j.id: j for j in model.joins}
    for jid in join_ids:
        if jid not in by_id:
            raise ModelError(f"no join with id {jid!r}")
    for name in metric_names:
        if name not in model.metrics:
            raise ModelError(f"no metric named {name!r}")
    targets: list[Any] = [by_id[i] for i in join_ids] + [model.metrics[n] for n in metric_names]
    if all_proposed:
        targets += [j for j in model.joins if j.status == "proposed"]
        targets += [m for m in model.metrics.values() if m.status == "proposed"]
    changed = 0
    done: set[int] = set()
    for obj in targets:
        if id(obj) in done:
            continue
        done.add(id(obj))
        if obj.status != status:
            obj.status = status
            changed += 1
    return changed
