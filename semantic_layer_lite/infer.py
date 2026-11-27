"""Read a SQLite file and propose a semantic model for a person to review.

Joins come from two places. Declared foreign keys are taken as written.
Undeclared joins need three things at once: the column name points at the
target table, the target column is unique, and every non-null value in the
source column exists in the target (checked with SQL, capped). Two integer
columns that merely happen to overlap are not enough.

Everything proposed is marked "proposed". Nothing is usable until approved.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import (IDENT, Dimension, Entity, Join, Metric, Model, affinity,
                    make_join_id)
from .run import connect_readonly

_ISO = re.compile(
    r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$"
)


class InferError(Exception):
    """Raised when the database cannot be read."""


@dataclass
class _Table:
    columns: dict[str, str]
    pk: list[str]
    fks: list[tuple[str, str, str | None]]
    unique_cols: set[str] = field(default_factory=set)


def singular(word: str) -> str:
    """Naive English singular: categories -> category, orders -> order."""
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _read_tables(conn: sqlite3.Connection, notes: list[str]) -> dict[str, _Table]:
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    tables: dict[str, _Table] = {}
    for t in names:
        if not IDENT.match(t):
            notes.append(f"skipped table {t!r}: the name is not a plain identifier")
            continue
        rows = conn.execute(f"PRAGMA table_info({_qi(t)})").fetchall()
        cols: dict[str, str] = {}
        for _cid, name, decl, _nn, _dflt, _pk in rows:
            if IDENT.match(name):
                cols[name] = decl or ""
            else:
                notes.append(f"skipped column {t}.{name!r}: the name is not a plain identifier")
        pk = [r[1] for r in sorted((r for r in rows if r[5] > 0), key=lambda r: r[5]) if r[1] in cols]
        groups: dict[int, list[tuple[int, str, str, Any]]] = {}
        for fid, seq, ref, frm, to, *_rest in conn.execute(f"PRAGMA foreign_key_list({_qi(t)})"):
            groups.setdefault(fid, []).append((seq, ref, frm, to))
        fks = []
        for fid in sorted(groups):
            if len(groups[fid]) != 1:
                notes.append(f"skipped a composite foreign key on {t}: only single-column joins are supported")
                continue
            _seq, ref, frm, to = groups[fid][0]
            fks.append((frm, ref, to))
        unique = set(pk) if len(pk) == 1 else set()
        for idx in conn.execute(f"PRAGMA index_list({_qi(t)})").fetchall():
            if idx[2] and not (len(idx) > 4 and idx[4]):
                info = conn.execute(f"PRAGMA index_info({_qi(idx[1])})").fetchall()
                if len(info) == 1 and info[0][2] in cols:
                    unique.add(info[0][2])
        tables[t] = _Table(cols, pk, fks, unique)
    return tables


def _is_unique(conn: sqlite3.Connection, table: str, col: str) -> bool:
    nonnull, distinct = conn.execute(
        f"SELECT COUNT({_qi(col)}), COUNT(DISTINCT {_qi(col)}) FROM {_qi(table)}").fetchone()
    return nonnull > 0 and nonnull == distinct


def _containment(conn: sqlite3.Connection, t: str, c: str, u: str, k: str, cap: int) -> tuple[int, int]:
    """Return (distinct values checked, how many of them exist in u.k), checking at most cap."""
    sql = (
        f"SELECT COUNT(*), COALESCE(SUM(v IN (SELECT {_qi(k)} FROM {_qi(u)})), 0) FROM "
        f"(SELECT DISTINCT {_qi(c)} AS v FROM {_qi(t)} WHERE {_qi(c)} IS NOT NULL LIMIT ?)"
    )
    total, inside = conn.execute(sql, (cap,)).fetchone()
    return int(total), int(inside)


def _name_rule(col: str, target: str, key: str) -> str | None:
    if col in (f"{target}_id", f"{singular(target)}_id"):
        return "table_name_id"
    if col == key and col.endswith("_id"):
        return "same_key_name"
    return None


def _find_joins(conn: sqlite3.Connection, tables: dict[str, _Table], cap: int, notes: list[str]) -> list[Join]:
    joins: list[Join] = []
    declared: set[tuple[str, str]] = set()
    for t in sorted(tables):
        for frm, ref, to in tables[t].fks:
            if frm not in tables[t].columns or ref not in tables:
                notes.append(f"skipped a foreign key on {t}.{frm}: it points at an unreadable table or column")
                continue
            if to is None:
                if len(tables[ref].pk) != 1:
                    notes.append(f"skipped foreign key {t}.{frm}: the target has no single-column primary key")
                    continue
                to = tables[ref].pk[0]
            if to not in tables[ref].columns:
                notes.append(f"skipped foreign key {t}.{frm}: {ref}.{to} does not exist")
                continue
            card = "one_to_one" if _is_unique(conn, t, frm) else "many_to_one"
            joins.append(Join(make_join_id(t, frm, ref, to), t, frm, ref, to, card, "proposed",
                              {"source": "declared_foreign_key"}))
            declared.add((t, frm))
    for t in sorted(tables):
        for c, decl in tables[t].columns.items():
            aff = affinity(decl)
            if (t, c) in declared or aff not in ("integer", "text"):
                continue
            for u in sorted(tables):
                if u == t:
                    continue
                for k in sorted(tables[u].unique_cols):
                    if affinity(tables[u].columns[k]) != aff:
                        continue
                    rule = _name_rule(c, u, k)
                    if rule is None:
                        continue
                    if tables[t].pk == [c] and tables[u].pk != [k]:
                        continue  # a table's own key is not a foreign key to a non-key column
                    total, inside = _containment(conn, t, c, u, k, cap)
                    if total == 0 or inside != total:
                        continue
                    card = "one_to_one" if _is_unique(conn, t, c) else "many_to_one"
                    joins.append(Join(make_join_id(t, c, u, k), t, c, u, k, card, "proposed", {
                        "source": "inferred", "name_rule": rule, "target_is_unique": True,
                        "values_checked": total, "values_contained": inside,
                        "capped": total >= cap,
                    }))
    return sorted(joins, key=lambda j: j.id)


def _sample(conn: sqlite3.Connection, t: str, c: str, n: int = 200) -> list[Any]:
    return [r[0] for r in conn.execute(
        f"SELECT {_qi(c)} FROM {_qi(t)} WHERE {_qi(c)} IS NOT NULL LIMIT {int(n)}")]


def _find_dimensions(conn: sqlite3.Connection, tables: dict[str, _Table], key_cols: set[tuple[str, str]],
                     categorical_max: int) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for t in sorted(tables):
        for c, decl in sorted(tables[t].columns.items()):
            aff = affinity(decl)
            if aff not in ("text", "numeric") or (t, c) in key_cols or c.endswith("_id"):
                continue
            values = _sample(conn, t, c)
            if values and all(isinstance(v, str) and _ISO.match(v) for v in values):
                found.append((t, c, "time"))
                continue
            if aff != "text" or not values:
                continue
            nonnull, distinct = conn.execute(
                f"SELECT COUNT({_qi(c)}), COUNT(DISTINCT {_qi(c)}) FROM {_qi(t)}").fetchone()
            if 1 <= distinct <= categorical_max and distinct * 2 <= nonnull:
                found.append((t, c, "categorical"))
    return found


def infer_model(db_path: str | Path, *, containment_cap: int = 10000,
                categorical_max: int = 20) -> tuple[Model, list[str]]:
    """Propose a Model from a SQLite file. Returns (model, notes). All items are "proposed"."""
    notes: list[str] = []
    try:
        conn = connect_readonly(db_path)
    except (FileNotFoundError, sqlite3.Error) as exc:
        raise InferError(f"cannot open {db_path} read-only: {exc}") from exc
    try:
        tables = _read_tables(conn, notes)
        joins = _find_joins(conn, tables, containment_cap, notes)
        key_cols = {(j.from_table, j.from_column) for j in joins} | {(j.to_table, j.to_column) for j in joins}
        key_cols |= {(t, c) for t, info in tables.items() for c in info.pk}
        dim_rows = _find_dimensions(conn, tables, key_cols, categorical_max)
        dimensions: dict[str, Dimension] = {}
        for t, c, kind in dim_rows:
            name = c if c not in dimensions else f"{t}_{c}"
            dimensions[name] = Dimension(name, t, c, kind)
    except sqlite3.DatabaseError as exc:
        raise InferError(f"cannot read {db_path}: {exc}") from exc
    finally:
        conn.close()

    entities = {t: Entity(t, list(info.pk), dict(info.columns)) for t, info in tables.items()}
    time_dims: dict[str, str] = {}
    for d in sorted(dimensions.values(), key=lambda d: d.name):
        if d.kind == "time":
            time_dims.setdefault(d.table, d.name)

    def default_time(base: str) -> str | None:
        if base in time_dims:
            return time_dims[base]
        for j in joins:  # joins are sorted by id
            if j.from_table == base and j.to_table in time_dims:
                return time_dims[j.to_table]
        return None

    metrics: dict[str, Metric] = {}
    time_cols = {(d.table, d.column) for d in dimensions.values() if d.kind == "time"}
    for t in sorted(tables):
        for c, decl in tables[t].columns.items():
            if affinity(decl) not in ("integer", "real", "numeric"):
                continue
            if (t, c) in key_cols or (t, c) in time_cols or c == "id" or c.endswith("_id"):
                continue
            name = f"total_{c}" if f"total_{c}" not in metrics else f"{t}_total_{c}"
            metrics[name] = Metric(name, t, "sum", c, f"Proposed: sum of {t}.{c}", "proposed", default_time(t))
        cname = f"{t}_count"
        metrics[cname] = Metric(cname, t, "count", "*", f"Proposed: number of rows in {t}", "proposed",
                                default_time(t))
    return Model(entities, joins, metrics, dimensions), notes
