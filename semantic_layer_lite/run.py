"""Run compiled queries against SQLite, read-only.

The connection is opened with mode=ro and PRAGMA query_only = ON, so a write
fails even if a bug let one through. run_sql adds a second guard: only
SELECT and WITH statements are accepted.
"""
from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .compile import Plan, compile_query
from .model import Model


class ReadOnlyError(Exception):
    """Raised when something other than a SELECT is sent to run_sql."""


@dataclass
class QueryResult:
    """Rows from one compiled query, with the plan that produced them."""

    plan: Plan
    columns: list[str]
    rows: list[tuple]


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite file read-only. Never creates the file."""
    p = Path(db_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"database file not found: {db_path}")
    conn = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def run_sql(db_path: str | Path, sql: str, params: Sequence[Any] = ()) -> tuple[list[str], list[tuple]]:
    """Run one read-only statement. Returns (column names, rows)."""
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        raise ReadOnlyError("only SELECT statements may run")
    conn = connect_readonly(db_path)
    try:
        cur = conn.execute(sql, tuple(params))
        columns = [d[0] for d in cur.description]
        return columns, cur.fetchall()
    finally:
        conn.close()


def run_plan(db_path: str | Path, plan: Plan) -> QueryResult:
    """Execute a compiled plan on a fresh read-only connection."""
    columns, rows = run_sql(db_path, plan.sql, plan.params)
    return QueryResult(plan, columns, rows)


def run_query(db_path: str | Path, model: Model, query: dict[str, Any]) -> QueryResult:
    """Compile a metric query and run it."""
    return run_plan(db_path, compile_query(model, query))


def same_rows(a: Sequence[Sequence[Any]], b: Sequence[Sequence[Any]], tol: float = 1e-6) -> bool:
    """True when two row lists match, comparing floats within a small tolerance."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if isinstance(x, float) or isinstance(y, float):
                if x is None or y is None:
                    return False
                if not math.isclose(x, y, rel_tol=1e-9, abs_tol=tol):
                    return False
            elif x != y:
                return False
    return True
