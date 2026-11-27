"""Shared helpers for the tests (not a test module itself)."""
import atexit
import json
import pathlib
import sqlite3
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_layer_lite.datasets import make_demo_db  # noqa: E402
from semantic_layer_lite.model import load_model  # noqa: E402

EXAMPLE = ROOT / "examples" / "semantic.json"
_state = {}


def demo():
    """Return (db_path, info) for one shared, seeded demo database. Never write to it."""
    if "db" not in _state:
        tmp = tempfile.TemporaryDirectory()
        atexit.register(tmp.cleanup)
        path = pathlib.Path(tmp.name) / "demo.sqlite"
        info = make_demo_db(path, seed=7)
        _state.update(tmp=tmp, db=path, info=info)
    return _state["db"], _state["info"]


def example_model():
    """The reviewed model in examples/semantic.json."""
    return load_model(EXAMPLE)


def example_dict():
    """A fresh parsed copy of examples/semantic.json that a test may edit."""
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def fetch(db, sql, params=()):
    """Run a plain read on the demo database with the sqlite3 module directly."""
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()
