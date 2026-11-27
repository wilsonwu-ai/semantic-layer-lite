"""semantic-layer-lite: approve joins and metrics once, compile every question to one safe query."""

__version__ = "0.1.0"

from .compile import CompileError, Plan, compile_query  # noqa: E402
from .datasets import make_demo_db  # noqa: E402
from .explain import explain_plan, render_plan  # noqa: E402
from .infer import infer_model  # noqa: E402
from .model import Model, load_model, save_model  # noqa: E402
from .run import run_plan, run_query  # noqa: E402

__all__ = [
    "CompileError", "Model", "Plan", "__version__", "compile_query", "explain_plan",
    "infer_model", "load_model", "make_demo_db", "render_plan", "run_plan", "run_query",
    "save_model",
]
