"""Optional --live mode: map a plain question to a metric query with your own model.

Nothing here runs unless the user passes --live to the ask command. The
network modules are imported inside ask_model, never at import time. The
reply is only a suggested query: it still goes through compile_query, so it
can use approved metrics, dimensions and joins and nothing else. No test
calls ask_model.
"""
from __future__ import annotations

import json
from typing import Any

from .model import GRAINS, Model


class LiveError(Exception):
    """Raised when the live model cannot be reached or answers badly."""


def build_prompt(model: Model, question: str) -> str:
    """Describe the approved metrics and dimensions, then ask for a JSON metric query."""
    metrics = [f"- {m.name}: {m.description or m.aggregation + ' of ' + m.expression}"
               for _, m in sorted(model.metrics.items()) if m.status == "approved"]
    dims = [f"- {d.name} ({d.kind})" for _, d in sorted(model.dimensions.items())]
    return "\n".join([
        "Turn the question into a metric query. Reply with one JSON object and nothing else.",
        'Keys: "metrics" (list of names), "dimensions" (list of names), "filters" '
        '(list of {"dimension", "op", "value"}), "limit" (optional whole number).',
        "A time dimension may be written as name:grain, or a bare grain, where grain is one of "
        + ", ".join(GRAINS) + ".",
        "Allowed filter ops: =, !=, <, <=, >, >=, in.",
        "Metrics:", *metrics, "Dimensions:", *dims,
        f"Question: {question}",
    ])


def parse_reply(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LiveError("the reply holds no JSON object")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LiveError(f"the reply is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LiveError("the reply is not a JSON object")
    return data


def ask_model(model: Model, question: str, *, endpoint: str, llm: str,
              key_env: str = "SLL_API_KEY", timeout: float = 60.0) -> dict[str, Any]:
    """Send the question to a chat-completions style endpoint and return the suggested query."""
    import os
    import urllib.request

    if not endpoint.startswith(("http://", "https://")):
        raise LiveError("endpoint must start with http:// or https://")
    key = os.environ.get(key_env)
    if not key:
        raise LiveError(f"set the {key_env} environment variable to your API key")
    body = json.dumps({
        "model": llm,
        "temperature": 0,
        "messages": [{"role": "user", "content": build_prompt(model, question)}],
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
    except (OSError, KeyError, IndexError, ValueError) as exc:
        raise LiveError(f"live request failed: {exc}") from exc
    return parse_reply(text)
