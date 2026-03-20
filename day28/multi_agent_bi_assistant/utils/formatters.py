from __future__ import annotations

from typing import Any


def pretty_result(result: Any) -> str:
    if isinstance(result, dict):
        return "\n".join(f"{k}: {v}" for k, v in result.items())
    return str(result)
