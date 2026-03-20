from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import MEMORY_PATH

MEMORY_FILE = Path(MEMORY_PATH)


def _ensure_parent() -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_memory() -> dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(data: dict[str, Any]) -> None:
    _ensure_parent()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_user_memory(user_id: str, key: str, value: str) -> None:
    data = load_memory()
    user_obj = data.setdefault(user_id, {})
    user_obj[key] = value
    save_memory(data)


def get_user_memory(user_id: str, key: str) -> str | None:
    data = load_memory()
    return data.get(user_id, {}).get(key)


def get_all_user_memory(user_id: str) -> dict[str, Any]:
    data = load_memory()
    return data.get(user_id, {})
