from __future__ import annotations

from memory.long_term import get_all_user_memory, get_user_memory, set_user_memory


def save_preference(user_id: str, key: str, value: str) -> dict:
    set_user_memory(user_id, key, value)
    return {"ok": True, "message": f"Saved memory for {user_id}: {key}={value}"}


def read_preference(user_id: str, key: str) -> dict:
    value = get_user_memory(user_id, key)
    if value is None:
        return {"ok": False, "message": "No memory found for that key."}
    return {"ok": True, "key": key, "value": value}


def read_all_preferences(user_id: str) -> dict:
    return {"ok": True, "memories": get_all_user_memory(user_id)}
