from __future__ import annotations

from tools.memory_tools import read_all_preferences, read_preference, save_preference


class MemoryAgent:
    def run(self, user_id: str, query: str) -> dict:
        q = query.strip()
        if q.startswith("remember "):
            content = q.replace("remember ", "", 1).strip()
            return {"agent": "memory", **save_preference(user_id, "preference", content)}
        if q.startswith("记住"):
            content = q.replace("记住", "", 1).strip()
            return {"agent": "memory", **save_preference(user_id, "preference", content)}
        if "all memory" in q.lower() or "所有记忆" in q:
            return {"agent": "memory", **read_all_preferences(user_id)}
        if "memory" in q.lower() or "我的偏好" in q or "我之前说过" in q:
            return {"agent": "memory", **read_preference(user_id, "preference")}
        return {"agent": "memory", "ok": False, "message": "MemoryAgent could not classify this request."}
