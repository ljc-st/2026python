from __future__ import annotations

from agents.supervisor_agent import SupervisorAgent
from db.init_db import init_db


def main() -> None:
    init_db()
    agent = SupervisorAgent()
    user_id = "student_001"
    print("Multi-Agent BI Assistant started. Type quit to exit.")
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        result = agent.run(user_id, query)
        print("\nAnswer:")
        print(result.get("final_answer", result))


if __name__ == "__main__":
    main()
