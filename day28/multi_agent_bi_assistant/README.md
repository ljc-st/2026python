# Multi-Agent BI Assistant

A GitHub-ready demo project for interview / graduate re-exam use.

## Features
- Multi-agent architecture with a supervisor
- SQL agent for structured data
- RAG agent for policy/document retrieval
- Analysis agent for trend summaries and chart generation
- Memory agent for long-term user preferences
- FastAPI endpoint and CLI mode

## Project structure
```text
multi_agent_bi_assistant/
├── agents/
├── config/
├── data/
├── db/
├── memory/
├── rag/
├── tools/
├── app.py
├── cli.py
├── requirements.txt
└── README.md
```

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
python db/init_db.py
uvicorn app:app --reload
```

Open:
- `GET /` health check
- `POST /ask` main endpoint

### Example request
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"user_id": "student_001", "query": "Analyze sales trend and tell me the refund policy"}'
```

## CLI mode
```bash
python cli.py
```

## Suggested GitHub repo description
> A LangGraph-based multi-agent BI assistant with SQL, RAG, analysis, and memory modules.

## Suggested interview wording
This project demonstrates how to orchestrate specialized agents with LangGraph rather than relying on a single monolithic LLM call.

## Notes
- The current supervisor uses deterministic routing for stability and interview clarity.
- You can later upgrade routing to an LLM-based router.
- Long-term memory is stored in JSON for demo simplicity; Redis/PostgreSQL can replace it later.
