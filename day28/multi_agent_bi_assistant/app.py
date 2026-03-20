from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from agents.supervisor_agent import SupervisorAgent
from db.init_db import init_db

supervisor: SupervisorAgent | None = None


class AskRequest(BaseModel):
    user_id: str = "student_001"
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global supervisor
    init_db()
    supervisor = SupervisorAgent()
    yield


app = FastAPI(title="Multi-Agent BI Assistant", lifespan=lifespan)


@app.get("/")
def root() -> dict:
    return {"ok": True, "message": "Multi-Agent BI Assistant is running."}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    assert supervisor is not None
    return supervisor.run(req.user_id, req.query)
