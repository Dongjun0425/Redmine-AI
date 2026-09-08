"""검색 API 서버. 웹페이지/안드로이드 앱이 여기에 요청을 보낸다."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import auth, db
from . import search as search_module
from . import sync as sync_module

logger = logging.getLogger("uvicorn.error")

SYNC_INTERVAL_MINUTES = 10


def _run_incremental_sync() -> None:
    try:
        n = sync_module.run_sync(full=False)
        logger.info("[auto-sync] %d건 처리", n)
    except Exception:
        logger.exception("[auto-sync] 실패")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _run_incremental_sync,
        "interval",
        minutes=SYNC_INTERVAL_MINUTES,
        next_run_time=datetime.now() + timedelta(seconds=30),
    )
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        db.close_pool()


app = FastAPI(title="Redmine AI 검색", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 사내 MVP: 우선 전체 허용, 필요시 도메인 좁힐 수 있음
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    token = authorization.split(" ", 1)[1].strip()
    payload = auth.verify_session_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="세션이 만료되었거나 유효하지 않습니다. 다시 로그인해주세요.")
    return payload


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
def login(body: LoginRequest):
    user = auth.verify_redmine_credentials(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Redmine 아이디 또는 비밀번호가 올바르지 않습니다.")
    token = auth.issue_session_token(user)
    name = f"{user.get('firstname', '')} {user.get('lastname', '')}".strip()
    return {"token": token, "login": user.get("login", ""), "name": name}


@app.get("/projects")
def list_projects(user: dict = Depends(get_current_user)):
    return db.list_projects()


@app.get("/search")
def run_search(
    q: str = Query(..., min_length=1, description="검색어(단어든 자연어 문장이든 그대로)"),
    project_id: int | None = Query(default=None, description="특정 프로젝트로 좁히고 싶을 때"),
    limit: int = Query(default=30, ge=1, le=100),
    user: dict = Depends(get_current_user),
):
    return search_module.search(q, project_id=project_id, limit=limit)
