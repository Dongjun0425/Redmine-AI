"""검색 API 서버. 웹페이지/안드로이드 앱이 여기에 요청을 보낸다."""
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import config, db
from . import search as search_module

app = FastAPI(title="Redmine AI 검색")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 사내 MVP: 우선 전체 허용, 필요시 도메인 좁힐 수 있음
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _check_token(x_app_token: str | None) -> None:
    if config.SEARCH_APP_TOKEN and x_app_token != config.SEARCH_APP_TOKEN:
        raise HTTPException(status_code=401, detail="인증 토큰이 올바르지 않습니다.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/projects")
def list_projects(x_app_token: str | None = Header(default=None)):
    _check_token(x_app_token)
    return db.list_projects()


@app.get("/search")
def run_search(
    q: str = Query(..., min_length=1, description="검색어(단어든 자연어 문장이든 그대로)"),
    project_id: int | None = Query(default=None, description="특정 프로젝트로 좁히고 싶을 때"),
    limit: int = Query(default=30, ge=1, le=100),
    x_app_token: str | None = Header(default=None),
):
    _check_token(x_app_token)
    return search_module.search(q, project_id=project_id, limit=limit)
