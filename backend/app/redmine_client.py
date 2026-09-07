"""Redmine REST API에서 이슈(게시물)를 가져오는 클라이언트.

참고: https://www.redmine.org/projects/redmine/wiki/Rest_Issues
"""
from __future__ import annotations

from typing import Iterator

import requests

from . import config

PAGE_SIZE = 100
TIMEOUT = 30


def _headers() -> dict:
    return {
        "X-Redmine-API-Key": config.require_redmine_api_key(),
        "Accept": "application/json",
    }


def fetch_issues(updated_since: str | None = None) -> Iterator[dict]:
    """이슈 목록을 페이지 단위로 가져와 하나씩 yield 한다.

    updated_since: "YYYY-MM-DDTHH:MM:SSZ" 형식이면 그 시각 이후 갱신된 이슈만 가져온다(증분 동기화용).
    None이면 전체 이슈를 가져온다(최초 백필용).
    """
    offset = 0
    while True:
        params = {
            "status_id": "*",  # 완료/종료된 이슈도 포함
            "limit": PAGE_SIZE,
            "offset": offset,
            "sort": "updated_on:asc",
            "include": "journals",  # 댓글(작업 내역) 포함
        }
        if updated_since:
            params["updated_on"] = f">={updated_since}"

        resp = requests.get(
            f"{config.REDMINE_BASE_URL}/issues.json",
            headers=_headers(),
            params=params,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        issues = data.get("issues", [])
        for issue in issues:
            yield issue

        total_count = data.get("total_count", 0)
        offset += len(issues)
        if not issues or offset >= total_count:
            break


def issue_to_searchable_text(issue: dict) -> str:
    """제목 + 본문 + 댓글(작업 내역)을 하나의 검색용 텍스트로 합친다."""
    parts = [issue.get("subject", ""), issue.get("description", "") or ""]
    for journal in issue.get("journals", []):
        note = journal.get("notes")
        if note:
            parts.append(note)
    return "\n".join(p for p in parts if p)


def issue_url(issue: dict) -> str:
    return f"{config.REDMINE_BASE_URL}/issues/{issue['id']}"
