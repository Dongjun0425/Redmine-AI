"""Redmine 이슈를 가져와 임베딩을 생성하고 DB에 저장(동기화)한다.

이슈 개수가 많을 수 있어(사내 Redmine 기준 2만 건 이상), 임베딩은 여러 건을
한 번의 OpenAI API 호출로 묶어서(batch) 처리한다.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import db, embeddings, redmine_client

BATCH_SIZE = 50
# 한 번의 임베딩 호출에 실리는 토큰 총량 안전 마진 (모델/요금제별 요청당 토큰 한도를 넘지 않도록)
MAX_BATCH_TOKENS = 100_000


def _issue_row(issue: dict, raw_text: str) -> dict:
    return {
        "id": issue["id"],
        "project_id": issue["project"]["id"],
        "project_name": issue["project"]["name"],
        "subject": issue.get("subject", ""),
        "description": issue.get("description") or "",
        "raw_text": raw_text,
        "url": redmine_client.issue_url(issue),
        "updated_on": issue["updated_on"],
    }


def run_sync(full: bool = False) -> int:
    """full=True 면 전체 백필, False 면 마지막 동기화 이후 갱신분만 가져온다."""
    db.init_schema()
    updated_since = None if full else db.get_last_synced_at()

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    count = 0

    batch_rows: list[dict] = []
    batch_texts: list[str] = []
    batch_tokens = 0

    with db.get_connection() as conn:
        def flush():
            nonlocal batch_rows, batch_texts, batch_tokens, count
            if not batch_rows:
                return
            vectors = embeddings.embed_batch(batch_texts)
            for row, vector in zip(batch_rows, vectors):
                db.upsert_issue(conn, row, vector)
            count += len(batch_rows)
            print(f"  ...{count}건 처리중")
            batch_rows = []
            batch_texts = []
            batch_tokens = 0

        for issue in redmine_client.fetch_issues(updated_since=updated_since):
            raw_text = redmine_client.issue_to_searchable_text(issue)
            tok_len = embeddings.count_tokens(raw_text)

            if batch_rows and (
                len(batch_rows) >= BATCH_SIZE or batch_tokens + tok_len > MAX_BATCH_TOKENS
            ):
                flush()

            batch_rows.append(_issue_row(issue, raw_text))
            batch_texts.append(raw_text)
            batch_tokens += tok_len

        flush()
        db.set_last_synced_at(conn, started_at)

    return count


if __name__ == "__main__":
    import sys

    is_full = "--full" in sys.argv
    print("전체 백필 시작" if is_full else "증분 동기화 시작 (마지막 동기화 이후 변경분만)")
    n = run_sync(full=is_full)
    print(f"동기화 완료: {n}건 처리")
