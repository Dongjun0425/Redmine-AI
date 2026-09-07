"""PostgreSQL(+pgvector) 접근 계층."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import psycopg
from pgvector.psycopg import register_vector

from . import config

EMBEDDING_DIM = 1536

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL,
    project_name TEXT NOT NULL,
    subject TEXT NOT NULL,
    description TEXT,
    raw_text TEXT NOT NULL,
    url TEXT NOT NULL,
    updated_on TIMESTAMPTZ NOT NULL,
    embedding VECTOR(1536)
);

CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

SET maintenance_work_mem = '128MB';

CREATE INDEX IF NOT EXISTS issues_embedding_cosine_idx
    ON issues USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL이 설정되어 있지 않습니다. .env를 확인하세요.")
    conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    try:
        register_vector(conn)
    except Exception:
        # 최초 실행 시(init_schema가 CREATE EXTENSION을 아직 안 돌렸을 때)는
        # vector 타입이 없어서 등록이 실패할 수 있다. 이 경우는 무시하고 넘어간다.
        pass
    try:
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    with get_connection() as conn:
        conn.execute(SCHEMA_SQL)


def get_last_synced_at() -> str | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM sync_state WHERE key = 'last_synced_at'"
        ).fetchone()
        return row[0] if row else None


def set_last_synced_at(conn: psycopg.Connection, value: str) -> None:
    conn.execute(
        """
        INSERT INTO sync_state (key, value) VALUES ('last_synced_at', %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """,
        (value,),
    )


def upsert_issue(conn: psycopg.Connection, issue: dict, embedding: Sequence[float]) -> None:
    conn.execute(
        """
        INSERT INTO issues (
            id, project_id, project_name, subject, description, raw_text, url, updated_on, embedding
        )
        VALUES (
            %(id)s, %(project_id)s, %(project_name)s, %(subject)s, %(description)s,
            %(raw_text)s, %(url)s, %(updated_on)s, %(embedding)s
        )
        ON CONFLICT (id) DO UPDATE SET
            project_id = EXCLUDED.project_id,
            project_name = EXCLUDED.project_name,
            subject = EXCLUDED.subject,
            description = EXCLUDED.description,
            raw_text = EXCLUDED.raw_text,
            url = EXCLUDED.url,
            updated_on = EXCLUDED.updated_on,
            embedding = EXCLUDED.embedding
        """,
        {**issue, "embedding": list(embedding)},
    )


def list_projects() -> list[dict]:
    """검색 화면의 프로젝트 선택 박스(select) 채우는 용도. 실제 저장된 이슈 기준 distinct 목록."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT project_id, project_name FROM issues ORDER BY project_name"
        ).fetchall()
        return [{"project_id": r[0], "project_name": r[1]} for r in rows]
