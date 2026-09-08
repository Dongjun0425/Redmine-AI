"""PostgreSQL(+pgvector) 접근 계층."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from . import config

EMBEDDING_DIM = 1536

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

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

-- ILIKE '%...%' 키워드 검색이 전체 스캔이라 느렸던 부분을 trigram GIN 인덱스로 가속
CREATE INDEX IF NOT EXISTS issues_subject_trgm_idx
    ON issues USING gin (subject gin_trgm_ops);
CREATE INDEX IF NOT EXISTS issues_raw_text_trgm_idx
    ON issues USING gin (raw_text gin_trgm_ops);

-- 게시물 전체에서 "자주 같이 쓰이는 단어" 통계(예: 지원부서 <-> 참고치/상한치/하한치)를
-- 저장해서, 검색어 하나를 입력해도 실제로 같이 쓰이는 관련 단어까지 알아서 넓혀 찾게 한다.
CREATE TABLE IF NOT EXISTS term_associations (
    term TEXT NOT NULL,
    related_term TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    cooccur_count INTEGER NOT NULL,
    PRIMARY KEY (term, related_term)
);
CREATE INDEX IF NOT EXISTS term_associations_term_idx ON term_associations (term);
"""

_pool: ConnectionPool | None = None


def _configure_connection(conn: psycopg.Connection) -> None:
    try:
        register_vector(conn)
    except Exception:
        # 최초 실행 시(init_schema가 CREATE EXTENSION을 아직 안 돌렸을 때)는
        # vector 타입이 없어서 등록이 실패할 수 있다. 이 경우는 무시하고 넘어간다.
        pass


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        if not config.DATABASE_URL:
            raise RuntimeError("DATABASE_URL이 설정되어 있지 않습니다. .env를 확인하세요.")
        # 매 요청마다 새 TCP/TLS 연결을 맺느라 느렸던 부분을, 커넥션을 재사용하는 풀로 교체.
        _pool = ConnectionPool(
            config.DATABASE_URL,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True},
            configure=_configure_connection,
            open=True,
        )
    return _pool


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    with _get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


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


def get_related_terms(terms: list[str], limit_per_term: int = 5) -> list[str]:
    """게시물 전체 통계상 이 단어(들)와 자주 같이 쓰인 단어들을 가져온다."""
    if not terms:
        return []
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT related_term
            FROM (
                SELECT related_term,
                       row_number() OVER (PARTITION BY term ORDER BY score DESC) AS rn
                FROM term_associations
                WHERE term = ANY(%s)
            ) ranked
            WHERE rn <= %s
            """,
            (terms, limit_per_term),
        ).fetchall()
        return [r[0] for r in rows]
