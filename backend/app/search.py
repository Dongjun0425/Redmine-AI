"""자연어 문장을 받아 관련 Redmine 이슈를 찾는 하이브리드 검색.

흐름:
1. 입력 문장에서 LLM으로 핵심 키워드/동의어/번역어를 추출한다.
2. 그 키워드들로 텍스트(ILIKE) 검색을 한다.
3. 입력 문장 원문을 임베딩해서 pgvector 코사인 유사도 검색을 한다.
4. 두 결과를 합쳐 점수순으로 정렬해 게시물 목록만 반환한다(대화형 답변 생성 없음).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

# openai 패키지는 chat/embeddings 서브모듈을 첫 접근 시 지연 임포트(lazy import)한다.
# 아래에서 키워드추출(chat)과 임베딩을 스레드 2개로 동시에 돌리는데, 두 서브모듈을
# 여러 스레드가 동시에 "처음" 임포트하면 파이썬 임포트 락끼리 걸려 교착상태(deadlock)가
# 날 수 있다. 그래서 프로세스 시작 시점(단일 스레드)에 미리 로드해 둔다.
import openai.resources.chat  # noqa: F401
import openai.resources.embeddings  # noqa: F401
from openai import OpenAI

from . import config, db, embeddings

CHAT_MODEL = "gpt-4o-mini"
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


KEYWORD_PROMPT = """다음은 사내 Redmine 이슈 검색을 위해 사용자가 입력한 문장이다.
이 문장에서 실제 검색에 쓸 핵심 주제어만 뽑아라.

규칙:
- "나 지금 ~ 처리중인데", "관련 게시물 보여줘" 같은 잡담/군더더기는 제외하고 핵심 주제만 남긴다.
- 같은 의미의 한국어/영어 동의어·번역어를 함께 포함한다 (예: 프린터 -> printer, 프린트, 출력, 인쇄기).
- 결과는 문자열 배열의 JSON으로만 응답한다. 다른 설명은 절대 붙이지 않는다.

문장: "{query}"
"""


def extract_keywords(query: str) -> list[str]:
    try:
        resp = _get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": KEYWORD_PROMPT.format(query=query)}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        keywords = json.loads(text)
        if isinstance(keywords, list):
            cleaned = [str(k).strip() for k in keywords if str(k).strip()]
            if cleaned:
                return cleaned
    except Exception:
        pass
    return [query.strip()]


def _keyword_search(keywords: list[str], project_id: int | None, limit: int) -> list[dict]:
    """단순 ILIKE만 쓰면 '출력'처럼 흔한 단어가 결과를 뒤덮어서, 정작 관련도 높은 글이
    LIMIT에 안 걸리고 밀려날 수 있다. 그래서 (제목 매치는 가중치 2, 본문/댓글 매치는 1)로
    키워드별 매치 점수를 합산해 점수 높은 순으로 정렬한다."""
    if not keywords:
        return []
    with db.get_connection() as conn:
        score_terms = []
        score_params: list = []
        where_terms = []
        where_params: list = []
        for kw in keywords:
            like = f"%{kw}%"
            score_terms.append(
                "(CASE WHEN subject ILIKE %s THEN 2 ELSE 0 END) + "
                "(CASE WHEN raw_text ILIKE %s THEN 1 ELSE 0 END)"
            )
            score_params.extend([like, like])
            where_terms.append("(subject ILIKE %s OR raw_text ILIKE %s)")
            where_params.extend([like, like])

        sql = (
            "SELECT id, project_id, project_name, subject, raw_text, url, "
            f"({' + '.join(score_terms)}) AS match_score "
            f"FROM issues WHERE ({' OR '.join(where_terms)})"
        )
        params = score_params + where_params
        if project_id is not None:
            sql += " AND project_id = %s"
            params.append(project_id)
        sql += " ORDER BY match_score DESC LIMIT %s"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": r[0], "project_id": r[1], "project_name": r[2],
                "subject": r[3], "raw_text": r[4], "url": r[5], "match_score": r[6],
            }
            for r in rows
        ]


def _vector_literal(vector: list[float]) -> str:
    """pgvector 텍스트 포맷("[0.1,0.2,...]")으로 직접 변환해, SQL에서 ::vector로 명시 캐스팅한다.
    (SELECT 표현식 안에서는 파라미터 타입이 자동으로 vector로 추론되지 않아 명시 캐스팅이 필요함)"""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _embedding_search(query: str, project_id: int | None, limit: int) -> list[dict]:
    vector = embeddings.embed_text(query)
    with db.get_connection() as conn:
        sql = (
            "SELECT id, project_id, project_name, subject, raw_text, url, "
            "embedding <=> %s::vector AS distance FROM issues"
        )
        params: list = [_vector_literal(vector)]
        if project_id is not None:
            sql += " WHERE project_id = %s"
            params.append(project_id)
        sql += " ORDER BY distance ASC LIMIT %s"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": r[0], "project_id": r[1], "project_name": r[2],
                "subject": r[3], "raw_text": r[4], "url": r[5], "distance": r[6],
            }
            for r in rows
        ]


def _snippet(text: str, length: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:length] + ("..." if len(text) > length else "")


def _keyword_pipeline(query: str, project_id: int | None, limit: int) -> tuple[list[str], list[dict]]:
    keywords = extract_keywords(query)
    return keywords, _keyword_search(keywords, project_id, limit)


JUDGE_MODEL = "gpt-4o-mini"
JUDGE_POOL_SIZE = 80
# 키워드/임베딩 각각의 1차 후보 검색 폭. 이게 너무 좁으면, 점수는 낮지만 실제로
# 정답인 게시물이 후보에도 못 들어가서 마지막 AI 판단 단계까지 갈 기회조차 없어진다.
RETRIEVAL_POOL_SIZE = 150

JUDGE_PROMPT = """당신은 사내 Redmine 이슈 검색을 돕는 도우미입니다.
아래는 사용자의 질문과, 후보로 뽑힌 게시물 목록(번호, 프로젝트, 제목, 요약)입니다.

같은 대상/기능이라도 게시물마다 서로 다른 용어(제품명·브랜드명·별칭·줄임말 등)로 표현되는 경우가
아주 흔합니다. 문자 그대로 일치하지 않아도, 실제로 사용자의 질문과 같은 주제/요청을 다루는
게시물이면 "관련 있음"으로 판단하세요. 반대로 겉보기엔 비슷한 단어가 섞여 있어도 실제 요청 내용이
다르면(예: 같은 장비의 전혀 다른 문제) "관련 없음"으로 제외하세요.

사용자 질문: "{query}"

후보 게시물:
{candidates_block}

정말 관련 있는 게시물의 번호만, 관련도가 높은 순서로 JSON 배열로 응답하세요. 예: [3, 1, 7]
관련 있는 게시물이 하나도 없으면 빈 배열 []을 응답하세요. 번호 외의 설명은 절대 붙이지 마세요.
"""


def _judge_relevance(query: str, candidates: list[dict]) -> list[int] | None:
    """후보 게시물들을 실제로 사용자 질문과 관련 있는지 AI가 판단해서, 관련 있는 것만
    관련도 순으로 골라낸다. 판단 자체가 실패하면(파싱 오류 등) None을 반환해서
    호출 쪽에서 기존 점수 순서로 대체할 수 있게 한다."""
    if not candidates:
        return []

    lines = []
    for idx, c in enumerate(candidates, start=1):
        lines.append(
            f"{idx}. [{c['project_name']}] {c['subject']} - {_snippet(c['raw_text'], 120)}"
        )
    candidates_block = "\n".join(lines)

    try:
        resp = _get_client().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(query=query, candidates_block=candidates_block),
                }
            ],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        order = json.loads(text)
        if not isinstance(order, list):
            return None

        indices: list[int] = []
        for n in order:
            try:
                i = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(candidates) and i not in indices:
                indices.append(i)
        return indices
    except Exception:
        return None


def search(query: str, project_id: int | None = None, limit: int = 30) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []

    # 키워드 추출(LLM 호출)+키워드 검색과, 임베딩 검색은 서로 결과를 필요로 하지 않으므로
    # 동시에 실행해서 전체 응답 시간을 줄인다(직렬로 하면 두 배 가까이 걸림).
    pool_size = max(limit * 2, RETRIEVAL_POOL_SIZE)
    with ThreadPoolExecutor(max_workers=2) as executor:
        kw_future = executor.submit(_keyword_pipeline, query, project_id, pool_size)
        emb_future = executor.submit(_embedding_search, query, project_id, pool_size)
        keywords, kw_results = kw_future.result()
        emb_results = emb_future.result()

    max_possible_score = max(len(keywords) * 3, 1)  # 키워드당 최대 2(제목)+1(본문) = 3점
    merged: dict[int, dict] = {}
    for row in kw_results:
        normalized = min(row["match_score"] / max_possible_score, 1.0)
        merged[row["id"]] = {**row, "score": normalized, "matched_by": "keyword"}
    for row in emb_results:
        similarity = 1 - row["distance"]
        if row["id"] in merged:
            merged[row["id"]]["score"] = max(merged[row["id"]]["score"], similarity)
            merged[row["id"]]["matched_by"] = "both"
        else:
            merged[row["id"]] = {**row, "score": similarity, "matched_by": "embedding"}

    ranked = sorted(merged.values(), key=lambda r: r["score"], reverse=True)[:JUDGE_POOL_SIZE]

    # 점수만으로는 "관련 주제라 걸린 것"과 "진짜 원하는 그 건"을 구분 못 하므로,
    # 후보 제목/요약을 AI에게 보여주고 실제로 관련 있는 것만 골라내게 한다.
    order = _judge_relevance(query, ranked)
    judged = ranked if order is None else [ranked[i] for i in order]
    judged = judged[:limit]

    return [
        {
            "id": r["id"],
            "subject": r["subject"],
            "snippet": _snippet(r["raw_text"]),
            "project_id": r["project_id"],
            "project_name": r["project_name"],
            "url": r["url"],
            "score": round(float(r["score"]), 4),
            "matched_by": r["matched_by"],
        }
        for r in judged
    ]
