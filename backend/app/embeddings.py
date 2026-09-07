"""OpenAI 임베딩(텍스트 -> 벡터) 헬퍼. 여러 건을 한 번의 API 호출로 묶어서 처리한다."""
from __future__ import annotations

import time

import tiktoken
from openai import OpenAI, RateLimitError

from . import config

EMBEDDING_MODEL = "text-embedding-3-small"
# text-embedding-3-* 모델의 입력 한도는 항목당 8192 토큰. 한글은 글자당 토큰 수가 들쭉날쭉해서
# 글자 수가 아니라 실제 토큰 수 기준으로 잘라야 안전하다.
MAX_TOKENS_PER_ITEM = 8000
_encoding = tiktoken.get_encoding("cl100k_base")

# 계정 등급이 낮으면 분당 요청수(RPM) 한도가 낮아서(예: 100 RPM), 매 호출 사이에
# 살짝 쉬어가며 호출한다. 그래도 순간적으로 걸리면 재시도(지수 백오프)로 넘어간다.
MIN_INTERVAL_SEC = 0.5
MAX_RETRIES = 6
_last_call_at = 0.0

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env를 확인하세요.")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text or ""))


def _prepare(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "(내용 없음)"
    tokens = _encoding.encode(text)
    if len(tokens) > MAX_TOKENS_PER_ITEM:
        return _encoding.decode(tokens[:MAX_TOKENS_PER_ITEM])
    return text


def embed_batch(texts: list[str]) -> list[list[float]]:
    """여러 텍스트를 한 번의 API 호출로 임베딩한다. 순서는 입력 순서와 동일하게 반환됨."""
    global _last_call_at

    prepared = [_prepare(t) for t in texts]

    for attempt in range(MAX_RETRIES):
        wait = MIN_INTERVAL_SEC - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=prepared)
            _last_call_at = time.monotonic()
            return [d.embedding for d in resp.data]
        except RateLimitError:
            _last_call_at = time.monotonic()
            backoff = 2 ** attempt
            print(f"  (rate limit, {backoff}초 대기 후 재시도)")
            time.sleep(backoff)

    raise RuntimeError("OpenAI 임베딩 요청이 반복적으로 rate limit에 걸려 중단했습니다.")


def embed_text(text: str) -> list[float]:
    return embed_batch([text])[0]
