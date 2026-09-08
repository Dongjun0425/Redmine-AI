"""Redmine 본인 계정(아이디/비밀번호)으로 로그인 인증을 처리한다.

흐름:
1. 사용자가 웹페이지에서 본인 Redmine 아이디/비밀번호를 입력.
2. 백엔드가 그 아이디/비밀번호로 Redmine REST API(/users/current.json)에 직접 요청해서
   실제로 로그인 가능한 계정인지 확인한다. (비밀번호는 저장하지 않고 검증에만 사용)
3. 맞으면 서명된 세션 토큰(JWT)을 발급해서 이후 요청에는 그 토큰만 쓰게 한다.
"""
from __future__ import annotations

import time

import jwt
import requests

from . import config

ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7  # 7일


def verify_redmine_credentials(username: str, password: str) -> dict | None:
    """Redmine에 실제로 로그인되는 계정인지 확인하고, 맞으면 사용자 정보를 반환한다."""
    try:
        resp = requests.get(
            f"{config.REDMINE_BASE_URL}/users/current.json",
            auth=(username, password),
            headers={"Accept": "application/json"},
            timeout=15,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return (resp.json() or {}).get("user")


def issue_session_token(user: dict) -> str:
    payload = {
        "sub": str(user["id"]),
        "login": user.get("login", ""),
        "name": f"{user.get('firstname', '')} {user.get('lastname', '')}".strip(),
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, config.require_session_secret(), algorithm=ALGORITHM)


def verify_session_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, config.require_session_secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
