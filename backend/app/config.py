import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"환경변수 {name} 이(가) 설정되어 있지 않습니다. .env 파일을 확인하세요.")
    return value


REDMINE_BASE_URL = os.environ.get("REDMINE_BASE_URL", "https://issues.drbit.kr/redmine").rstrip("/")
REDMINE_API_KEY = os.environ.get("REDMINE_API_KEY", "").strip()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
# 로그인 세션 토큰 서명용 비밀키. openssl rand -hex 32 등으로 생성한 임의의 긴 문자열을 넣는다.
SESSION_SECRET = os.environ.get("SESSION_SECRET", "").strip()


def require_session_secret() -> str:
    return _require("SESSION_SECRET")


def require_redmine_api_key() -> str:
    return _require("REDMINE_API_KEY")
