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
SEARCH_APP_TOKEN = os.environ.get("SEARCH_APP_TOKEN", "").strip()
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


def require_redmine_api_key() -> str:
    return _require("REDMINE_API_KEY")
