"""3단계 검증용: 로컬에서 돌고 있는 /search API를 실제로 호출해보는 스크립트."""
import sys

import requests

from app import config

BASE = "http://127.0.0.1:8000"


def run(query: str):
    print(f"\n=== 검색어: {query!r} ===")
    r = requests.get(
        f"{BASE}/search",
        params={"q": query},
        headers={"x-app-token": config.SEARCH_APP_TOKEN},
        timeout=30,
    )
    print("status:", r.status_code)
    if r.status_code != 200:
        print(r.text)
        return
    data = r.json()
    if not data:
        print("(결과 없음)")
    for item in data[:10]:
        print(f"  [{item['score']:.3f} / {item['matched_by']}] ({item['project_name']}) {item['subject']}")


if __name__ == "__main__":
    queries = sys.argv[1:] or ["프린터", "나 지금 프린터 관련 내역 처리중인데 관련 게시물 보여줘"]
    for q in queries:
        run(q)
