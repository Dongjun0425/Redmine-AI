"""1단계 검증용: Redmine API 키로 실제 이슈를 가져오는지 확인하는 스크립트.

사용법:
  1) backend/.env.example 을 backend/.env 로 복사
  2) .env 안의 REDMINE_API_KEY 를 본인 API 키로 채우기
     (https://issues.drbit.kr/redmine/my/account 접속 -> 오른쪽 'API 접근 키' 확인/생성)
  3) 이 폴더(backend)에서 실행:
       pip install -r requirements.txt
       python test_fetch.py
"""
from app import redmine_client

MAX_PREVIEW = 5


def main():
    print(f"Redmine 접속 시도: {redmine_client.config.REDMINE_BASE_URL}")
    count = 0
    for issue in redmine_client.fetch_issues():
        count += 1
        if count <= MAX_PREVIEW:
            subject = issue.get("subject", "(제목 없음)")
            project = issue.get("project", {}).get("name", "?")
            print(f"  [{project}] #{issue['id']} {subject}")
        if count >= 300:  # 테스트 목적이라 너무 오래 걸리지 않게 상한선
            print("  ... (300건 이상이라 테스트는 여기서 중단)")
            break

    print(f"\n총 {count}건의 이슈를 가져왔습니다. (위 {min(count, MAX_PREVIEW)}건 미리보기)")
    if count == 0:
        print("한 건도 못 가져왔다면: API 키가 맞는지, REST API가 활성화되어 있는지 확인해주세요.")


if __name__ == "__main__":
    main()
