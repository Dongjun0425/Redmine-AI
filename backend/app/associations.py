"""이슈 전체에서 "자주 같이 언급되는 단어"를 통계적으로 찾아내서 검색 키워드 확장에 쓴다.

예: "지원부서"와 "참고치"/"상한치"/"하한치"가 여러 게시물의 제목·댓글에서 계속 같이
나온다면, 그 연관성을 자동으로 학습해서 term_associations 테이블에 저장해 둔다.
사람이 일일이 "이 단어는 이 단어랑 관련 있다"를 알려줄 필요 없이, 이미 쌓인 22,000여 건의
게시물 데이터 자체에서 통계적으로 뽑아내는 방식이라 모든 주제에 똑같이 적용된다.

원리: 두 단어가 실제로 관련이 있다면 "같은 게시물 안에" 우연 이상으로 자주 함께 등장한다.
이를 PMI(점별 상호정보량)로 측정해서, 그 값이 높은 단어 쌍만 연관어로 저장한다.

한 번(또는 이슈가 많이 쌓인 뒤 가끔) 실행하면 된다: python -m app.associations
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from . import db

TOKEN_RE = re.compile(r"[가-힣]{2,8}|[A-Za-z0-9]{2,}")

# 완벽한 형태소 분석은 아니지만, 흔히 붙는 조사만 떼어내도 "지원부서가/지원부서는/지원부서"가
# 같은 단어로 잘 묶인다. 긴 조사부터 먼저 시도해야 잘못 잘리지 않는다.
_PARTICLES = sorted(
    ["에서는", "으로는", "에게서", "까지는", "에서", "에게", "으로", "부터", "까지",
     "에는", "이나", "이랑", "라서", "에도",
     "이", "가", "을", "를", "은", "는", "의", "로", "와", "과", "도", "만", "에"],
    key=len, reverse=True,
)

# 거의 모든 게시물에 등장해서 통계적으로 의미 없는 흔한 단어들.
STOPWORDS = {
    "문제", "확인", "요청", "관련", "발생", "내용", "처리", "방법", "필요", "부탁",
    "드립니다", "하는", "하고", "있는", "합니다", "됩니다", "때문", "경우", "이후",
    "이전", "지금", "현재", "그리고", "그래서", "위해", "대해", "적용", "설정",
    "변경", "추가", "수정", "삭제", "기능", "화면", "버튼", "작성", "입력", "출력",
    "저장", "조회", "환자", "진료", "거래처", "요청자", "작업", "부분", "이상",
}

MIN_DOC_FREQ = 5           # 최소 이만큼의 게시물에는 등장해야 후보 단어로 취급
MAX_DOC_FREQ_RATIO = 0.15  # 전체 게시물의 이 비율 넘게 나오면 너무 흔해서 제외
TOP_RELATED = 8            # 단어 하나당 저장할 연관어 최대 개수
MIN_COOCCUR = 5            # 최소 몇 번 이상 같이 나와야 연관어로 인정할지


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for m in TOKEN_RE.finditer(text or ""):
        w = m.group(0)
        for p in _PARTICLES:
            if w.endswith(p) and len(w) - len(p) >= 2:
                w = w[: -len(p)]
                break
        if len(w) >= 2 and w not in STOPWORDS:
            tokens.add(w)
    return tokens


def build_term_associations() -> int:
    with db.get_connection() as conn:
        rows = conn.execute("SELECT raw_text FROM issues").fetchall()

    total_docs = len(rows)
    if not total_docs:
        return 0

    doc_freq: Counter[str] = Counter()
    doc_tokens: list[set[str]] = []
    for (raw_text,) in rows:
        toks = _tokenize(raw_text)
        doc_tokens.append(toks)
        doc_freq.update(toks)

    max_df = max(int(total_docs * MAX_DOC_FREQ_RATIO), 50)
    vocab = {t for t, c in doc_freq.items() if MIN_DOC_FREQ <= c <= max_df}
    print(f"전체 {total_docs}건, 후보 단어 {len(vocab)}개")

    cooccur: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for toks in doc_tokens:
        present = [t for t in toks if t in vocab]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    to_write: list[tuple[str, str, float, int]] = []
    for term, related_counter in cooccur.items():
        pa = doc_freq[term] / total_docs
        scored = []
        for related, cnt in related_counter.items():
            if cnt < MIN_COOCCUR:
                continue
            pb = doc_freq[related] / total_docs
            pab = cnt / total_docs
            pmi = math.log(pab / (pa * pb))
            if pmi <= 0:
                continue
            # PMI만 쓰면 "우연히 드물게 몇 번 같이 나온" 노이즈가, 실제로 자주 같이 쓰이는
            # 진짜 연관어보다 점수가 높게 나오는 경향이 있다(둘 다 흔한 단어면 개별 등장
            # 확률이 커서 PMI가 낮아지기 때문). 그래서 "얼마나 자주 같이 나왔는지"(cnt)에
            # 로그 가중치를 곱해서, 증거가 충분히 쌓인 연관어가 위로 오도록 보정한다.
            weighted = pmi * math.log(cnt + 1)
            scored.append((related, weighted, cnt))
        scored.sort(key=lambda x: x[1], reverse=True)
        for related, weighted, cnt in scored[:TOP_RELATED]:
            to_write.append((term, related, weighted, cnt))

    with db.get_connection() as conn:
        conn.execute("DELETE FROM term_associations")
        if to_write:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO term_associations (term, related_term, score, cooccur_count) "
                    "VALUES (%s, %s, %s, %s)",
                    to_write,
                )

    return len(to_write)


if __name__ == "__main__":
    n = build_term_associations()
    print(f"연관어 {n}개 저장 완료")
