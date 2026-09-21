"""그래프 상태 정의 (CHAP6 rag_agent/state.py, CHAP7 settings.py 패턴)"""

from typing import Dict, List

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """범용 멀티 에이전트 그래프의 상태

    MessagesState를 상속하므로 messages 필드(add_messages 리듀서)를 그대로 쓴다.
    """

    # --- 입력 ---
    query: str                  # 사용자의 원래 요청 (제약이 담긴 원문 그대로)
    doc_id: str                 # 업로드한 문서의 식별자 (retriever 조회 키)
    doc_name: str               # 원본 파일명

    # --- planning 결과 ---
    # plan과 queries는 항상 같은 길이의 평행 리스트다. plan[0]/queries[0]이 현재 작업.
    # 같은 축을 두 번 쓰는 계획(파일작업 → 파일작업)이 흔하므로 라벨을 키로 쓰면
    # 뒤 단계가 앞 단계의 쿼리를 덮어써 사용자 의도가 증발한다. 반드시 리스트로 둔다.
    intent: str                 # 요청 의도 한 줄 요약
    plan: List[str]             # 남은 작업 라벨 ["웹검색", "파일작업", "파일작업"]
    queries: List[str]          # 각 단계에서 쓸 쿼리 (plan과 인덱스 일치)

    # --- 라우팅 ---
    next: str                   # 슈퍼바이저가 선택한 에이전트 키 (web / rag / file)

    # --- 수집 결과 (누적) ---
    web_results: List[dict]     # [{"title", "url", "content"}]
    rag_results: List[dict]     # [{"page", "quote", "doc_name"}]
    file_results: List[dict]    # [{"action", "path", "detail"}]

    # --- 검증 (CHAP6 환각 검사 루프) ---
    last_batch: List[dict]      # 방금 수집해 아직 검증되지 않은 결과
    last_evidence: str          # 그 결과의 근거 원문
    rejected: List[str]         # 검증에서 탈락한 사유 로그
    retry_num: int              # 현재 작업의 재시도 횟수

    # --- 출력 ---
    final_answer: str
    answer_grounded: bool   # 최종 답변이 근거에 기반하는지 (CHAP6 환각 검사 결과)
