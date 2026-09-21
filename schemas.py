"""구조화 출력 스키마 (CHAP6 edges.py / CHAP7 planning_agent.py 패턴)"""

from typing import List, Literal

from pydantic import BaseModel, Field


# ===== 작업 계획 =====

class PlanStep(BaseModel):
    """계획의 한 단계"""

    task: Literal["웹검색", "문서검색", "파일작업"] = Field(
        description="이 단계에서 수행할 작업의 종류"
    )
    query: str = Field(
        description=(
            "이 단계에서 수행할 구체적인 쿼리. "
            "사용자가 말한 수량·조건·형식 제약을 반드시 그대로 포함할 것"
        )
    )


class TaskPlan(BaseModel):
    """사용자 요청을 분해한 작업 계획"""

    intent: str = Field(description="사용자 요청의 의도를 한 문장으로 요약")
    steps: List[PlanStep] = Field(
        description="수행할 작업 단계. 필요 없으면 빈 목록", default_factory=list
    )


# ===== 검증 스키마 =====
#
# CHAP6에는 성격이 다른 평가자가 둘 있다. 섞어 쓰면 안 된다.
#   Relevance : 검색해 온 자료가 질문과 관련 있는가      (grade_documents)
#   Grade     : 생성한 답변이 근거에 기반하는가          (check_hallucinations)
#
# 검색 결과는 에이전트가 지어낸 주장이 아니라 근거 그 자체이므로
# 환각 검사(Grade)를 걸 대상이 아니다. 관련성(Relevance)만 본다.


class Relevance(BaseModel):
    """검색해 온 자료가 질문과 관련 있는지 판단하는 이진 점수 (CHAP6 grade_documents)"""

    binary_score: Literal["yes", "no"] = Field(
        description="자료가 질문에 답하는 데 도움이 되면 'yes', 무관하면 'no'"
    )
    reason: str = Field(description="'no'인 경우의 사유. 'yes'면 빈 문자열")


class Grade(BaseModel):
    """생성한 답변이 근거 자료에 기반하는지 판단하는 이진 점수 (CHAP6 check_hallucinations)"""

    binary_score: Literal["yes", "no"] = Field(
        description="답변이 근거에 기반하면 'yes', 지어낸 내용이 있으면 'no'"
    )
    reason: str = Field(description="'no'인 경우의 사유. 'yes'면 빈 문자열")
