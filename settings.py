"""모델과 공통 프롬프트 설정 (CHAP7 settings.py 패턴)"""

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


DEFAULT_MODEL = "gpt-4o"
FAST_MODEL = "gpt-4o-mini"

# 이 시스템이 다루는 작업 축 세 가지
TASK_TYPES = {
    "web": "웹검색",
    "rag": "문서검색",
    "file": "파일작업",
}

# 작업 라벨 -> 노드 키 역방향 매핑
TYPE_BY_LABEL = {label: key for key, label in TASK_TYPES.items()}

# 노드 키 -> 상태에 누적할 필드명
RESULT_FIELD = {"web": "web_results", "rag": "rag_results", "file": "file_results"}

# 파일 에이전트가 건드릴 수 있는 작업 디렉터리 (이 밖으로는 못 나간다)
WORKSPACE = Path(os.getenv("AGENT_WORKSPACE", Path(__file__).parent / "workspace"))


def get_model(model_name: str = DEFAULT_MODEL, temperature: float = 0.3):
    """공통 모델 팩토리"""
    return ChatOpenAI(model=model_name, temperature=temperature)


def get_planner_prompt(available: list[str]) -> str:
    """요청을 작업 계획으로 분해하는 프롬프트"""
    return f"""
    당신은 사용자의 요청을 처리하기 위한 작업 계획을 세우는 플래너입니다.

    ## 오늘 날짜
    {date.today().isoformat()}
    "최신", "최근", "올해" 같은 말은 이 날짜를 기준으로 해석하세요.
    검색어에 연도를 임의로 붙이지 마세요. 사용자가 연도를 말하지 않았으면 넣지 않습니다.

    ## 사용할 수 있는 작업
    {chr(10).join(f"    - {label}" for label in available)}

    - 웹검색: 최신 정보, 뉴스, 동향, 외부 사실 확인이 필요할 때
    - 문서검색: 사용자가 업로드한 문서 안의 내용을 찾아야 할 때
    - 파일작업: 작업 폴더의 파일을 찾고, 읽고, 쓰고, 정리해야 할 때

    ## 규칙
    1. 요청을 처리하는 데 **실제로 필요한 작업만** 순서대로 나열하세요.
       인사나 일반 상식 질문이면 빈 목록을 반환하세요.
    2. 각 작업에는 그 단계에서 수행할 구체적인 쿼리를 함께 쓰세요.
    3. **사용자가 말한 제약을 그대로 보존하세요.**
       "하나만", "3개만", "작년 것만", "요약해서" 같은 수량·조건·형식 제약이
       쿼리에서 사라지면 하위 에이전트가 복구할 방법이 없습니다.
       나쁜 예: "보고서 폴더 파일 중 하나만 인덱싱" -> "파일을 인덱싱해줘"
       좋은 예: "보고서 폴더 파일 중 하나만 인덱싱" -> "보고서 폴더 파일 중 하나만 인덱싱해줘"
    4. 앞 단계의 결과를 뒤 단계가 써야 하면 그 의존 관계가 드러나게 쿼리를 쓰세요.
    """


def get_supervisor_prompt(current_task: str, plan: list[str], done: dict) -> str:
    """슈퍼바이저 시스템 프롬프트 (CHAP7 supervisor_agent.py 패턴)"""
    done_summary = ", ".join(
        f"{label} {count}건" for label, count in done.items() if count
    ) or "아직 없음"

    return f"""
    당신은 범용 멀티 에이전트 시스템의 슈퍼바이저입니다.
    Router 도구를 사용하여 현재 작업을 수행할 에이전트를 결정하세요.

    ## 현재 작업 (이것만 처리하세요)
    {current_task}

    ## 남은 작업 목록
    {plan}

    ## 이미 완료한 작업
    {done_summary}

    ## 팀 멤버
    - web: 웹 검색 담당. Tavily로 외부 최신 정보를 찾는다
    - rag: 문서 검색 담당. 업로드된 문서를 벡터 검색한다
    - file: 파일 관리 담당. 작업 폴더의 파일을 목록/읽기/쓰기/이동한다

    현재 작업 "{current_task}"에 해당하는 에이전트 **하나만** 호출하세요.
    """


def get_file_agent_prompt(workspace: str) -> str:
    """파일 관리 에이전트 시스템 프롬프트"""
    return f"""
    당신은 파일 관리 에이전트입니다. 주어진 도구만으로 작업 폴더를 다룹니다.

    ## 작업 폴더
    {workspace}

    ## 규칙
    1. 경로는 항상 작업 폴더 기준 **상대 경로**로 씁니다. 절대 경로나 `..`는 거부됩니다.
    2. 파일을 쓰거나 옮기기 전에 먼저 list_files 또는 read_file로 현재 상태를 확인하세요.
    3. 덮어쓰기는 사용자가 명시적으로 요청했을 때만 하세요.
    4. 삭제 도구는 없습니다. 불필요한 파일은 이동만 제안하세요.
    5. 작업을 마치면 **무엇을 했는지** 한국어로 간결히 보고하세요.
    """


def get_relevance_prompt() -> str:
    """검색 자료의 관련성 평가 프롬프트 (CHAP6 grade_documents 패턴)

    검색 결과는 에이전트가 지어낸 주장이 아니라 근거 자료 그 자체다.
    따라서 "지어냈는가"가 아니라 "질문에 쓸모 있는가"만 본다.
    """
    return """
    당신은 검색해 온 자료가 사용자의 질문에 답하는 데 쓸모가 있는지 판단하는 평가자입니다.

    자료에 질문과 관련된 키워드나 내용이 담겨 있으면 'yes'로 평가하세요.
    질문과 전혀 무관한 주제이기만 하면 'no'입니다.

    ## 주의
    - 엄격한 테스트가 아닙니다. 목표는 **명백히 무관한 자료만** 걸러내는 것입니다.
    - 자료 안에 잡음(광고, 목차, 다른 주제의 예제 출력 등)이 섞여 있어도,
      질문과 관련된 내용이 조금이라도 있으면 'yes'입니다.
    - 자료의 내용이 사실인지 판단하지 마세요. 관련성만 봅니다.
    - 제목과 본문이 달라 보인다는 이유로 'no'를 주지 마세요.

    'yes' 또는 'no'의 이진 점수와, 'no'인 경우 짧은 사유를 함께 제공하세요.
    """


def get_grader_prompt() -> str:
    """생성한 답변의 환각 검사 프롬프트 (CHAP6 check_hallucinations 패턴)

    검색 결과가 아니라 **최종 답변**에만 건다.
    """
    return """
    당신은 에이전트가 생성한 답변이 실제 근거 자료에 기반하는지 평가하는 평가자입니다.

    다음을 모두 만족하면 'yes', 하나라도 어기면 'no'로 평가하세요.
    1. 답변의 핵심 주장이 제공된 근거 자료 안에 실제로 존재하는가
    2. 근거에 없는 사실을 지어내거나 일반 상식으로 보충하지 않았는가

    엄격한 테스트일 필요는 없습니다. 목표는 근거에 없는 내용을 지어낸 답변을 걸러내는 것입니다.
    근거가 부족하다고 정직하게 밝힌 답변은 'yes'입니다.
    'yes' 또는 'no'의 이진 점수와, 'no'인 경우 짧은 사유를 함께 제공하세요.
    """


def get_final_prompt() -> str:
    """수집된 결과를 최종 답변으로 통합하는 프롬프트"""
    return """
    당신은 여러 에이전트가 수집한 결과를 종합해 사용자에게 답하는 역할입니다.

    ## 규칙
    1. **수집된 근거 안의 내용만** 사용하세요. 부족하면 부족하다고 쓰세요.
    2. 사실마다 출처를 붙이세요. 웹은 URL, 문서는 페이지, 파일은 경로.
    3. 사용자의 원래 요청에 담긴 제약(수량·조건·형식)을 지켜서 답하세요.
    4. 한국어로, 군더더기 없이 작성하세요.
    5. 수행한 작업이 없으면 그냥 질문에 직접 답하세요.
    """
