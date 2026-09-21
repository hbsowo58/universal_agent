"""조건부 엣지 (CHAP6 rag_agent/edges.py 패턴)

노드가 갱신해 둔 상태를 읽어 다음에 갈 노드 이름만 돌려준다. 상태는 바꾸지 않는다.
"""

from settings import TASK_TYPES
from state import AgentState


def decide_after_validate(state: AgentState) -> str:
    """검증 후 라우팅

    판단 기준은 retry_num이다.
    - supervisor가 작업을 넘길 때 retry_num을 0으로 초기화한다.
    - validate_node는 재시도가 필요할 때만 retry_num을 올리고, 작업을 마쳤으면 0으로 되돌린다.
    따라서 retry_num > 0 이면 아직 이 작업이 끝나지 않은 것이다.

    plan에 같은 라벨이 두 번 들어올 수 있으므로(파일작업 → 파일작업)
    "라벨이 plan에 남아 있는가"로는 완료 여부를 구분할 수 없다.
    """

    type_key = state.get("next", "web")
    label = TASK_TYPES[type_key]
    retry_num = state.get("retry_num", 0)

    if retry_num > 0:
        print(f"---DECISION: {label} 재시도 (retry={retry_num})---")
        return type_key

    print(f"---DECISION: {label} 완료, SUPERVISOR로 복귀---")
    return "supervisor"
