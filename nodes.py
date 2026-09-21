"""그래프 노드 정의

- planning_node   : 요청을 작업 계획으로 분해 (CHAP7 planning_agent.py 패턴)
- supervisor_node : Router 도구로 담당 에이전트 선택 (CHAP7 supervisor_agent.py 패턴)
- web / rag / file: 각 축의 수집 에이전트 (CHAP6 rag_agent/nodes.py 패턴)
- validate_node   : 수집 결과의 근거 검증 (CHAP6 check_hallucinations 패턴)
- final_node      : 수집된 근거를 하나의 답변으로 통합
"""

import json
import os
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

import file_client
from retriever import has_index, retrieve_context
from schemas import Grade, Relevance, TaskPlan
from settings import (
    FAST_MODEL,
    RESULT_FIELD,
    TASK_TYPES,
    TYPE_BY_LABEL,
    WORKSPACE,
    get_file_agent_prompt,
    get_final_prompt,
    get_grader_prompt,
    get_model,
    get_relevance_prompt,
    get_planner_prompt,
    get_supervisor_prompt,
)
from state import AgentState

MAX_RETRY = 2          # 작업당 재시도 상한 (CHAP6 retry_num 패턴)
MAX_TOOL_STEPS = 4     # 파일 에이전트의 도구 호출 왕복 상한


class Router(TypedDict):
    """작업을 수행할 에이전트를 라우팅합니다."""

    next: Literal["web", "rag", "file"]


# =====================================================================
# 1) Planning — 요청을 작업 계획으로 분해
# =====================================================================

def planning_node(state: AgentState) -> Command:
    """사용자 요청을 읽고 어떤 축을 어떤 순서로 쓸지 정한다"""

    last = state["messages"][-1] if state.get("messages") else None
    query = state.get("query") or getattr(last, "content", "") or ""
    query = str(query).strip()

    # 쓸 수 있는 축만 후보로 올린다 (키가 없거나 문서가 없으면 제외)
    available = []
    if os.getenv("TAVILY_API_KEY"):
        available.append(TASK_TYPES["web"])
    if state.get("doc_id") and has_index(state["doc_id"]):
        available.append(TASK_TYPES["rag"])
    available.append(TASK_TYPES["file"])

    planner_prompt = ChatPromptTemplate.from_messages(
        [("system", get_planner_prompt(available)), ("user", "사용자 요청:\n{query}")]
    )
    planner = planner_prompt | get_model().with_structured_output(TaskPlan)
    result = planner.invoke({"query": query})

    # 쓸 수 없는 축이 계획에 섞여 들어오면 여기서 걸러낸다
    steps = [step for step in result.steps if step.task in available]

    # plan과 queries는 인덱스가 일치하는 평행 리스트. 같은 축이 두 번 나와도
    # 각 단계가 자기 쿼리를 그대로 들고 간다 (라벨 키로 묶으면 덮어써진다).
    plan = [step.task for step in steps]
    queries = [step.query for step in steps]

    message = AIMessage(
        content=(
            f"**의도**: {result.intent}\n\n"
            f"**작업 계획**: {' → '.join(plan) if plan else '외부 작업 없이 바로 답변'}"
        ),
        name="planning",
    )

    return Command(
        goto="supervisor" if plan else "final",
        update={
            "query": query,
            "intent": result.intent,
            "plan": plan,
            "queries": queries,
            "messages": [message],
        },
    )


# =====================================================================
# 2) Supervisor — 남은 작업을 담당 에이전트에게 넘긴다
# =====================================================================

def supervisor_node(state: AgentState) -> Command[Literal["web", "rag", "file", "final"]]:
    """남은 작업이 있으면 담당 에이전트로, 없으면 최종 통합으로"""

    plan = state.get("plan", [])

    # [1] 남은 작업이 없으면 통합 단계로
    if not plan:
        done = _done_counts(state)
        summary = ", ".join(f"{label} {n}건" for label, n in done.items() if n) or "수집 결과 없음"
        return Command(
            goto="final",
            update={
                "messages": [
                    AIMessage(content=f"수집을 마쳤습니다. ({summary})", name="supervisor")
                ]
            },
        )

    # [2] Router 도구로 담당 에이전트 결정
    current_task = plan[0]

    llm = get_model(temperature=0)
    response = llm.bind_tools([Router]).invoke(
        [
            {
                "role": "system",
                "content": get_supervisor_prompt(current_task, plan, _done_counts(state)),
            }
        ]
    )

    if getattr(response, "tool_calls", None):
        goto = response.tool_calls[0]["args"]["next"]
    else:
        # [3] 도구를 호출하지 않은 경우엔 계획을 그대로 따른다 (안전장치)
        goto = TYPE_BY_LABEL[current_task]

    return Command(
        goto=goto,
        update={
            "next": goto,
            "retry_num": 0,
            "messages": [AIMessage(content=f"'{current_task}'을(를) 시작합니다.", name="supervisor")],
        },
    )


def _done_counts(state: AgentState) -> dict:
    return {
        TASK_TYPES["web"]: len(state.get("web_results", [])),
        TASK_TYPES["rag"]: len(state.get("rag_results", [])),
        TASK_TYPES["file"]: len(state.get("file_results", [])),
    }


def _current_query(state: AgentState, type_key: str) -> str:
    """이 단계에서 쓸 쿼리. 슈퍼바이저가 항상 plan[0]을 처리하므로 queries[0]이 짝이다."""
    queries = state.get("queries") or []
    return (queries[0] if queries else "") or state.get("query", "")


def _prior_context(state: AgentState, max_chars: int = 5000) -> str:
    """앞 단계들이 수집한 결과를 요약해 돌려준다.

    각 워커 노드는 매번 새 대화로 시작하므로, 이걸 넣어 주지 않으면
    2단계가 1단계의 결과를 전혀 볼 수 없다.
    "읽어서 → 요약해 저장해줘" 같은 요청이 조용히 엉뚱한 내용을 쓰는 원인이 된다.
    """
    parts: list[str] = []

    for item in state.get("rag_results", []):
        parts.append(f"[문서검색] {item.get('query', '')}\n{item.get('quote', '')}")

    for item in state.get("web_results", []):
        parts.append(f"[웹검색] {item.get('title', '')} ({item.get('url', '')})\n{item.get('content', '')}")

    for item in state.get("file_results", []):
        result = item.get("result") or {}
        # 읽어 온 내용이 있으면 그게 다음 단계가 실제로 써야 할 재료다
        detail = result.get("content") or json.dumps(result, ensure_ascii=False)
        parts.append(f"[파일작업] {item.get('action')}({json.dumps(item.get('args', {}), ensure_ascii=False)})\n{detail}")

    if not parts:
        return ""

    return "\n\n---\n\n".join(parts)[:max_chars]


# =====================================================================
# 3) 수집 에이전트 3종
# =====================================================================

def web_node(state: AgentState) -> Command[Literal["validate"]]:
    """Tavily로 웹을 검색한다 (AGENTSTUDY/app.py research_node 패턴)"""

    from langchain_tavily import TavilySearch

    query = _current_query(state, "web")

    try:
        tool = TavilySearch(max_results=4, tavily_api_key=os.getenv("TAVILY_API_KEY"))
        raw = tool.invoke(query)
    except Exception as error:
        return _failed(state, "web", f"웹 검색 실패: {error}")

    # langchain-tavily는 버전에 따라 dict 또는 list를 돌려준다
    hits = raw.get("results", []) if isinstance(raw, dict) else (raw or [])

    batch = [
        {
            "title": hit.get("title", ""),
            "url": hit.get("url", ""),
            "content": (hit.get("content") or "")[:1200],
        }
        for hit in hits
        if isinstance(hit, dict)
    ]

    evidence = "\n\n".join(
        f"[{item['title']}] {item['url']}\n{item['content']}" for item in batch
    )

    return Command(
        goto="validate",
        update={
            "last_batch": batch,
            "last_evidence": evidence,
            "messages": [
                AIMessage(content=f"웹에서 {len(batch)}건을 찾았습니다: {query}", name="web")
            ],
        },
    )


def rag_node(state: AgentState) -> Command[Literal["validate"]]:
    """업로드된 문서를 벡터 검색한다 (CHAP6 retrieve 패턴)"""

    query = _current_query(state, "rag")
    doc_id = state.get("doc_id", "")

    if not doc_id or not has_index(doc_id):
        return _failed(state, "rag", "색인된 문서가 없습니다. 먼저 문서를 업로드하세요.")

    try:
        context = retrieve_context(doc_id, [query], k=4)
    except Exception as error:
        return _failed(state, "rag", f"문서 검색 실패: {error}")

    batch = [
        {
            "doc_name": state.get("doc_name", ""),
            "query": query,
            "quote": context[:4000],
        }
    ]

    return Command(
        goto="validate",
        update={
            "last_batch": batch,
            "last_evidence": context,
            "messages": [
                AIMessage(content=f"문서에서 관련 대목을 찾았습니다: {query}", name="rag")
            ],
        },
    )


def file_node(state: AgentState) -> Command[Literal["validate"]]:
    """MCP 파일 서버의 도구를 써서 작업 폴더를 다룬다 (CHAP9 MCP 패턴)"""

    query = _current_query(state, "file")

    try:
        tools = file_client.list_tools()
    except Exception as error:
        return _failed(state, "file", f"파일 MCP 서버를 띄우지 못했습니다: {error}")

    # MCP 도구 스키마를 OpenAI function 형식으로 변환해 모델에 바인딩한다
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["schema"],
            },
        }
        for tool in tools
    ]

    llm = get_model(temperature=0).bind_tools(openai_tools)

    # 앞 단계 결과를 넣어 준다. 없으면 이 노드는 1단계에서 읽은 내용을 볼 수 없어
    # "읽은 내용을 요약해 저장" 같은 요청에서 근거 없는 내용을 써 버린다.
    user_content = query
    prior = _prior_context(state)
    if prior:
        user_content = (
            f"## 앞 단계에서 이미 수집한 내용\n{prior}\n\n"
            f"## 지금 할 일\n{query}\n\n"
            "앞 단계에서 이미 읽어 온 내용이 있으면 그것을 재료로 쓰세요. "
            "같은 파일을 다시 읽을 필요는 없습니다. "
            "위에 없는 내용은 지어내지 말고, 필요하면 도구로 직접 확인하세요."
        )

    conversation = [
        {"role": "system", "content": get_file_agent_prompt(str(WORKSPACE))},
        {"role": "user", "content": user_content},
    ]

    batch: list[dict] = []

    # 도구 호출 -> 결과 -> 재호출 루프 (상한 MAX_TOOL_STEPS)
    for _ in range(MAX_TOOL_STEPS):
        response = llm.invoke(conversation)
        conversation.append(response)

        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            break

        try:
            outputs = file_client.call_tools([(c["name"], c["args"]) for c in calls])
        except Exception as error:
            return _failed(state, "file", f"파일 도구 호출 실패: {error}")

        for call, output in zip(calls, outputs):
            batch.append({"action": call["name"], "args": call["args"], "result": output})
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(output, ensure_ascii=False)[:4000],
                }
            )

    report = getattr(conversation[-1], "content", "") if conversation else ""
    evidence = json.dumps(batch, ensure_ascii=False, indent=2)[:6000]

    return Command(
        goto="validate",
        update={
            "last_batch": batch,
            "last_evidence": evidence,
            "messages": [
                AIMessage(
                    content=report or f"파일 작업 {len(batch)}건을 수행했습니다.",
                    name="file",
                )
            ],
        },
    )


def _failed(state: AgentState, type_key: str, reason: str) -> Command:
    """축 하나가 실패해도 그래프 전체를 죽이지 않고 사유만 남기고 넘어간다"""
    return Command(
        goto="validate",
        update={
            "last_batch": [],
            "last_evidence": "",
            "rejected": state.get("rejected", []) + [f"[{TASK_TYPES[type_key]}] {reason}"],
            "messages": [AIMessage(content=reason, name=type_key)],
        },
    )


# =====================================================================
# 4) Validate — 근거 검증과 재시도 (CHAP6 check_hallucinations 패턴)
# =====================================================================

def validate_node(state: AgentState) -> dict:
    """수집 결과가 근거에 기반하는지 확인하고, 통과분만 상태에 누적한다"""

    type_key = state.get("next", "web")
    label = TASK_TYPES[type_key]
    field = RESULT_FIELD[type_key]

    batch = state.get("last_batch", [])
    evidence = state.get("last_evidence", "")
    plan = list(state.get("plan", []))
    queries = list(state.get("queries", []))
    rejected = list(state.get("rejected", []))
    retry_num = state.get("retry_num", 0)

    def consume() -> dict:
        """현재 단계를 큐에서 빼낸다. plan과 queries를 항상 같이 움직인다."""
        return {"plan": plan[1:], "queries": queries[1:]}

    # [1] 수집이 비었으면 재시도 여유를 보고 판단한다
    if not batch:
        if retry_num < MAX_RETRY:
            return {"retry_num": retry_num + 1}
        return {**consume(), "retry_num": 0, "rejected": rejected}

    # [2] 파일 작업은 도구 실행 결과가 곧 사실이므로 LLM 검증을 걸지 않는다.
    #     success 플래그로 이미 판별되므로 불필요한 호출을 줄인다.
    if type_key == "file":
        passed = []
        for item in batch:
            result = item.get("result") or {}
            if result.get("success"):
                passed.append(item)
            else:
                rejected.append(f"[{label}] {item.get('action')}: {result.get('error')}")
    else:
        # 웹·문서 결과는 '지어낸 주장'이 아니라 근거 자료 그 자체다.
        # 따라서 환각 검사가 아니라 **관련성**만 본다 (CHAP6 grade_documents).
        # 여기에 환각 검사를 걸면 근거를 근거와 대조하는 꼴이라 멀쩡한 결과가 전부 탈락한다.
        query = _current_query(state, type_key)
        grader = get_model(FAST_MODEL, temperature=0).with_structured_output(Relevance)
        passed = []
        for item in batch:
            material = json.dumps(item, ensure_ascii=False)[:3000]
            try:
                verdict = grader.invoke(
                    [
                        {"role": "system", "content": get_relevance_prompt()},
                        {
                            "role": "user",
                            "content": f"질문:\n{query}\n\n검색해 온 자료:\n{material}",
                        },
                    ]
                )
            except Exception:
                passed.append(item)  # 평가 호출 실패가 수집 결과를 버리게 두지 않는다
                continue

            if verdict.binary_score == "yes":
                passed.append(item)
            else:
                rejected.append(f"[{label}] 관련성 없음 — {verdict.reason}")

    # [3] 전부 탈락했고 재시도 여유가 있으면 같은 에이전트로 되돌아간다
    if not passed and retry_num < MAX_RETRY:
        return {"retry_num": retry_num + 1, "rejected": rejected}

    return {
        **consume(),
        field: state.get(field, []) + passed,
        "retry_num": 0,
        "rejected": rejected,
        "last_batch": [],
    }


# =====================================================================
# 5) Final — 수집한 근거를 하나의 답변으로 통합
# =====================================================================

def final_node(state: AgentState) -> dict:
    """모든 축의 결과를 모아 최종 답변을 만든다"""

    collected = {
        TASK_TYPES["web"]: state.get("web_results", []),
        TASK_TYPES["rag"]: state.get("rag_results", []),
        TASK_TYPES["file"]: state.get("file_results", []),
    }
    filled = {k: v for k, v in collected.items() if v}
    evidence = json.dumps(filled, ensure_ascii=False, indent=2)[:12000]

    llm = get_model()
    user_content = (
        f"사용자 요청:\n{state.get('query', '')}\n\n"
        f"수집된 근거:\n{evidence if filled else '(없음)'}"
    )

    response = llm.invoke(
        [
            {"role": "system", "content": get_final_prompt()},
            {"role": "user", "content": user_content},
        ]
    )
    answer = response.content

    # 환각 검사는 **여기서** 건다 (CHAP6 check_hallucinations).
    # 검사 대상은 검색 결과가 아니라 방금 생성한 답변이다.
    grounded = True
    if filled:
        grader = get_model(FAST_MODEL, temperature=0).with_structured_output(Grade)
        try:
            grade = grader.invoke(
                [
                    {"role": "system", "content": get_grader_prompt()},
                    {
                        "role": "user",
                        "content": f"근거 자료:\n{evidence}\n\n생성된 답변:\n{answer}",
                    },
                ]
            )
            grounded = grade.binary_score == "yes"
        except Exception:
            grounded = True  # 검사 실패가 답변을 버리게 두지 않는다

        if not grounded:
            # 한 번만 더, 근거 밖으로 나가지 말라고 못박아 다시 생성한다
            retry = llm.invoke(
                [
                    {"role": "system", "content": get_final_prompt()},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            f"위 답변에 근거 없는 내용이 있습니다: {grade.reason}\n"
                            "근거 자료에 실제로 있는 내용만으로 다시 작성하세요. "
                            "근거가 부족한 부분은 부족하다고 밝히세요."
                        ),
                    },
                ]
            )
            answer = retry.content

    return {
        "final_answer": answer,
        "answer_grounded": grounded,
        "messages": [AIMessage(content=answer, name="final")],
    }
