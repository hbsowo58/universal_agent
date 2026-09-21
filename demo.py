"""CLI 데모 — Streamlit 없이 그래프만 돌려본다 (영상 촬영/디버깅용)

    python demo.py "작업 폴더 파일 목록 보여줘"
    python demo.py                      # 기본 시나리오
"""

import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from make_graph import build_graph
from settings import TASK_TYPES

DEFAULT_QUERY = (
    "작업 폴더에 어떤 파일이 있는지 목록을 보여주고, "
    "report_2026H1.md 의 매출 수치를 알려줘"
)


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "demo"}}

    print("=" * 66)
    print("요청:", query)
    print("=" * 66)

    for chunk in graph.stream(
        {"messages": [HumanMessage(content=query)], "query": query}, config=config
    ):
        for node, update in chunk.items():
            print(f"\n[{node}]")
            if node == "validate":
                print("  남은 작업:", update.get("plan"))
            for message in update.get("messages") or []:
                text = str(getattr(message, "content", "")).replace("\n", " ")
                print("  ", text[:300])

    state = graph.get_state(config).values

    print("\n" + "=" * 66)
    print(state.get("final_answer", "(답변 없음)"))
    print("=" * 66)

    for key in ("web", "rag", "file"):
        results = state.get(f"{key}_results", [])
        if results:
            print(f"{TASK_TYPES[key]}: {len(results)}건")

    rejected = state.get("rejected", [])
    if rejected:
        print("\n검증 제외:")
        for reason in rejected:
            print(" -", reason)


if __name__ == "__main__":
    main()
