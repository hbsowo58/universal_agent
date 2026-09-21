"""PDF를 색인한 뒤 그래프를 돌리는 데모 (문서 축 확인용)

    python demo_rag.py "이 문서에서 주요 리스크를 찾아서 risks.md로 저장해줘"
"""

import sys
from pathlib import Path
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import retriever
from make_graph import build_graph

pdf = Path("workspace/hanbit_2026H1_report.pdf")
data = pdf.read_bytes()
doc_id = retriever.make_doc_id(data)

print("색인 중...")
retriever.build_index(data, doc_id)
print("색인 완료:", retriever.get_stats(doc_id))
print()

graph = build_graph(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ragtest"}}
q = " ".join(sys.argv[1:]) or "이 문서에서 하반기 계획이 뭐야? 목표 연매출도 알려줘"

print("=" * 66); print("요청:", q); print("=" * 66)
for chunk in graph.stream(
    {"messages": [HumanMessage(content=q)], "query": q,
     "doc_id": doc_id, "doc_name": pdf.name},
    config=config,
):
    for node, update in chunk.items():
        print(f"\n[{node}]")
        if node == "validate":
            print("  남은 작업:", update.get("plan"))
        for m in (update.get("messages") or []):
            print("  ", str(getattr(m, "content", "")).replace("\n", " ")[:250])

st = graph.get_state(config).values
print("\n" + "=" * 66)
print(st.get("final_answer", "(없음)"))
print("=" * 66)
print("문서검색:", len(st.get("rag_results", [])), "건 / 파일작업:", len(st.get("file_results", [])), "건")
print("rejected:", st.get("rejected", []))
