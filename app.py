"""범용 멀티 에이전트 — Streamlit UI (PART2 app.py 골격)

웹 검색 · 문서 RAG · 파일 관리 세 축을 슈퍼바이저가 조율한다.
"""

import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import retriever
from make_graph import build_graph
from settings import TASK_TYPES, WORKSPACE

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

st.set_page_config(page_title="범용 멀티 에이전트", page_icon="🧭", layout="wide")
st.title("🧭 웹 검색 · RAG · 파일 관리 멀티 에이전트")

# 노드 이름 -> 화면에 보여줄 라벨
NODE_LABEL = {
    "planning": "🧠 계획 수립",
    "supervisor": "🧭 슈퍼바이저",
    "web": "🌐 웹 검색",
    "rag": "📄 문서 검색",
    "file": "📁 파일 작업",
    "validate": "🔍 근거 검증",
    "final": "✅ 최종 답변",
}


# =====================================================================
# 키 확인 (st.secrets -> .env 순, PART2 패턴)
# =====================================================================

def get_key(name: str) -> str:
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, "")


openai_key = get_key("OPENAI_API_KEY")
tavily_key = get_key("TAVILY_API_KEY")

if not openai_key:
    st.error("OPENAI_API_KEY가 없습니다. `universal_agent/.env`에 넣어주세요.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_key
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key


# =====================================================================
# 세션 초기화 (CHAP8 thread_id = 세션 하나)
# =====================================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()
if "graph" not in st.session_state:
    st.session_state.graph = build_graph(checkpointer=st.session_state.checkpointer)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_id" not in st.session_state:
    st.session_state.doc_id = ""
    st.session_state.doc_name = ""


# =====================================================================
# 사이드바 — 사용 가능한 축, 문서 업로드, 작업 폴더
# =====================================================================

with st.sidebar:
    st.subheader("사용 가능한 축")
    st.write(f"🌐 웹 검색 — {'✅ 사용 가능' if tavily_key else '❌ TAVILY_API_KEY 없음'}")
    st.write(f"📄 문서 검색 — {'✅ ' + st.session_state.doc_name if st.session_state.doc_id else '⬜ 문서 미업로드'}")
    st.write(f"📁 파일 관리 — ✅ MCP 서버")

    st.divider()
    st.subheader("📄 문서 업로드 (RAG)")
    uploaded = st.file_uploader("PDF를 올리면 문서 검색 축이 켜집니다", type=["pdf"])

    if uploaded is not None:
        pdf_bytes = uploaded.getvalue()
        doc_id = retriever.make_doc_id(pdf_bytes)

        if doc_id != st.session_state.doc_id:
            with st.status("문서를 색인하는 중...", expanded=True) as status:
                def report(done: int, total: int) -> None:
                    status.update(label=f"OCR 진행 {done}/{total} 페이지")

                try:
                    retriever.build_index(pdf_bytes, doc_id, progress=report)
                    st.session_state.doc_id = doc_id
                    st.session_state.doc_name = uploaded.name
                    stats = retriever.get_stats(doc_id)
                    status.update(label=f"색인 완료 — {stats}", state="complete")
                except Exception as error:
                    status.update(label=f"색인 실패: {error}", state="error")

    st.divider()
    st.subheader("📁 작업 폴더")
    st.caption(str(WORKSPACE))

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in WORKSPACE.rglob("*") if p.is_file())
    if files:
        for path in files[:30]:
            st.write(f"• `{path.relative_to(WORKSPACE).as_posix()}`")
    else:
        st.caption("(비어 있음)")

    st.divider()
    if st.button("🔄 세션 초기화", use_container_width=True):
        # CHAP8 체크포인트 관리 — 스레드를 지우고 새로 시작
        try:
            st.session_state.checkpointer.delete_thread(st.session_state.thread_id)
        except Exception:
            pass
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


# =====================================================================
# 대화 화면
# =====================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("무엇을 도와드릴까요? (웹 · 문서 · 파일을 함께 씁니다)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        progress = st.container()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        payload = {
            "messages": [HumanMessage(content=prompt)],
            "query": prompt,
            "doc_id": st.session_state.doc_id,
            "doc_name": st.session_state.doc_name,
        }

        final_answer = ""
        try:
            # 노드가 끝날 때마다 무슨 일이 있었는지 화면에 흘려준다
            for chunk in st.session_state.graph.stream(payload, config=config):
                for node_name, update in chunk.items():
                    label = NODE_LABEL.get(node_name, node_name)

                    if node_name == "final":
                        final_answer = update.get("final_answer", "")
                        continue

                    note = ""
                    for message in update.get("messages", []) or []:
                        note = getattr(message, "content", "")
                    if node_name == "validate":
                        note = f"남은 작업: {update.get('plan', [])}"

                    progress.caption(f"{label} — {note}" if note else label)

            if final_answer:
                st.markdown(final_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )

            # 근거 확인용 — 어떤 축에서 무엇을 모았는지
            state = st.session_state.graph.get_state(config).values
            with st.expander("🔎 수집된 근거 보기"):
                st.write(
                    {
                        TASK_TYPES["web"]: state.get("web_results", []),
                        TASK_TYPES["rag"]: state.get("rag_results", []),
                        TASK_TYPES["file"]: state.get("file_results", []),
                    }
                )
                rejected = state.get("rejected", [])
                if rejected:
                    st.warning("검증에서 제외된 항목")
                    for reason in rejected:
                        st.write(f"• {reason}")

        except Exception as error:
            st.error(f"에이전트 실행 중 오류가 발생했습니다: {error}")
