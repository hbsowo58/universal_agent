# 🧭 범용 멀티 에이전트 — 웹 검색 · RAG · 파일 관리

하나의 자연어 요청을 **슈퍼바이저**가 작업으로 쪼개, 세 축의 에이전트에게 순서대로 맡기고,
결과를 근거와 함께 하나의 답변으로 통합한다.

CHAP6~9에서 배운 패턴을 조합했고, `quiz_generator`에서 검증된 모듈(`retriever`, `ocr_*`)을 그대로 이식했다.

## 실행

```bash
# 1) 의존성 (루트 venv 기준)
venv/Scripts/python.exe -m pip install -r universal_agent/requirements.txt

# 2) API 키
cp universal_agent/.env.example universal_agent/.env
#   OPENAI_API_KEY=sk-...       (필수)
#   TAVILY_API_KEY=tvly-...     (없으면 웹 검색 축이 자동으로 꺼진다)

# 3) 실행
venv/Scripts/python.exe -m streamlit run universal_agent/app.py
```

Streamlit 없이 그래프만 확인하려면:

```bash
cd universal_agent
../venv/Scripts/python.exe demo.py "작업 폴더 파일 목록 보여줘"
../venv/Scripts/python.exe demo_rag.py "이 문서의 리스크를 risks.md로 저장해줘"   # 문서 축
../venv/Scripts/python.exe make_graph.py      # graph.png 생성
../venv/Scripts/python.exe file_client.py     # MCP 파일 서버 단독 점검
```

## 그래프 구조

```
START → planning → supervisor ─┬→ web  ─┐
                               ├→ rag  ─┼→ validate ─┬→ (재시도: 같은 에이전트로)
                               └→ file ─┘            └→ supervisor → … → final → END
```

| 노드 | 역할 |
|---|---|
| `planning` | 요청을 읽고 `plan`(작업 라벨)과 `queries`(단계별 쿼리)를 만든다. 쓸 수 없는 축은 후보에서 제외 |
| `supervisor` | `Router` 도구로 `plan[0]`을 담당할 에이전트를 고른다. `plan`이 비면 `final` |
| `web` | Tavily로 외부 정보 검색 |
| `rag` | 업로드된 PDF를 벡터 검색 (`retriever.py`) |
| `file` | MCP 파일 서버의 도구 5종을 ReAct 루프로 호출 |
| `validate` | 검색 자료의 **관련성**을 이진 채점(`Relevance`). 탈락분은 버리고 부족하면 재시도 |
| `final` | 수집된 근거만으로 최종 답변 생성 후 **환각 검사**(`Grade`). 탈락하면 1회 재생성 |

재시도는 작업당 최대 2회(`nodes.MAX_RETRY`). 소진하면 확보한 결과만 가지고 다음 작업으로 넘어간다.

## 세 축

### 🌐 웹 검색
`TAVILY_API_KEY`가 있을 때만 켜진다. 없으면 `planning`이 후보에서 빼므로 그래프가 깨지지 않는다.

### 📄 문서 검색 (RAG)
`retriever.py` — PDF → 페이지 분해(+OCR 폴백) → 청킹 → 인메모리 벡터 색인 → 검색.
텍스트 레이어가 없는 스캔 페이지는 `ocr_server.py`(MCP)가 gpt-4o 비전으로 전사한다.

### 📁 파일 관리 (MCP)
`file_server.py`가 stdio MCP 서버로 뜨고, `file_client.py`가 자식 프로세스로 띄운다.

| 도구 | 역할 |
|---|---|
| `list_files` | 작업 폴더 파일 목록 (glob 패턴) |
| `read_file` | 텍스트 파일 읽기 |
| `write_file` | 텍스트 파일 쓰기 (기본 덮어쓰기 금지) |
| `move_file` | 이동 / 이름 변경 |
| `search_files` | 파일 내용 grep |

모든 경로는 `_safe_path()`를 거쳐 작업 폴더(`AGENT_WORKSPACE`) 밖으로 나가지 못한다.
삭제 도구는 의도적으로 넣지 않았다.

## 파일 구성

| 파일 | 내용 | 출처 |
|---|---|---|
| `app.py` | Streamlit UI (업로드 / 진행 상황 / 근거 보기 / 세션 초기화) | PART2 `app.py` |
| `state.py` | `MessagesState` 확장 상태 | CHAP6 `rag_agent/state.py` |
| `schemas.py` | `TaskPlan`, `PlanStep`, `Grade` | CHAP7 `planning_agent.py` |
| `settings.py` | 모델 팩토리와 시스템 프롬프트 | CHAP7 `settings.py` |
| `nodes.py` | planning / supervisor / web·rag·file / validate / final | CHAP6 `nodes.py`, CHAP7 `supervisor_agent.py` |
| `edges.py` | 검증 후 라우팅 | CHAP6 `edges.py` |
| `make_graph.py` | 그래프 조립 + 체크포인터 | CHAP7 `make_graph.py`, CHAP8 |
| `retriever.py` | PDF 인제스트와 검색 | `quiz_generator`에서 이식 |
| `ocr_server.py` / `ocr_client.py` | 스캔 PDF OCR MCP | `quiz_generator`에서 이식 (CHAP9) |
| `file_server.py` / `file_client.py` | 파일 관리 MCP | 신규 (CHAP9 패턴) |
| `demo.py` | CLI 데모 | — |

## 구현하면서 걸렸던 것

**1. 같은 축을 두 번 쓰는 계획에서 사용자 의도가 증발한다.**

처음에는 단계별 쿼리를 `{작업 라벨: 쿼리}` 딕셔너리로 들고 있었다.
그런데 "목록 보여주고, 그중 매출 수치 알려줘" 같은 요청은 계획이
`[파일작업, 파일작업]`으로 나온다 — **뒤 단계가 앞 단계의 쿼리를 덮어써서**
"목록 보여줘"가 사라지고 같은 작업이 두 번 돌았다.

```
plan     = ["파일작업", "파일작업"]
queries  = {"파일작업": "매출 수치 알려줘"}   ← 첫 단계 쿼리 소실
```

`plan`과 `queries`를 **인덱스가 일치하는 평행 리스트**로 바꾸고, `validate`가 둘을
항상 같이 `pop(0)` 하도록 고쳤다. CHAP11 레퍼런스가 프롬프트로 방어하던
"의도 보존" 문제가 자료구조 층위에서 똑같이 나타난 사례다.

**2. 완료 판정을 "라벨이 plan에 남아 있는가"로 하면 안 된다.**

같은 이유로, `label not in plan`은 중복 라벨을 구분하지 못한다.
`retry_num`을 기준으로 바꿨다 — supervisor가 작업을 넘길 때 0으로 초기화하고,
validate는 재시도할 때만 올린다. 따라서 `retry_num > 0`이면 아직 안 끝난 것이다.

**3. 워커는 매번 새 대화로 시작한다 — 앞 단계 결과를 넣어 주지 않으면 조용히 지어낸다.**

"meeting_notes.md를 읽고 요약해서 summary.md로 저장해줘"를 돌렸더니
최종 답변은 **성공했다고 말하는데 실제 파일에는 엉뚱한 내용**이 들어가 있었다.

```
답변:        "회의록을 요약하여 summary.md에 저장했습니다"
summary.md:  "이 파일은 작업 폴더의 파일 관리 에이전트가 수행한 작업을 요약한 것입니다…"
             ↑ 회의록이 아니라 자기 시스템 프롬프트를 요약해 버렸다
```

계획은 `[파일작업(읽기), 파일작업(요약·저장)]`인데 **2단계가 1단계의 결과를 볼 수 없었다.**
각 워커 노드는 호출될 때마다 빈 대화로 시작하기 때문이다.
읽어 온 내용이 없으니 모델은 눈앞에 있는 유일한 텍스트(시스템 프롬프트)를 요약했다.

가장 무서운 점은 **도구 호출이 전부 `success: True`였다는 것**이다.
`write_file`은 실제로 성공했다 — 내용이 틀렸을 뿐이라 검증 단계도 통과했다.

`_prior_context()`로 이전 단계의 수집 결과를 다음 워커의 입력에 주입해 해결했다.
CHAP11 레퍼런스가 `previous_result_content`로 하던 일과 같다.

**4. 검색 결과에 환각 검사를 걸면 멀쩡한 결과가 전부 탈락한다.**

웹 축을 처음 켰을 때 Tavily가 4건을 제대로 가져왔는데 **검증이 12건을 전부 탈락**시켰다.
(4건 × 재시도 3회)

```
[웹검색] 결과의 핵심 주장이 제공된 근거 자료와 일치하지 않으며, 출처와 내용이 일치하지 않음
[웹검색] 제공된 근거 자료와 결과의 핵심 주장이 일치하지 않으며…
… 12건
```

CHAP6에는 성격이 다른 평가자가 **둘** 있는데 하나로 뭉뚱그린 게 원인이었다.

| CHAP6 | 대상 | 질문 |
|---|---|---|
| `grade_documents` | 검색해 온 **자료** | 질문과 관련 있나? |
| `check_hallucinations` | 생성한 **답변** | 근거에 기반하나? |

검색 결과는 에이전트가 지어낸 주장이 아니라 **근거 자료 그 자체**다.
거기에 환각 검사를 걸면 근거를 자기 자신과 대조하는 꼴이라 질문이 성립하지 않는다.
실제로 모델은 판단할 게 없으니 엉뚱한 기준("제목과 본문이 안 맞는다")으로 떨어뜨렸다.
첫 검색 결과가 LangGraph 튜토리얼 글이었는데 본문에 예제 출력(북한 출산율 통계)이
섞여 있었던 게 빌미가 됐다.

`Relevance`(관련성)와 `Grade`(환각)를 분리하고, 거는 위치를 바꿨다.

```
validate → Relevance 로 검색 자료의 관련성만 본다
final    → Grade 로 생성된 답변이 근거에 기반하는지 본다 (탈락하면 1회 재생성)
```

**5. 플래너에게 오늘 날짜를 주지 않으면 검색어에 학습 시점 연도를 붙인다.**

"최신 동향을 찾아줘"가 `"LangGraph 멀티 에이전트 최신 동향 2023"`으로 나갔다.
사용자는 연도를 말한 적이 없다. 프롬프트에 오늘 날짜를 넣고
"사용자가 연도를 말하지 않았으면 넣지 말 것"을 명시하자 `2026`으로 바뀌었다.

**6. stdio MCP에서 stdout은 프로토콜 채널이다.**

서버가 stdout에 무언가를 찍으면 JSON-RPC 응답에 달라붙어 세션이
`unhandled errors in a TaskGroup (1 sub-exception)`으로 끊긴다 — 원인이 전혀 드러나지 않는다.
`file_server.py`에서는 `print`를 쓰지 않고 진단은 stderr로만 보낸다.
(OCR 서버는 PyMuPDF가 경고를 stdout에 찍어서 실제로 이 문제를 겪었다. `ocr_server.py` 상단 주석 참고)

**7. Windows에서 stdio MCP 서버는 자식 프로세스로 뜬다.**

`SelectorEventLoop`는 `subprocess`를 지원하지 않아 서버가 뜨지 않는다 → **Proactor 루프**를 쓴다(`file_client.py`).
또 `stdio_client`의 기본 환경은 안전한 최소 집합이라 `AGENT_WORKSPACE`/`OPENAI_API_KEY`가
서버까지 가지 않는다 → `StdioServerParameters(env=dict(os.environ))`로 현재 환경을 넘긴다.

**8. 파일 작업에는 LLM 검증을 걸지 않았다.**

도구 실행 결과의 `success` 플래그가 곧 사실이라 환각 검사를 한 번 더 돌릴 이유가 없다.
웹·문서 결과에만 `Grade` 이진 채점을 적용해 불필요한 호출을 줄였다.

다만 **3번 사례가 이 선택의 대가를 보여준다.** `success: True`는
"쓰기가 성공했다"는 뜻이지 "쓴 내용이 맞다"는 뜻이 아니다.
지금은 앞 단계 결과를 주입해 원인 쪽을 막았지만, 쓴 내용까지 검증하려면
`write_file` 뒤에 "쓴 내용이 근거와 일치하는가"를 보는 단계가 따로 필요하다.

## 한계 / 더 붙일 것

- 계획은 **순차 실행**이다. 웹과 문서를 동시에 칠 수 있는데 지금은 하나씩 돈다 (CHAP10 A2A로 분리하면 병렬화 가능).
- 벡터스토어가 인메모리라 프로세스가 죽으면 색인이 사라진다 (`Chroma(persist_directory=...)`로 교체 가능).
- 파일 에이전트의 ReAct 루프 상한이 4회(`MAX_TOOL_STEPS`)라 복잡한 파일 작업은 중간에 끊길 수 있다.
- 앞 단계 결과 주입(`_prior_context`)은 현재 `file` 노드에만 적용돼 있다. `web`/`rag`의 쿼리가 앞 단계 결과에 의존하는 경우("위 문서에서 찾은 주제로 검색")는 아직 약하다.
