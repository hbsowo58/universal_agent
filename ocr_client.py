"""ocr_server.py(MCP)를 호출하는 클라이언트

retriever는 동기 코드이고 Streamlit은 입력마다 스크립트를 다시 실행하므로,
MCP 세션은 전용 스레드의 이벤트 루프에서 열고 닫는다.
"""

import json
import os
import queue
import sys
import threading
from datetime import timedelta
from pathlib import Path
from typing import Callable, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OCR_SERVER_PATH = Path(__file__).parent / "ocr_server.py"

# MCP 서버의 stderr를 모아두는 로그. 세션이 끊겼을 때 원인을 여기서 확인한다.
SERVER_LOG_PATH = Path(__file__).parent / "ocr_server.log"

# 한 번의 call_tool로 보낼 페이지 수. 서버는 이 안에서 병렬로 처리하고,
# 청크가 끝날 때마다 진행 상황을 화면에 알려줄 수 있다.
CHUNK_SIZE = 4

# OCR은 페이지당 수 초가 걸리므로 넉넉히 잡는다
CALL_TIMEOUT = timedelta(seconds=600)


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,  # 가상환경 파이썬으로 서버를 띄운다
        args=[str(OCR_SERVER_PATH)],
        cwd=str(OCR_SERVER_PATH.parent),
        # 기본값은 안전한 최소 환경만 넘겨서 OPENAI_API_KEY가 서버까지 가지 않는다.
        # 현재 환경을 그대로 물려준다.
        env=dict(os.environ),
    )


def describe_error(error: BaseException) -> str:
    """anyio가 감싼 ExceptionGroup에서 실제 예외를 꺼내 읽을 수 있게 만든다.

    그냥 두면 "unhandled errors in a TaskGroup (1 sub-exception)"처럼
    원인이 전혀 드러나지 않는 메시지만 남는다.
    """
    subs = getattr(error, "exceptions", None)
    if subs:
        return " / ".join(describe_error(sub) for sub in subs)

    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def read_server_log(max_chars: int = 1500) -> str:
    """MCP 서버가 stderr에 남긴 마지막 로그를 읽는다"""
    try:
        text = SERVER_LOG_PATH.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text[-max_chars:]


def _parse(result) -> dict:
    """MCP 도구 응답(JSON 텍스트)을 dict로 변환"""
    for item in result.content:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"success": False, "error": text}
    return {"success": False, "error": "MCP 서버가 빈 응답을 반환했습니다."}


async def _ocr_async(
    pdf_path: str,
    pages: Optional[List[int]],
    dpi: int,
    report: Callable[[int, int], None],
    errlog,
) -> List[dict]:
    async with stdio_client(_server_params(), errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if pages is None:
                probe = _parse(
                    await session.call_tool("pdf_probe", {"pdf_path": pdf_path})
                )
                if probe.get("error"):
                    raise RuntimeError(probe["error"])
                pages = probe.get("pages_without_text") or []

            if not pages:
                return []

            collected: List[dict] = []
            total = len(pages)

            for start in range(0, total, CHUNK_SIZE):
                chunk = pages[start : start + CHUNK_SIZE]

                result = _parse(
                    await session.call_tool(
                        "pdf_ocr",
                        {"pdf_path": pdf_path, "pages": chunk, "dpi": dpi},
                        read_timeout_seconds=CALL_TIMEOUT,
                    )
                )

                if not result.get("success"):
                    raise RuntimeError(result.get("error", "OCR에 실패했습니다."))

                collected.extend(result.get("pages", []))
                report(len(collected), total)

            return collected


def ocr_pdf(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    dpi: int = 200,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    """스캔 PDF를 OCR한다 (동기 래퍼).

    Args:
        pdf_path: OCR할 PDF 경로
        pages: OCR할 페이지 번호(1부터). None이면 서버가 텍스트 없는 페이지를 스스로 찾는다
        dpi: 렌더링 해상도
        progress: (완료 페이지 수, 전체 페이지 수)를 받는 콜백

    Returns:
        [{"page": int, "text": str}, ...] 페이지 오름차순
    """
    result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

    # 진행 상황은 큐로 넘겨 호출한 스레드에서 소비한다.
    # 작업 스레드에서 직접 콜백을 부르면 Streamlit UI를 건드릴 때
    # NoSessionContext(ScriptRunContext 없음)로 죽는다.
    progress_queue: "queue.Queue[tuple[int, int]]" = queue.Queue()

    def report(done: int, total: int) -> None:
        progress_queue.put((done, total))

    def worker() -> None:
        # 스레드마다 새 이벤트 루프를 만든다.
        # Windows에서 stdio 서버는 자식 프로세스로 뜨므로 Proactor 루프가 필요하다
        # (SelectorEventLoop는 subprocess를 지원하지 않는다).
        import asyncio

        if sys.platform.startswith("win"):
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
        try:
            # 서버 stderr는 파일로 받는다. Streamlit처럼 sys.stderr가
            # 실제 파일 디스크립터를 갖지 않는 환경에서도 안전하다.
            with open(SERVER_LOG_PATH, "w", encoding="utf-8") as errlog:
                pages_text = loop.run_until_complete(
                    _ocr_async(pdf_path, pages, dpi, report, errlog)
                )
            result_queue.put(("ok", pages_text))
        except BaseException as error:  # noqa: BLE001 - ExceptionGroup 포함
            result_queue.put(("error", error))
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # 작업이 끝날 때까지 진행 상황을 호출한 스레드에서 전달한다
    while thread.is_alive() or not progress_queue.empty():
        try:
            done, total = progress_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if progress is not None:
            try:
                progress(done, total)
            except Exception:
                # 진행 표시 실패가 OCR 결과를 버리게 두지 않는다
                pass

    thread.join()

    status, payload = result_queue.get()
    if status == "error":
        detail = describe_error(payload)  # type: ignore[arg-type]
        log = read_server_log()
        if log:
            detail = f"{detail}\n\n[OCR 서버 로그]\n{log}"
        raise RuntimeError(detail) from payload  # type: ignore[misc]
    return payload  # type: ignore[return-value]
