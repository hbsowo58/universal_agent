"""file_server.py(MCP)를 호출하는 동기 클라이언트 — CHAP9 ocr_client.py 패턴

그래프 노드는 동기 코드이고 Streamlit은 입력마다 스크립트를 다시 실행하므로,
MCP 세션은 전용 스레드의 이벤트 루프에서 열고 닫는다.
"""

import json
import os
import queue
import sys
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

FILE_SERVER_PATH = Path(__file__).parent / "file_server.py"

# MCP 서버의 stderr를 모아두는 로그. 세션이 끊겼을 때 원인을 여기서 확인한다.
SERVER_LOG_PATH = Path(__file__).parent / "file_server.log"

CALL_TIMEOUT = timedelta(seconds=60)


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,  # 가상환경 파이썬으로 서버를 띄운다
        args=[str(FILE_SERVER_PATH)],
        cwd=str(FILE_SERVER_PATH.parent),
        # 기본값은 안전한 최소 환경만 넘겨서 AGENT_WORKSPACE / OPENAI_API_KEY가
        # 서버까지 가지 않는다. 현재 환경을 그대로 물려준다.
        env=dict(os.environ),
    )


def describe_error(error: BaseException) -> str:
    """anyio가 감싼 ExceptionGroup에서 실제 예외를 꺼내 읽을 수 있게 만든다"""
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


async def _run_async(calls: List[tuple], errlog) -> List[dict]:
    """한 세션 안에서 여러 도구 호출을 순서대로 처리한다"""
    async with stdio_client(_server_params(), errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            results: List[dict] = []
            for tool_name, arguments in calls:
                raw = await session.call_tool(
                    tool_name, arguments, read_timeout_seconds=CALL_TIMEOUT
                )
                results.append(_parse(raw))
            return results


async def _list_tools_async(errlog) -> List[dict]:
    async with stdio_client(_server_params(), errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": (tool.description or "").strip().split("\n")[0],
                    "schema": tool.inputSchema,
                }
                for tool in listed.tools
            ]


def _in_thread(coro_factory):
    """전용 스레드의 이벤트 루프에서 코루틴을 돌리고 결과를 돌려준다.

    Windows에서 stdio 서버는 자식 프로세스로 뜨므로 Proactor 루프가 필요하다
    (SelectorEventLoop는 subprocess를 지원하지 않는다).
    """
    result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

    def worker() -> None:
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
                payload = loop.run_until_complete(coro_factory(errlog))
            result_queue.put(("ok", payload))
        except BaseException as error:  # noqa: BLE001 - ExceptionGroup 포함
            result_queue.put(("error", error))
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()

    status, payload = result_queue.get()
    if status == "error":
        detail = describe_error(payload)  # type: ignore[arg-type]
        log = read_server_log()
        if log:
            detail = f"{detail}\n\n[파일 서버 로그]\n{log}"
        raise RuntimeError(detail) from payload  # type: ignore[misc]
    return payload


def list_tools() -> List[dict]:
    """서버가 제공하는 도구 목록 (LLM에 바인딩할 때 쓴다)"""
    return _in_thread(_list_tools_async)


def call_tools(calls: List[tuple]) -> List[dict]:
    """도구를 순서대로 호출한다.

    Args:
        calls: [(도구명, 인자 dict), ...]

    Returns:
        각 호출의 결과 dict 목록 (호출 순서 그대로)
    """
    if not calls:
        return []
    return _in_thread(lambda errlog: _run_async(calls, errlog))


def call_tool(name: str, arguments: Dict[str, Any]) -> dict:
    """도구 하나를 호출한다"""
    return call_tools([(name, arguments)])[0]


if __name__ == "__main__":
    # 단독 점검: python file_client.py
    for tool in list_tools():
        print(f"- {tool['name']}: {tool['description']}")
    print()
    print(call_tool("list_files", {"pattern": "*"}))
