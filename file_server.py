"""파일 관리 MCP 서버 (stdio transport) — CHAP9_MCP 패턴

작업 폴더(AGENT_WORKSPACE) 안의 파일만 다루는 도구 모음.
ocr_server.py와 같은 구조이며, file_client.py가 자식 프로세스로 띄운다.

도구
    list_files   : 작업 폴더의 파일 목록 (glob 패턴 지원)
    read_file    : 텍스트 파일 읽기
    write_file   : 텍스트 파일 쓰기 (기본은 덮어쓰기 금지)
    move_file    : 파일 이동/이름 변경
    search_files : 파일 내용에서 문자열 검색 (grep)

단독 점검: python file_server.py  (stdio 대기 상태로 멈춰 있으면 정상)
"""

import fnmatch
import os
import sys
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────
# stdio MCP에서 stdout은 JSON-RPC 전용 채널이다. 라이브러리가 stdout에
# 무언가를 찍으면 프로토콜 스트림이 오염되어 세션이
# "unhandled errors in a TaskGroup"으로 끊긴다.
# 이 서버에서 print는 절대 쓰지 않는다. 진단은 stderr로만 보낸다.
# ─────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

mcp = FastMCP(
    "FileManager",
    instructions="작업 폴더 안의 파일을 찾고 읽고 쓰는 파일 관리 어시스턴트입니다.",
)

WORKSPACE = Path(
    os.getenv("AGENT_WORKSPACE", Path(__file__).parent / "workspace")
).resolve()

MAX_READ_CHARS = 20000     # 한 파일에서 읽어올 최대 글자 수
MAX_LIST_ITEMS = 200       # 목록 응답 최대 개수
TEXT_SUFFIXES = {
    ".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml",
    ".html", ".js", ".ts", ".toml", ".ini", ".log",
}


def _safe_path(relative: str) -> Path:
    """작업 폴더를 벗어나는 경로를 차단한다 (path traversal 방어)

    `..`나 절대 경로로 폴더 밖 파일에 손대는 것을 막는다.
    에이전트가 만들어 낸 경로를 그대로 파일 시스템에 넘기므로 반드시 거친다.
    """
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    candidate = (WORKSPACE / relative).resolve()
    try:
        candidate.relative_to(WORKSPACE)
    except ValueError:
        raise ValueError(
            f"작업 폴더를 벗어난 경로입니다: {relative} "
            f"(허용 범위: {WORKSPACE})"
        )
    return candidate


def _rel(path: Path) -> str:
    """작업 폴더 기준 상대 경로 문자열 (응답에는 절대 경로를 노출하지 않는다)"""
    return path.relative_to(WORKSPACE).as_posix()


@mcp.tool()
def list_files(pattern: str = "*", subdir: str = "") -> dict:
    """작업 폴더의 파일 목록을 돌려준다.

    Args:
        pattern: 파일명 glob 패턴. 예) "*.md", "보고서*"
        subdir: 하위 폴더. 비우면 작업 폴더 전체를 재귀 탐색
    """
    try:
        root = _safe_path(subdir) if subdir else WORKSPACE
        root.mkdir(parents=True, exist_ok=True)

        items: List[dict] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if not fnmatch.fnmatch(path.name, pattern):
                continue
            stat = path.stat()
            items.append(
                {
                    "path": _rel(path),
                    "size": stat.st_size,
                    "suffix": path.suffix,
                }
            )
            if len(items) >= MAX_LIST_ITEMS:
                break

        return {"success": True, "workspace": str(WORKSPACE), "count": len(items), "files": items}
    except Exception as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}


@mcp.tool()
def read_file(path: str, max_chars: int = MAX_READ_CHARS) -> dict:
    """텍스트 파일의 내용을 읽는다.

    Args:
        path: 작업 폴더 기준 상대 경로
        max_chars: 최대 글자 수. 넘으면 잘라서 돌려준다
    """
    try:
        target = _safe_path(path)
        if not target.is_file():
            return {"success": False, "error": f"파일이 없습니다: {path}"}

        if target.suffix.lower() not in TEXT_SUFFIXES:
            return {
                "success": False,
                "error": f"텍스트 파일이 아닙니다: {path} (지원: {sorted(TEXT_SUFFIXES)})",
            }

        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars

        return {
            "success": True,
            "path": _rel(target),
            "truncated": truncated,
            "content": text[:max_chars],
        }
    except Exception as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}


@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = False) -> dict:
    """텍스트 파일을 쓴다.

    Args:
        path: 작업 폴더 기준 상대 경로
        content: 파일에 쓸 내용
        overwrite: 기존 파일을 덮어쓸지 여부. 기본은 False(덮어쓰기 금지)
    """
    try:
        target = _safe_path(path)
        existed = target.exists()

        if existed and not overwrite:
            return {
                "success": False,
                "error": f"이미 존재합니다: {path}. 덮어쓰려면 overwrite=True로 호출하세요.",
            }

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "path": _rel(target),
            "bytes": len(content.encode("utf-8")),
            "overwritten": existed,
        }
    except Exception as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}


@mcp.tool()
def move_file(src: str, dst: str, overwrite: bool = False) -> dict:
    """파일을 이동하거나 이름을 바꾼다.

    Args:
        src: 원본 경로 (작업 폴더 기준 상대 경로)
        dst: 대상 경로 (작업 폴더 기준 상대 경로)
        overwrite: 대상이 이미 있을 때 덮어쓸지 여부
    """
    try:
        source = _safe_path(src)
        target = _safe_path(dst)

        if not source.is_file():
            return {"success": False, "error": f"원본 파일이 없습니다: {src}"}
        if target.exists() and not overwrite:
            return {"success": False, "error": f"대상이 이미 존재합니다: {dst}"}

        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)

        return {"success": True, "src": _rel(source), "dst": _rel(target)}
    except Exception as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}


@mcp.tool()
def search_files(keyword: str, pattern: str = "*", max_hits: int = 30) -> dict:
    """파일 내용에서 키워드를 찾는다 (단순 grep).

    Args:
        keyword: 찾을 문자열 (대소문자 무시)
        pattern: 검색 대상 파일명 glob 패턴
        max_hits: 최대 결과 수
    """
    try:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        needle = keyword.lower()
        hits: List[dict] = []

        for path in sorted(WORKSPACE.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if not fnmatch.fnmatch(path.name, pattern):
                continue

            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            for lineno, line in enumerate(lines, start=1):
                if needle in line.lower():
                    hits.append(
                        {"path": _rel(path), "line": lineno, "text": line.strip()[:200]}
                    )
                    if len(hits) >= max_hits:
                        return {"success": True, "count": len(hits), "hits": hits}

        return {"success": True, "count": len(hits), "hits": hits}
    except Exception as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    # stdio 트랜스포트로 대기. 진단 메시지는 stderr로만.
    print(f"[file_server] workspace={WORKSPACE}", file=sys.stderr)
    mcp.run(transport="stdio")
