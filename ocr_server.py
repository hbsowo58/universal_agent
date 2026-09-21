"""스캔 PDF OCR MCP 서버 (stdio transport) — CHAP9_MCP 패턴

PyMuPDF로 페이지를 이미지로 렌더링한 뒤, gpt-4o 비전 모델에 넘겨 텍스트를 전사한다.
Tesseract 같은 별도 설치가 필요 없고, 이미 쓰고 있는 OPENAI_API_KEY를 그대로 재사용한다.

도구
    pdf_probe    : 페이지별 텍스트 레이어 유무를 조사한다 (OCR이 필요한지 판단용)
    pdf_ocr      : 지정한 페이지들을 OCR해 텍스트를 돌려준다

직접 실행할 일은 없고 ocr_client.py가 자식 프로세스로 띄운다.
단독 점검이 필요하면: python ocr_server.py  (stdio 대기 상태로 멈춰 있으면 정상)
"""

import asyncio
import base64
import os
import sys
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────
# stdio MCP 서버에서 stdout은 JSON-RPC 프로토콜 전용 채널이다.
# PyMuPDF는 기본적으로 경고를 stdout에 찍기 때문에(폰트 인코딩 미지원,
# xref 손상 등 일부 PDF에서만 발생) 그대로 두면 프로토콜 스트림이 오염되어
# 세션이 "unhandled errors in a TaskGroup"으로 끊긴다.
# import 전에 목적지를 stderr로 돌려놓는다.
# ─────────────────────────────────────────────────────────────────────
os.environ.setdefault("PYMUPDF_MESSAGE", "fd:2")

import pymupdf as fitz  # noqa: E402

try:  # import 시점 환경 변수를 놓쳤을 경우를 대비한 이중 안전장치
    fitz.set_messages(fd=2)
except Exception:  # pragma: no cover - 버전에 따라 없을 수 있다
    pass

try:  # MuPDF C 레벨 경고/에러 출력도 끈다
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
except Exception:  # pragma: no cover
    pass

from dotenv import load_dotenv  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

mcp = FastMCP(
    "PdfOcr",
    instructions="스캔 이미지 PDF에서 텍스트를 추출하는 OCR 어시스턴트입니다.",
)

OCR_MODEL = os.getenv("OCR_MODEL", "gpt-4o")
RENDER_DPI = 200          # 너무 높이면 이미지가 커져 비용/지연이 늘어난다
MAX_OCR_PAGES = 40        # 한 번에 OCR할 수 있는 최대 페이지 (비용 안전장치)
MAX_CONCURRENCY = 4       # 동시에 호출할 비전 요청 수
MAX_PIXELS = 2600         # 렌더링 이미지의 긴 변 한도 (px)

OCR_PROMPT = (
    "이 이미지는 문서를 스캔한 페이지입니다. 보이는 텍스트를 원문 그대로 전사하세요.\n"
    "- 요약하거나 설명하지 말고, 텍스트만 그대로 옮깁니다.\n"
    "- 읽는 순서(단 구성)를 지키고, 문단 구분은 빈 줄로 표현합니다.\n"
    "- 표는 마크다운 표로 옮깁니다.\n"
    "- 그림/도표는 [그림: 간단한 설명] 형태로 한 줄만 남깁니다.\n"
    "- 판독이 불가능한 글자는 □로 표시합니다.\n"
    "- 페이지에 텍스트가 전혀 없으면 빈 문자열만 반환합니다."
)


def _client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 없습니다. quiz_generator/.env에 넣거나 "
            "환경 변수로 전달해주세요."
        )
    return AsyncOpenAI(api_key=api_key)


def _render_page(doc: "fitz.Document", page_index: int, dpi: int) -> str:
    """페이지를 PNG로 렌더링해 base64 문자열로 반환

    큰 판형(A3, 포스터 등)이 그대로 들어오면 이미지가 수십 MB가 되므로
    긴 변을 MAX_PIXELS로 제한한다.
    """
    page = doc[page_index]
    rect = page.rect
    longest_inch = max(rect.width, rect.height) / 72  # PDF 단위는 1/72인치

    effective_dpi = dpi
    if longest_inch * dpi > MAX_PIXELS:
        effective_dpi = max(72, int(MAX_PIXELS / longest_inch))

    pixmap = page.get_pixmap(dpi=effective_dpi)
    return base64.b64encode(pixmap.tobytes("png")).decode("ascii")


async def _ocr_one_page(
    client: AsyncOpenAI,
    image_b64: str,
    page_number: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=OCR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OCR_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                temperature=0,
            )
            text = (response.choices[0].message.content or "").strip()
            return {"page": page_number, "text": text}
        except Exception as e:
            return {"page": page_number, "text": "", "error": str(e)}


@mcp.tool()
async def pdf_probe(pdf_path: str) -> dict:
    """PDF의 페이지별 텍스트 레이어 유무를 조사합니다.

    스캔본인지(=OCR이 필요한지) 판단할 때 사용합니다.

    Args:
        pdf_path (str): 조사할 PDF 파일 경로

    Returns:
        dict: 전체 페이지 수, 텍스트가 있는 페이지 목록, 비어 있는 페이지 목록
    """
    path = Path(pdf_path)
    if not path.exists():
        return {"error": f"파일을 찾을 수 없습니다: {pdf_path}"}

    try:
        with fitz.open(path) as doc:
            with_text, without_text = [], []
            for index, page in enumerate(doc, start=1):
                if page.get_text().strip():
                    with_text.append(index)
                else:
                    without_text.append(index)

        return {
            "page_count": len(with_text) + len(without_text),
            "pages_with_text": with_text,
            "pages_without_text": without_text,
            "needs_ocr": bool(without_text),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def pdf_ocr(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    dpi: int = RENDER_DPI,
) -> dict:
    """스캔 PDF의 페이지를 OCR해 텍스트를 추출합니다.

    페이지를 이미지로 렌더링한 뒤 비전 모델로 전사합니다.

    Args:
        pdf_path (str): OCR할 PDF 파일 경로
        pages (list[int] | None): OCR할 페이지 번호(1부터). 생략하면 전체 페이지
        dpi (int): 렌더링 해상도. 기본 200

    Returns:
        dict: success 여부와 [{page, text}] 목록
    """
    path = Path(pdf_path)
    if not path.exists():
        return {"success": False, "error": f"파일을 찾을 수 없습니다: {pdf_path}"}

    try:
        client = _client()
    except RuntimeError as e:
        return {"success": False, "error": str(e)}

    try:
        with fitz.open(path) as doc:
            total = doc.page_count
            targets = pages or list(range(1, total + 1))
            targets = [p for p in targets if 1 <= p <= total]

            if not targets:
                return {"success": False, "error": "OCR할 페이지가 없습니다."}

            if len(targets) > MAX_OCR_PAGES:
                return {
                    "success": False,
                    "error": (
                        f"OCR 대상이 {len(targets)}페이지로 한도({MAX_OCR_PAGES})를 넘습니다. "
                        "페이지를 나눠서 처리해주세요."
                    ),
                }

            # 렌더링은 동기 작업이라 먼저 끝내고, 비전 호출만 동시에 보낸다.
            # 페이지 하나가 깨져도 나머지는 살린다.
            images = []
            render_errors = []
            for page in targets:
                try:
                    images.append((page, _render_page(doc, page - 1, dpi)))
                except Exception as e:
                    render_errors.append({"page": page, "text": "", "error": f"렌더링 실패: {e}"})

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        results = await asyncio.gather(
            *(
                _ocr_one_page(client, image_b64, page, semaphore)
                for page, image_b64 in images
            )
        )
        results = list(results) + render_errors
        results.sort(key=lambda item: item["page"])

        failed = [item["page"] for item in results if item.get("error")]

        return {
            "success": True,
            "engine": f"{OCR_MODEL} (vision)",
            "dpi": dpi,
            "pages": results,
            "failed_pages": failed,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
