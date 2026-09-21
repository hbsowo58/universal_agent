"""PDF 인제스트와 검색기 (CHAP6 rag_agent/retriever.py 패턴)

CHAP6에서는 Chroma를 디스크에 영속화했지만, 여기서는 업로드된 PDF를 그때그때
색인하므로 langchain-core에 내장된 InMemoryVectorStore를 사용한다.
(영속화가 필요하면 아래 build_index의 벡터스토어만 Chroma로 교체하면 된다.)
"""

import hashlib
import io
import os
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from dotenv import load_dotenv

import ocr_client

load_dotenv()


EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# doc_id -> 색인 결과. 그래프 노드는 상태에 담긴 doc_id로 여기서 검색기를 꺼내 쓴다.
_INDEXES: Dict[str, dict] = {}


def make_doc_id(pdf_bytes: bytes) -> str:
    """파일 내용 해시를 문서 식별자로 사용한다 (같은 PDF는 재색인하지 않음)"""
    return hashlib.sha1(pdf_bytes).hexdigest()[:16]


def extract_text_layer(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """PDF의 텍스트 레이어를 페이지 단위로 읽는다.

    스캔 페이지는 빈 문자열이 들어오므로, 호출부에서 OCR 대상 판별에 쓴다.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # 손상된 페이지는 OCR로 넘긴다
            text = ""
        pages.append((page_num, text))

    return pages


def load_pdf(
    pdf_bytes: bytes,
    use_ocr: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Document], List[int]]:
    """PDF 바이트를 페이지 단위 Document로 변환한다.

    텍스트 레이어가 없는 페이지(= 스캔본)는 OCR MCP 서버에 맡겨 채운다.
    텍스트 페이지와 스캔 페이지가 섞인 PDF도 처리되며, OCR은 빈 페이지에만 돈다.

    Returns:
        (페이지 Document 목록, OCR로 채운 페이지 번호 목록)
    """
    pages = extract_text_layer(pdf_bytes)

    if not pages:
        raise ValueError("PDF에서 페이지를 읽지 못했습니다. 파일이 손상되었을 수 있습니다.")

    text_by_page = {page_num: text for page_num, text in pages}
    empty_pages = [page_num for page_num, text in pages if not text]
    ocr_pages: List[int] = []

    if empty_pages:
        if not use_ocr:
            raise ValueError(
                "PDF에서 텍스트를 추출하지 못했습니다. "
                "스캔 이미지 PDF인 경우 OCR을 거친 파일이 필요합니다."
            )

        for page_num, text in _run_ocr(pdf_bytes, empty_pages, progress):
            if text:
                text_by_page[page_num] = text
                ocr_pages.append(page_num)

    docs = [
        Document(page_content=text_by_page[page_num], metadata={"page": page_num})
        for page_num, _ in pages
        if text_by_page[page_num]
    ]

    if not docs:
        raise ValueError(
            "PDF에서 텍스트를 추출하지 못했습니다. "
            "OCR도 글자를 찾지 못했습니다. 해상도가 너무 낮거나 빈 문서일 수 있습니다."
        )

    return docs, ocr_pages


def _run_ocr(
    pdf_bytes: bytes,
    pages: List[int],
    progress: Optional[Callable[[int, int], None]],
) -> List[Tuple[int, str]]:
    """OCR MCP 서버에 스캔 페이지를 넘긴다.

    MCP 도구는 파일 경로를 받으므로 업로드된 바이트를 임시 파일로 떨어뜨린다.
    """
    handle, temp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(pdf_bytes)

        results = ocr_client.ocr_pdf(temp_path, pages=pages, progress=progress)

    except Exception as error:
        raise ValueError(
            f"스캔 PDF를 OCR하는 중 오류가 발생했습니다: {error}"
        ) from error
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return [(item["page"], (item.get("text") or "").strip()) for item in results]


def build_index(
    pdf_bytes: bytes,
    doc_name: str,
    use_ocr: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> str:
    """PDF를 청킹해 벡터스토어에 색인하고 doc_id를 반환

    Args:
        use_ocr: 텍스트 레이어가 없는 페이지를 OCR로 채울지 여부
        progress: OCR 진행 상황 콜백 (완료 페이지 수, 전체 OCR 대상 수)
    """
    doc_id = make_doc_id(pdf_bytes)

    if doc_id in _INDEXES:  # 이미 색인된 문서
        return doc_id

    pages, ocr_pages = load_pdf(pdf_bytes, use_ocr=use_ocr, progress=progress)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    vectorstore = InMemoryVectorStore.from_documents(
        chunks, embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL)
    )

    _INDEXES[doc_id] = {
        "name": doc_name,
        "pages": pages,
        "chunks": chunks,
        "vectorstore": vectorstore,
        "ocr_pages": ocr_pages,
    }

    return doc_id


def has_index(doc_id: str) -> bool:
    return doc_id in _INDEXES


def get_retriever(doc_id: str, k: int = 4):
    """doc_id에 해당하는 검색기를 반환"""
    index = _INDEXES.get(doc_id)
    if index is None:
        raise KeyError(f"색인되지 않은 문서입니다: {doc_id}. 먼저 build_index를 호출하세요.")
    return index["vectorstore"].as_retriever(search_kwargs={"k": k})


def get_outline(doc_id: str, max_chars: int = 6000) -> str:
    """출제 계획 수립용 문서 개요 (앞부분 텍스트를 잘라서 사용)"""
    index = _INDEXES[doc_id]

    buffer = []
    total = 0
    for page in index["pages"]:
        snippet = f"[p.{page.metadata['page']}] {page.page_content}"
        if total + len(snippet) > max_chars:
            buffer.append(snippet[: max_chars - total])
            break
        buffer.append(snippet)
        total += len(snippet)

    return "\n\n".join(buffer)


def get_stats(doc_id: str) -> dict:
    index = _INDEXES[doc_id]
    return {
        "name": index["name"],
        "page_count": len(index["pages"]),
        "chunk_count": len(index["chunks"]),
        "ocr_pages": index.get("ocr_pages", []),
        "ocr_page_count": len(index.get("ocr_pages", [])),
    }


def retrieve_context(doc_id: str, queries: List[str], k: int = 4) -> str:
    """여러 주제 쿼리로 검색한 뒤 페이지 순서대로 정리한 발췌문을 만든다
    (CHAP6 rag_agent의 retrieve + context_organizer를 하나로 합친 형태)
    """
    retriever = get_retriever(doc_id, k=k)

    seen = set()
    collected = []
    for query in queries:
        for doc in retriever.invoke(query):
            key = (doc.metadata.get("page"), doc.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            collected.append(doc)

    collected.sort(key=lambda d: d.metadata.get("page", 0))

    return "\n\n".join(
        f"[p.{doc.metadata.get('page', '?')}] {doc.page_content}" for doc in collected
    )
