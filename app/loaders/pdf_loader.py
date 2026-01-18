"""
=============================================================================
PDF 문서 로더
=============================================================================

이 모듈은 PDF 파일에서 텍스트를 추출하는 기능을 제공합니다.

사용 라이브러리:
    - PyPDF: 기본 PDF 텍스트 추출 (가볍고 빠름)
    - LangChain PyPDFLoader: LangChain 통합 로더

PDF 처리 특징:
    - 페이지별로 별도의 Document 생성
    - 페이지 번호 메타데이터 자동 추가
    - 암호화된 PDF는 지원하지 않음
    - 이미지 기반 PDF (스캔)는 OCR 필요 (별도 처리)

사용 예시:
    >>> loader = PDFDocumentLoader()
    >>> documents = loader.load("document.pdf")
    >>> for doc in documents:
    ...     print(f"페이지 {doc.metadata['page']}: {doc.page_content[:100]}")
"""

from typing import List
from pathlib import Path

from langchain_core.documents import Document

from app.loaders.base_loader import BaseDocumentLoader
from app.utils.logger import get_logger
from app.utils.exceptions import DocumentLoadError

# 모듈 레벨 로거
logger = get_logger(__name__)


class PDFDocumentLoader(BaseDocumentLoader):
    """
    PDF 문서 로더

    PDF 파일을 읽어서 페이지별 Document 객체를 생성합니다.
    각 페이지는 별도의 Document로 변환되며, 페이지 번호가 메타데이터에 포함됩니다.

    Attributes:
        SUPPORTED_EXTENSIONS: 지원하는 확장자 집합 ({".pdf"})

    Features:
        - 다중 페이지 PDF 지원
        - 페이지별 메타데이터 (페이지 번호, 출처 등)
        - 텍스트 추출 후 자동 정제

    Limitations:
        - 암호화된 PDF 미지원
        - 이미지 기반 PDF는 텍스트 추출 불가 (OCR 필요)
        - 복잡한 레이아웃은 텍스트 순서가 뒤섞일 수 있음

    Example:
        >>> loader = PDFDocumentLoader()

        >>> # 단일 파일 로드
        >>> docs = loader.load("report.pdf")
        >>> print(f"총 {len(docs)} 페이지 로드됨")

        >>> # 지원 여부 확인
        >>> loader.supports("document.pdf")  # True
        >>> loader.supports("document.docx")  # False
    """

    # 지원하는 파일 확장자
    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self):
        """PDFDocumentLoader 초기화"""
        super().__init__()
        self.logger.info("PDFDocumentLoader 초기화됨")

    def load(self, source: str) -> List[Document]:
        """
        PDF 파일에서 문서 로드

        PDF의 각 페이지를 별도의 Document 객체로 변환합니다.
        페이지 번호는 1부터 시작합니다.

        Args:
            source: PDF 파일 경로

        Returns:
            List[Document]: 페이지별 Document 객체 목록
                           각 Document의 metadata에 page 번호 포함

        Raises:
            DocumentLoadError: 파일이 없거나 읽기 실패 시

        Example:
            >>> docs = loader.load("manual.pdf")
            >>> # 각 페이지 확인
            >>> for doc in docs:
            ...     page_num = doc.metadata.get("page", "?")
            ...     preview = doc.page_content[:100]
            ...     print(f"[페이지 {page_num}] {preview}...")
        """
        self.logger.info(f"PDF 로딩 시작: {source}")

        # 파일 검증
        path = self._validate_file_exists(source)
        self._validate_extension(source)

        try:
            # LangChain PyPDFLoader 사용
            from langchain_community.document_loaders import PyPDFLoader

            # PDF 로더 생성 및 실행
            pdf_loader = PyPDFLoader(str(path))
            documents = pdf_loader.load()

            # 빈 페이지 필터링 및 메타데이터 추가
            processed_docs = []
            for i, doc in enumerate(documents):
                # 내용이 있는 페이지만 포함
                content = self._clean_text(doc.page_content)
                if content:
                    # 메타데이터 보강
                    doc.metadata["source"] = source
                    doc.metadata["loader_type"] = self.__class__.__name__
                    doc.metadata["file_name"] = path.name
                    doc.metadata["file_type"] = "pdf"

                    # 페이지 번호가 없으면 추가 (1부터 시작)
                    if "page" not in doc.metadata:
                        doc.metadata["page"] = i + 1

                    doc.page_content = content
                    processed_docs.append(doc)

            self.logger.info(
                f"PDF 로딩 완료: {path.name}, "
                f"총 {len(documents)}페이지 중 {len(processed_docs)}페이지 추출됨"
            )

            return processed_docs

        except ImportError as e:
            raise DocumentLoadError(
                "PDF 로더 라이브러리가 설치되지 않았습니다. "
                "pip install pypdf 를 실행하세요.",
                source=source,
                original_error=e
            )
        except Exception as e:
            self.logger.error(f"PDF 로딩 실패: {source}, 오류: {e}")
            raise DocumentLoadError(
                f"PDF 파일 로딩에 실패했습니다: {path.name}",
                source=source,
                details={"error_type": type(e).__name__},
                original_error=e
            )

    def supports(self, source: str) -> bool:
        """
        이 로더가 주어진 소스를 지원하는지 확인

        Args:
            source: 확인할 파일 경로

        Returns:
            bool: .pdf 파일이면 True

        Example:
            >>> loader.supports("report.pdf")  # True
            >>> loader.supports("DOCUMENT.PDF")  # True (대소문자 무관)
            >>> loader.supports("notes.txt")  # False
        """
        ext = Path(source).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def load_with_page_range(
        self,
        source: str,
        start_page: int = 1,
        end_page: int = None
    ) -> List[Document]:
        """
        특정 페이지 범위만 로드

        대용량 PDF에서 필요한 페이지만 추출할 때 유용합니다.

        Args:
            source: PDF 파일 경로
            start_page: 시작 페이지 (1부터 시작, 포함)
            end_page: 끝 페이지 (포함). None이면 끝까지

        Returns:
            List[Document]: 지정된 범위의 페이지 Document 목록

        Example:
            >>> # 1-10 페이지만 로드
            >>> docs = loader.load_with_page_range("large.pdf", 1, 10)

            >>> # 50 페이지부터 끝까지
            >>> docs = loader.load_with_page_range("large.pdf", 50)
        """
        self.logger.info(
            f"PDF 페이지 범위 로딩: {source}, "
            f"페이지 {start_page}-{end_page or '끝'}"
        )

        # 전체 로드 후 필터링
        all_docs = self.load(source)

        # 페이지 범위 필터링
        filtered_docs = []
        for doc in all_docs:
            page = doc.metadata.get("page", 0)
            if page >= start_page:
                if end_page is None or page <= end_page:
                    filtered_docs.append(doc)

        self.logger.info(f"페이지 범위 필터링 완료: {len(filtered_docs)}페이지")
        return filtered_docs

    def get_page_count(self, source: str) -> int:
        """
        PDF의 총 페이지 수 반환

        전체 로드 없이 페이지 수만 확인할 때 사용합니다.

        Args:
            source: PDF 파일 경로

        Returns:
            int: 총 페이지 수

        Example:
            >>> count = loader.get_page_count("document.pdf")
            >>> print(f"총 {count} 페이지")
        """
        path = self._validate_file_exists(source)

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return len(reader.pages)

        except ImportError:
            # pypdf가 없으면 전체 로드 후 카운트
            return len(self.load(source))
        except Exception as e:
            self.logger.warning(f"페이지 수 확인 실패: {e}")
            return 0

    def extract_metadata(self, source: str) -> dict:
        """
        PDF 메타데이터 추출 (제목, 저자 등)

        Args:
            source: PDF 파일 경로

        Returns:
            dict: PDF 메타데이터
                 - title: 문서 제목
                 - author: 저자
                 - subject: 주제
                 - creator: 생성 프로그램
                 - producer: PDF 생성기
                 - creation_date: 생성일
                 - modification_date: 수정일
                 - page_count: 총 페이지 수

        Example:
            >>> metadata = loader.extract_metadata("book.pdf")
            >>> print(f"제목: {metadata.get('title')}")
            >>> print(f"저자: {metadata.get('author')}")
        """
        path = self._validate_file_exists(source)

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            info = reader.metadata or {}

            return {
                "title": info.get("/Title", ""),
                "author": info.get("/Author", ""),
                "subject": info.get("/Subject", ""),
                "creator": info.get("/Creator", ""),
                "producer": info.get("/Producer", ""),
                "creation_date": str(info.get("/CreationDate", "")),
                "modification_date": str(info.get("/ModDate", "")),
                "page_count": len(reader.pages),
            }

        except ImportError:
            return {"page_count": len(self.load(source))}
        except Exception as e:
            self.logger.warning(f"메타데이터 추출 실패: {e}")
            return {}
