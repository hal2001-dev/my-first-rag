"""
=============================================================================
문서 로더 추상 베이스 클래스
=============================================================================

이 모듈은 모든 문서 로더가 구현해야 하는 인터페이스를 정의합니다.
새로운 문서 형식을 지원하려면 이 클래스를 상속받아 구현하세요.

설계 원칙:
    - 단일 책임: 각 로더는 하나의 문서 형식만 처리
    - 개방-폐쇄: 새로운 형식 추가 시 기존 코드 수정 불필요
    - 리스코프 치환: 모든 로더는 동일한 인터페이스로 사용 가능

새로운 로더 추가 방법:
    1. BaseDocumentLoader를 상속받는 새 클래스 생성
    2. load()와 supports() 추상 메서드 구현
    3. LoaderFactory에 새 로더 등록

사용 예시:
    >>> class CSVLoader(BaseDocumentLoader):
    ...     SUPPORTED_EXTENSIONS = {".csv"}
    ...
    ...     def load(self, source: str) -> List[Document]:
    ...         # CSV 로딩 구현
    ...         pass
    ...
    ...     def supports(self, source: str) -> bool:
    ...         return Path(source).suffix.lower() in self.SUPPORTED_EXTENSIONS
"""

from abc import ABC, abstractmethod
from typing import List, Set, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from langchain_core.documents import Document

from app.utils.logger import get_logger, LoggerMixin

# 모듈 레벨 로거
logger = get_logger(__name__)


class BaseDocumentLoader(ABC, LoggerMixin):
    """
    문서 로더의 추상 베이스 클래스

    모든 문서 로더는 이 클래스를 상속받아 구현해야 합니다.
    이를 통해 어떤 형식의 문서든 동일한 방식으로 처리할 수 있습니다.

    Attributes:
        SUPPORTED_EXTENSIONS (Set[str]): 지원하는 파일 확장자 집합
                                        (하위 클래스에서 정의)

    Abstract Methods:
        load(source): 문서를 로드하여 Document 목록 반환
        supports(source): 이 로더가 해당 소스를 지원하는지 확인

    Example:
        >>> loader = PDFDocumentLoader()
        >>> if loader.supports("document.pdf"):
        ...     documents = loader.load("document.pdf")
    """

    # 하위 클래스에서 오버라이드할 지원 확장자 집합
    SUPPORTED_EXTENSIONS: Set[str] = set()

    def __init__(self):
        """BaseDocumentLoader 초기화"""
        self.logger.debug(
            f"{self.__class__.__name__} 초기화됨, "
            f"지원 형식: {self.SUPPORTED_EXTENSIONS}"
        )

    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        소스에서 문서를 로드

        이 메서드는 반드시 하위 클래스에서 구현해야 합니다.
        문서 로딩 실패 시 DocumentLoadError를 발생시켜야 합니다.

        Args:
            source: 문서 소스 (파일 경로 또는 URL)

        Returns:
            List[Document]: 로드된 Document 객체 목록
                           각 Document는 page_content와 metadata를 포함

        Raises:
            DocumentLoadError: 문서 로딩 실패 시
            UnsupportedSourceError: 지원하지 않는 소스인 경우

        Example:
            >>> documents = loader.load("/path/to/document.pdf")
            >>> for doc in documents:
            ...     print(doc.page_content[:100])
            ...     print(doc.metadata)
        """
        pass

    @abstractmethod
    def supports(self, source: str) -> bool:
        """
        이 로더가 주어진 소스를 지원하는지 확인

        LoaderFactory에서 적절한 로더를 선택할 때 사용됩니다.

        Args:
            source: 확인할 소스 (파일 경로 또는 URL)

        Returns:
            bool: 지원하면 True, 아니면 False

        Example:
            >>> loader = PDFDocumentLoader()
            >>> loader.supports("document.pdf")  # True
            >>> loader.supports("document.txt")  # False
        """
        pass

    def _create_document(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        page: Optional[int] = None
    ) -> Document:
        """
        Document 객체 생성 헬퍼 메서드

        모든 문서에 공통 메타데이터를 자동으로 추가합니다.

        Args:
            content: 문서 텍스트 내용
            source: 원본 소스 (파일 경로 또는 URL)
            metadata: 추가 메타데이터 (선택)
            page: 페이지 번호 (PDF 등 다중 페이지 문서용)

        Returns:
            Document: 생성된 Document 객체
        """
        # 기본 메타데이터 구성
        doc_metadata = {
            "source": source,
            "loader_type": self.__class__.__name__,
            "loaded_at": datetime.now().isoformat(),
        }

        # 페이지 번호 추가 (있는 경우)
        if page is not None:
            doc_metadata["page"] = page

        # 추가 메타데이터 병합
        if metadata:
            doc_metadata.update(metadata)

        return Document(
            page_content=content,
            metadata=doc_metadata
        )

    def _add_common_metadata(
        self,
        documents: List[Document],
        source: str
    ) -> List[Document]:
        """
        문서 목록에 공통 메타데이터 추가

        이미 생성된 Document 객체들에 공통 정보를 추가합니다.

        Args:
            documents: Document 객체 목록
            source: 원본 소스

        Returns:
            List[Document]: 메타데이터가 추가된 Document 목록
        """
        for doc in documents:
            # 기존 메타데이터 유지하면서 추가
            doc.metadata["source"] = source
            doc.metadata["loader_type"] = self.__class__.__name__

            # loaded_at이 없으면 추가
            if "loaded_at" not in doc.metadata:
                doc.metadata["loaded_at"] = datetime.now().isoformat()

        return documents

    def _validate_file_exists(self, file_path: str) -> Path:
        """
        파일 존재 여부 검증

        Args:
            file_path: 검증할 파일 경로

        Returns:
            Path: 검증된 Path 객체

        Raises:
            DocumentLoadError: 파일이 없거나 읽을 수 없는 경우
        """
        from app.utils.exceptions import DocumentLoadError

        path = Path(file_path)

        if not path.exists():
            raise DocumentLoadError(
                f"파일을 찾을 수 없습니다",
                source=file_path,
                details={"reason": "file_not_found"}
            )

        if not path.is_file():
            raise DocumentLoadError(
                f"파일이 아닙니다 (디렉토리일 수 있음)",
                source=file_path,
                details={"reason": "not_a_file"}
            )

        return path

    def _validate_extension(self, file_path: str) -> str:
        """
        파일 확장자 검증

        Args:
            file_path: 검증할 파일 경로

        Returns:
            str: 소문자로 변환된 확장자

        Raises:
            UnsupportedSourceError: 지원하지 않는 확장자인 경우
        """
        from app.utils.exceptions import UnsupportedSourceError

        ext = Path(file_path).suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedSourceError(
                f"지원하지 않는 파일 형식입니다: {ext}",
                source=file_path,
                supported_formats=list(self.SUPPORTED_EXTENSIONS)
            )

        return ext

    def _clean_text(self, text: str) -> str:
        """
        텍스트 정제

        불필요한 공백, 특수 문자 등을 정리합니다.

        Args:
            text: 원본 텍스트

        Returns:
            str: 정제된 텍스트
        """
        if not text:
            return ""

        # 연속된 공백을 하나로
        import re
        text = re.sub(r'\s+', ' ', text)

        # 앞뒤 공백 제거
        text = text.strip()

        return text

    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        """
        이 로더가 지원하는 확장자 목록 반환

        Returns:
            Set[str]: 지원하는 확장자 집합

        Example:
            >>> PDFDocumentLoader.get_supported_extensions()
            {'.pdf'}
        """
        return cls.SUPPORTED_EXTENSIONS.copy()

    def __repr__(self) -> str:
        """로더의 문자열 표현"""
        return (
            f"{self.__class__.__name__}("
            f"extensions={self.SUPPORTED_EXTENSIONS})"
        )
