"""
=============================================================================
문서 로더 팩토리
=============================================================================

이 모듈은 팩토리 패턴을 사용하여 적절한 문서 로더를 자동 선택합니다.

팩토리 패턴의 장점:
    - 클라이언트 코드가 구체적인 로더 클래스를 알 필요 없음
    - 새로운 로더 추가 시 팩토리만 수정하면 됨
    - 소스 형식에 따른 로더 선택 로직 중앙화

새로운 로더 추가 방법:
    1. BaseDocumentLoader를 상속받는 새 클래스 생성
    2. DocumentLoaderFactory._loaders 리스트에 추가

사용 예시:
    >>> factory = DocumentLoaderFactory()

    >>> # 소스에 맞는 로더 자동 선택
    >>> loader = factory.get_loader("document.pdf")
    >>> documents = loader.load("document.pdf")

    >>> # 또는 팩토리로 직접 로드
    >>> documents = factory.load("document.pdf")
"""

from typing import List, Optional, Type, Dict, Any

from langchain_core.documents import Document

from app.loaders.base_loader import BaseDocumentLoader
from app.loaders.pdf_loader import PDFDocumentLoader
from app.loaders.text_loader import TextDocumentLoader
from app.loaders.web_loader import WebDocumentLoader
from app.utils.logger import get_logger, LoggerMixin
from app.utils.exceptions import UnsupportedSourceError, DocumentLoadError

# 모듈 레벨 로거
logger = get_logger(__name__)


class DocumentLoaderFactory(LoggerMixin):
    """
    문서 로더 팩토리

    주어진 소스(파일 경로 또는 URL)에 적합한 로더를 자동으로 선택합니다.
    팩토리 패턴을 사용하여 클라이언트 코드와 구체적인 로더 클래스를 분리합니다.

    Attributes:
        _loaders (List[BaseDocumentLoader]): 등록된 로더 인스턴스 목록

    Methods:
        get_loader: 소스에 맞는 로더 반환
        load: 소스에서 문서 로드
        load_multiple: 여러 소스에서 문서 로드

    새로운 로더 추가:
        >>> factory = DocumentLoaderFactory()
        >>> factory.register_loader(MyCustomLoader())

    Example:
        >>> factory = DocumentLoaderFactory()

        >>> # PDF 파일 로드
        >>> docs = factory.load("report.pdf")

        >>> # 웹 페이지 로드
        >>> docs = factory.load("https://example.com/article")

        >>> # 텍스트 파일 로드
        >>> docs = factory.load("notes.txt")
    """

    def __init__(self):
        """
        DocumentLoaderFactory 초기화

        기본 로더들(PDF, 텍스트, 웹)을 자동으로 등록합니다.
        """
        # 등록된 로더 목록
        # 순서가 중요: 먼저 매칭되는 로더가 사용됨
        self._loaders: List[BaseDocumentLoader] = [
            PDFDocumentLoader(),
            TextDocumentLoader(),
            WebDocumentLoader(),  # URL 기반이므로 마지막에 체크
        ]

        self.logger.info(
            f"DocumentLoaderFactory 초기화됨, "
            f"등록된 로더: {len(self._loaders)}개"
        )

    def register_loader(self, loader: BaseDocumentLoader) -> None:
        """
        새로운 로더 등록

        기존 로더 목록의 맨 앞에 추가됩니다.
        (높은 우선순위)

        Args:
            loader: 등록할 로더 인스턴스

        Example:
            >>> factory = DocumentLoaderFactory()
            >>> factory.register_loader(MyCustomLoader())
        """
        if not isinstance(loader, BaseDocumentLoader):
            raise TypeError(
                f"로더는 BaseDocumentLoader를 상속해야 합니다. "
                f"입력: {type(loader).__name__}"
            )

        # 맨 앞에 추가 (높은 우선순위)
        self._loaders.insert(0, loader)

        self.logger.info(
            f"새 로더 등록됨: {loader.__class__.__name__}, "
            f"지원 형식: {loader.SUPPORTED_EXTENSIONS or '(URL 기반)'}"
        )

    def unregister_loader(
        self,
        loader_type: Type[BaseDocumentLoader]
    ) -> bool:
        """
        특정 타입의 로더 등록 해제

        Args:
            loader_type: 해제할 로더 클래스 타입

        Returns:
            bool: 해제 성공 여부

        Example:
            >>> factory.unregister_loader(PDFDocumentLoader)
        """
        original_count = len(self._loaders)
        self._loaders = [
            loader for loader in self._loaders
            if not isinstance(loader, loader_type)
        ]

        removed = original_count - len(self._loaders)
        if removed > 0:
            self.logger.info(f"로더 등록 해제됨: {loader_type.__name__}")
            return True

        return False

    def get_loader(self, source: str) -> BaseDocumentLoader:
        """
        소스에 적합한 로더 반환

        등록된 로더들을 순회하며 첫 번째로 매칭되는 로더를 반환합니다.

        Args:
            source: 문서 소스 (파일 경로 또는 URL)

        Returns:
            BaseDocumentLoader: 매칭된 로더 인스턴스

        Raises:
            UnsupportedSourceError: 지원하는 로더가 없는 경우

        Example:
            >>> loader = factory.get_loader("document.pdf")
            >>> print(loader.__class__.__name__)  # "PDFDocumentLoader"
        """
        self.logger.debug(f"로더 검색 중: {source}")

        for loader in self._loaders:
            if loader.supports(source):
                self.logger.debug(
                    f"로더 선택됨: {loader.__class__.__name__} for {source}"
                )
                return loader

        # 매칭되는 로더 없음
        supported = self.get_supported_formats()
        raise UnsupportedSourceError(
            f"지원하지 않는 소스 형식입니다: {source}",
            source=source,
            supported_formats=supported
        )

    def load(self, source: str) -> List[Document]:
        """
        소스에서 문서 로드

        적절한 로더를 자동으로 선택하여 문서를 로드합니다.

        Args:
            source: 문서 소스 (파일 경로 또는 URL)

        Returns:
            List[Document]: 로드된 Document 목록

        Raises:
            UnsupportedSourceError: 지원하지 않는 형식인 경우
            DocumentLoadError: 로딩 실패 시

        Example:
            >>> # 소스 형식에 따라 자동으로 적절한 로더 사용
            >>> docs = factory.load("report.pdf")
            >>> docs = factory.load("https://example.com/page")
            >>> docs = factory.load("notes.md")
        """
        self.logger.info(f"문서 로드 시작: {source}")

        loader = self.get_loader(source)
        documents = loader.load(source)

        self.logger.info(
            f"문서 로드 완료: {source}, "
            f"로더: {loader.__class__.__name__}, "
            f"문서 수: {len(documents)}"
        )

        return documents

    def load_multiple(
        self,
        sources: List[str],
        fail_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        여러 소스에서 문서 로드

        Args:
            sources: 소스 목록 (파일 경로 또는 URL 혼합 가능)
            fail_on_error: True면 하나라도 실패 시 예외 발생
                          False면 실패한 소스 건너뛰고 계속 진행

        Returns:
            dict: 로드 결과
                - documents: 로드된 Document 목록
                - sources_processed: 처리된 소스 수
                - sources_failed: 실패한 소스 수
                - errors: 오류 목록 [{"source": ..., "error": ...}]

        Example:
            >>> sources = [
            ...     "report.pdf",
            ...     "https://example.com/article",
            ...     "notes.txt",
            ... ]
            >>> result = factory.load_multiple(sources)
            >>> print(f"로드 성공: {result['sources_processed']}개")
            >>> print(f"실패: {result['sources_failed']}개")
            >>> documents = result["documents"]
        """
        self.logger.info(f"다중 소스 로드 시작: {len(sources)}개")

        result = {
            "documents": [],
            "sources_processed": 0,
            "sources_failed": 0,
            "errors": [],
        }

        for source in sources:
            try:
                documents = self.load(source)
                result["documents"].extend(documents)
                result["sources_processed"] += 1

            except (UnsupportedSourceError, DocumentLoadError) as e:
                self.logger.warning(f"소스 로드 실패: {source}, 오류: {e}")
                result["errors"].append({
                    "source": source,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                result["sources_failed"] += 1

                if fail_on_error:
                    raise

            except Exception as e:
                self.logger.error(f"예상치 못한 오류: {source}, {e}")
                result["errors"].append({
                    "source": source,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                result["sources_failed"] += 1

                if fail_on_error:
                    raise DocumentLoadError(
                        f"문서 로드 중 예상치 못한 오류",
                        source=source,
                        original_error=e
                    )

        self.logger.info(
            f"다중 소스 로드 완료: "
            f"성공 {result['sources_processed']}, "
            f"실패 {result['sources_failed']}, "
            f"총 문서 {len(result['documents'])}개"
        )

        return result

    def can_load(self, source: str) -> bool:
        """
        소스를 로드할 수 있는지 확인

        Args:
            source: 확인할 소스

        Returns:
            bool: 로드 가능하면 True

        Example:
            >>> factory.can_load("document.pdf")  # True
            >>> factory.can_load("unknown.xyz")  # False
        """
        for loader in self._loaders:
            if loader.supports(source):
                return True
        return False

    def get_supported_formats(self) -> List[str]:
        """
        지원하는 모든 형식 목록 반환

        Returns:
            List[str]: 지원 형식 목록

        Example:
            >>> formats = factory.get_supported_formats()
            >>> print(formats)
            ['.pdf', '.txt', '.md', '.markdown', 'http://', 'https://']
        """
        formats = []

        for loader in self._loaders:
            # 파일 확장자 기반 로더
            extensions = loader.get_supported_extensions()
            if extensions:
                formats.extend(extensions)
            else:
                # URL 기반 로더 (WebLoader)
                if isinstance(loader, WebDocumentLoader):
                    formats.extend(["http://", "https://"])

        return list(set(formats))  # 중복 제거

    def get_loader_for_extension(
        self,
        extension: str
    ) -> Optional[BaseDocumentLoader]:
        """
        특정 확장자를 처리하는 로더 반환

        Args:
            extension: 파일 확장자 (예: ".pdf")

        Returns:
            Optional[BaseDocumentLoader]: 해당 확장자를 지원하는 로더
                                         지원하지 않으면 None

        Example:
            >>> loader = factory.get_loader_for_extension(".pdf")
            >>> print(loader.__class__.__name__)  # "PDFDocumentLoader"
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext

        for loader in self._loaders:
            if ext in loader.get_supported_extensions():
                return loader

        return None

    def get_registered_loaders(self) -> List[str]:
        """
        등록된 로더 클래스명 목록 반환

        Returns:
            List[str]: 로더 클래스명 목록

        Example:
            >>> loaders = factory.get_registered_loaders()
            >>> print(loaders)
            ['PDFDocumentLoader', 'TextDocumentLoader', 'WebDocumentLoader']
        """
        return [loader.__class__.__name__ for loader in self._loaders]

    def __repr__(self) -> str:
        """팩토리의 문자열 표현"""
        loaders = self.get_registered_loaders()
        formats = self.get_supported_formats()
        return (
            f"DocumentLoaderFactory("
            f"loaders={loaders}, "
            f"formats={formats})"
        )


# =============================================================================
# 편의 함수
# =============================================================================

# 전역 팩토리 인스턴스 (싱글톤)
_factory_instance: Optional[DocumentLoaderFactory] = None


def get_loader_factory() -> DocumentLoaderFactory:
    """
    팩토리 싱글톤 인스턴스 반환

    앱 전체에서 하나의 팩토리 인스턴스를 공유합니다.

    Returns:
        DocumentLoaderFactory: 팩토리 인스턴스

    Example:
        >>> factory = get_loader_factory()
        >>> docs = factory.load("document.pdf")
    """
    global _factory_instance

    if _factory_instance is None:
        _factory_instance = DocumentLoaderFactory()

    return _factory_instance


def load_document(source: str) -> List[Document]:
    """
    단일 소스에서 문서 로드 (편의 함수)

    전역 팩토리를 사용하여 문서를 로드합니다.

    Args:
        source: 문서 소스

    Returns:
        List[Document]: 로드된 Document 목록

    Example:
        >>> docs = load_document("report.pdf")
    """
    return get_loader_factory().load(source)


def load_documents(sources: List[str]) -> List[Document]:
    """
    여러 소스에서 문서 로드 (편의 함수)

    전역 팩토리를 사용하여 여러 문서를 로드합니다.
    실패한 소스는 건너뛰고 계속 진행합니다.

    Args:
        sources: 문서 소스 목록

    Returns:
        List[Document]: 로드된 모든 Document 목록

    Example:
        >>> docs = load_documents(["a.pdf", "b.txt", "c.md"])
    """
    result = get_loader_factory().load_multiple(sources, fail_on_error=False)
    return result["documents"]
