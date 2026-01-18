"""
=============================================================================
텍스트 파일 로더
=============================================================================

이 모듈은 일반 텍스트 파일 (.txt, .md, .markdown)을 로드합니다.

지원 형식:
    - .txt: 일반 텍스트 파일
    - .md: 마크다운 파일
    - .markdown: 마크다운 파일 (긴 확장자)

특징:
    - UTF-8 인코딩 기본 지원
    - 다른 인코딩 자동 감지 시도
    - 마크다운 구조 보존

사용 예시:
    >>> loader = TextDocumentLoader()
    >>> documents = loader.load("readme.md")
    >>> print(documents[0].page_content)
"""

from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document

from app.loaders.base_loader import BaseDocumentLoader
from app.utils.logger import get_logger
from app.utils.exceptions import DocumentLoadError

# 모듈 레벨 로거
logger = get_logger(__name__)


class TextDocumentLoader(BaseDocumentLoader):
    """
    텍스트 파일 로더

    일반 텍스트 파일과 마크다운 파일을 로드합니다.
    전체 파일 내용이 하나의 Document로 변환됩니다.

    Attributes:
        SUPPORTED_EXTENSIONS: 지원하는 확장자 ({".txt", ".md", ".markdown"})
        default_encoding: 기본 인코딩 ("utf-8")

    Features:
        - UTF-8 및 다양한 인코딩 지원
        - 파일 타입별 메타데이터 자동 추가
        - 빈 파일 감지 및 경고

    Example:
        >>> loader = TextDocumentLoader()

        >>> # 텍스트 파일 로드
        >>> docs = loader.load("notes.txt")

        >>> # 마크다운 파일 로드
        >>> docs = loader.load("README.md")
        >>> print(docs[0].metadata["file_type"])  # "markdown"
    """

    # 지원하는 파일 확장자
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}

    # 인코딩 시도 순서 (UTF-8 우선)
    ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]

    def __init__(self, default_encoding: str = "utf-8"):
        """
        TextDocumentLoader 초기화

        Args:
            default_encoding: 기본 인코딩 (기본값: "utf-8")
        """
        super().__init__()
        self.default_encoding = default_encoding
        self.logger.info(f"TextDocumentLoader 초기화됨, 인코딩: {default_encoding}")

    def load(self, source: str) -> List[Document]:
        """
        텍스트 파일에서 문서 로드

        파일 전체를 하나의 Document로 변환합니다.

        Args:
            source: 텍스트 파일 경로

        Returns:
            List[Document]: Document 객체 목록 (보통 1개)

        Raises:
            DocumentLoadError: 파일이 없거나 읽기 실패 시

        Example:
            >>> docs = loader.load("notes.txt")
            >>> content = docs[0].page_content
            >>> file_type = docs[0].metadata["file_type"]
        """
        self.logger.info(f"텍스트 파일 로딩 시작: {source}")

        # 파일 검증
        path = self._validate_file_exists(source)
        ext = self._validate_extension(source)

        # 파일 타입 결정
        file_type = self._get_file_type(ext)

        # 파일 읽기 (인코딩 자동 감지)
        content = self._read_file_with_encoding(path)

        # 빈 파일 체크
        if not content or not content.strip():
            self.logger.warning(f"빈 파일입니다: {source}")
            return []

        # Document 생성
        document = self._create_document(
            content=content,
            source=source,
            metadata={
                "file_name": path.name,
                "file_type": file_type,
                "file_size": path.stat().st_size,
                "extension": ext,
            }
        )

        self.logger.info(
            f"텍스트 파일 로딩 완료: {path.name}, "
            f"크기: {len(content)}자"
        )

        return [document]

    def supports(self, source: str) -> bool:
        """
        이 로더가 주어진 소스를 지원하는지 확인

        Args:
            source: 확인할 파일 경로

        Returns:
            bool: .txt, .md, .markdown 파일이면 True

        Example:
            >>> loader.supports("readme.md")  # True
            >>> loader.supports("notes.txt")  # True
            >>> loader.supports("doc.pdf")  # False
        """
        ext = Path(source).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def _get_file_type(self, extension: str) -> str:
        """
        확장자에 따른 파일 타입 결정

        Args:
            extension: 파일 확장자 (소문자)

        Returns:
            str: 파일 타입 ("text" 또는 "markdown")
        """
        if extension in {".md", ".markdown"}:
            return "markdown"
        return "text"

    def _read_file_with_encoding(self, path: Path) -> str:
        """
        여러 인코딩을 시도하여 파일 읽기

        UTF-8을 우선 시도하고, 실패하면 다른 인코딩을 순차적으로 시도합니다.

        Args:
            path: 파일 경로

        Returns:
            str: 파일 내용

        Raises:
            DocumentLoadError: 모든 인코딩으로 읽기 실패 시
        """
        errors = []

        for encoding in self.ENCODING_FALLBACKS:
            try:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()

                self.logger.debug(f"파일 읽기 성공: 인코딩={encoding}")
                return content

            except UnicodeDecodeError as e:
                errors.append(f"{encoding}: {e}")
                continue
            except Exception as e:
                raise DocumentLoadError(
                    f"파일 읽기 중 오류 발생",
                    source=str(path),
                    original_error=e
                )

        # 모든 인코딩 실패
        raise DocumentLoadError(
            f"파일 인코딩을 감지할 수 없습니다. "
            f"시도한 인코딩: {self.ENCODING_FALLBACKS}",
            source=str(path),
            details={"encoding_errors": errors}
        )

    def load_with_line_numbers(
        self,
        source: str,
        start_line: int = 1,
        end_line: Optional[int] = None
    ) -> List[Document]:
        """
        특정 라인 범위만 로드

        대용량 파일에서 특정 부분만 추출할 때 유용합니다.

        Args:
            source: 텍스트 파일 경로
            start_line: 시작 라인 (1부터 시작)
            end_line: 끝 라인 (포함). None이면 끝까지

        Returns:
            List[Document]: 지정된 범위의 Document

        Example:
            >>> # 1-100 라인만 로드
            >>> docs = loader.load_with_line_numbers("log.txt", 1, 100)
        """
        self.logger.info(
            f"라인 범위 로딩: {source}, "
            f"라인 {start_line}-{end_line or '끝'}"
        )

        path = self._validate_file_exists(source)
        ext = self._validate_extension(source)

        content = self._read_file_with_encoding(path)
        lines = content.splitlines()

        # 라인 범위 추출 (인덱스는 0부터, 라인 번호는 1부터)
        start_idx = start_line - 1
        end_idx = end_line if end_line else len(lines)

        selected_lines = lines[start_idx:end_idx]
        selected_content = "\n".join(selected_lines)

        if not selected_content.strip():
            return []

        document = self._create_document(
            content=selected_content,
            source=source,
            metadata={
                "file_name": path.name,
                "file_type": self._get_file_type(ext),
                "start_line": start_line,
                "end_line": end_line or len(lines),
                "total_lines": len(lines),
            }
        )

        return [document]

    def load_by_sections(
        self,
        source: str,
        section_delimiter: str = "\n\n"
    ) -> List[Document]:
        """
        섹션 구분자로 나누어 로드

        빈 줄이나 특정 구분자로 나뉜 섹션을 각각 별도의 Document로 생성합니다.

        Args:
            source: 텍스트 파일 경로
            section_delimiter: 섹션 구분자 (기본값: 빈 줄)

        Returns:
            List[Document]: 섹션별 Document 목록

        Example:
            >>> # 빈 줄로 구분된 섹션들
            >>> docs = loader.load_by_sections("article.txt")

            >>> # 커스텀 구분자 사용
            >>> docs = loader.load_by_sections("log.txt", "---")
        """
        self.logger.info(f"섹션별 로딩: {source}")

        path = self._validate_file_exists(source)
        ext = self._validate_extension(source)

        content = self._read_file_with_encoding(path)

        # 섹션 분리
        sections = content.split(section_delimiter)

        documents = []
        for i, section in enumerate(sections):
            section = section.strip()
            if section:  # 빈 섹션 제외
                doc = self._create_document(
                    content=section,
                    source=source,
                    metadata={
                        "file_name": path.name,
                        "file_type": self._get_file_type(ext),
                        "section_index": i + 1,
                        "total_sections": len(sections),
                    }
                )
                documents.append(doc)

        self.logger.info(f"섹션 로딩 완료: {len(documents)}개 섹션")
        return documents

    def get_line_count(self, source: str) -> int:
        """
        파일의 총 라인 수 반환

        Args:
            source: 텍스트 파일 경로

        Returns:
            int: 총 라인 수

        Example:
            >>> count = loader.get_line_count("log.txt")
            >>> print(f"총 {count} 라인")
        """
        path = self._validate_file_exists(source)
        content = self._read_file_with_encoding(path)
        return len(content.splitlines())

    def get_file_stats(self, source: str) -> dict:
        """
        파일 통계 정보 반환

        Args:
            source: 텍스트 파일 경로

        Returns:
            dict: 파일 통계
                 - line_count: 라인 수
                 - word_count: 단어 수 (대략)
                 - char_count: 문자 수
                 - file_size: 파일 크기 (바이트)

        Example:
            >>> stats = loader.get_file_stats("document.txt")
            >>> print(f"라인: {stats['line_count']}, 단어: {stats['word_count']}")
        """
        path = self._validate_file_exists(source)
        content = self._read_file_with_encoding(path)

        return {
            "line_count": len(content.splitlines()),
            "word_count": len(content.split()),
            "char_count": len(content),
            "file_size": path.stat().st_size,
        }
