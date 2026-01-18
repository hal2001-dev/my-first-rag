"""
=============================================================================
텍스트 청킹 모듈
=============================================================================

이 모듈은 문서를 작은 청크(조각)로 분할하는 기능을 제공합니다.

청킹이 필요한 이유:
    1. LLM의 컨텍스트 윈도우 제한 (토큰 수 제한)
    2. 유사도 검색 시 관련 없는 내용 제외
    3. 더 정확한 검색 결과를 위한 세분화

청킹 전략:
    - RecursiveCharacterTextSplitter (기본)
      의미 단위로 분할 시도: 단락 → 문장 → 단어 순
    - CharacterTextSplitter
      단순 문자 수 기반 분할
    - TokenTextSplitter
      토큰 수 기반 분할 (LLM 토큰 제한에 정확)

청크 크기 가이드:
    - 너무 작으면 (< 200자): 문맥 손실, 의미 파편화
    - 너무 크면 (> 1000자): 검색 정확도 저하
    - 권장: 256-512자 (일반 문서)

사용 예시:
    >>> from app.chunking import TextChunker
    >>> chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    >>> chunks = chunker.split_documents(documents)
"""

from typing import List, Optional
from enum import Enum

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

from app.config import get_settings
from app.utils.logger import get_logger, LoggerMixin

# 모듈 레벨 로거
logger = get_logger(__name__)


class SplitterType(str, Enum):
    """
    지원하는 텍스트 분할기 유형

    새로운 분할기 추가 시:
        1. 여기에 새 값 추가
        2. TextChunker._create_splitter()에 구현 추가
    """
    RECURSIVE = "recursive"     # 의미 단위 분할 (기본, 권장)
    CHARACTER = "character"     # 단순 문자 기반 분할
    # TOKEN = "token"           # 토큰 기반 분할 (향후 추가)


class TextChunker(LoggerMixin):
    """
    텍스트 청킹 클래스

    문서를 지정된 크기의 청크로 분할합니다.
    RecursiveCharacterTextSplitter를 기본으로 사용하여
    의미 단위로 분할을 시도합니다.

    Attributes:
        chunk_size (int): 청크 크기 (문자 수)
        chunk_overlap (int): 청크 간 오버랩 (문자 수)
        splitter_type (SplitterType): 사용할 분할기 유형
        separators (List[str]): 분할 시 사용할 구분자 목록

    분할 우선순위 (RecursiveCharacterTextSplitter):
        1. 단락 분리 ("\\n\\n")
        2. 줄바꿈 ("\\n")
        3. 문장 끝 (". ", "? ", "! ")
        4. 공백 (" ")
        5. 문자 단위 ("")

    Example:
        >>> chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.split_documents(documents)
        >>> print(f"총 {len(chunks)}개 청크 생성")
    """

    # 기본 구분자 (분할 우선순위 순)
    DEFAULT_SEPARATORS = [
        "\n\n",     # 단락 분리
        "\n",       # 줄바꿈
        ". ",       # 문장 끝 (마침표)
        "? ",       # 문장 끝 (물음표)
        "! ",       # 문장 끝 (느낌표)
        "。",       # 일본어/중국어 마침표
        "？",       # 전각 물음표
        "！",       # 전각 느낌표
        " ",        # 공백 (단어 단위)
        "",         # 문자 단위 (최후 수단)
    ]

    # 마크다운 전용 구분자
    MARKDOWN_SEPARATORS = [
        "\n## ",    # H2 헤더
        "\n### ",   # H3 헤더
        "\n#### ",  # H4 헤더
        "\n\n",     # 단락
        "\n",       # 줄바꿈
        ". ",       # 문장
        " ",        # 단어
        "",         # 문자
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        splitter_type: SplitterType = SplitterType.RECURSIVE,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        """
        TextChunker 초기화

        Args:
            chunk_size: 청크 크기 (문자 수). None이면 설정에서 읽음
            chunk_overlap: 청크 간 오버랩. None이면 설정에서 읽음
            splitter_type: 분할기 유형 (recursive, character)
            separators: 커스텀 구분자 목록. None이면 기본값 사용
            keep_separator: 구분자를 청크에 포함할지 여부

        Raises:
            ValueError: chunk_overlap >= chunk_size인 경우

        Example:
            >>> # 기본 설정 사용
            >>> chunker = TextChunker()

            >>> # 커스텀 설정
            >>> chunker = TextChunker(
            ...     chunk_size=1000,
            ...     chunk_overlap=100,
            ...     splitter_type=SplitterType.RECURSIVE
            ... )
        """
        # 설정에서 기본값 로드
        settings = get_settings()

        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.splitter_type = splitter_type
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator

        # 오버랩 검증
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap})은 "
                f"chunk_size({self.chunk_size})보다 작아야 합니다"
            )

        # 분할기 생성
        self._splitter = self._create_splitter()

        self.logger.info(
            f"TextChunker 초기화됨: "
            f"size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"type={self.splitter_type.value}"
        )

    def _create_splitter(self):
        """
        설정에 따른 분할기 인스턴스 생성

        Returns:
            TextSplitter: LangChain 텍스트 분할기 인스턴스
        """
        if self.splitter_type == SplitterType.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=self.separators,
                keep_separator=self.keep_separator,
            )

        elif self.splitter_type == SplitterType.CHARACTER:
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separator="\n\n",  # 단락 기준
            )

        else:
            # 기본값: Recursive
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=self.separators,
                keep_separator=self.keep_separator,
            )

    def split_documents(
        self,
        documents: List[Document],
        add_chunk_metadata: bool = True
    ) -> List[Document]:
        """
        Document 목록을 청크로 분할

        각 Document를 지정된 크기의 청크로 분할합니다.
        원본 메타데이터는 보존되고, 청크 관련 메타데이터가 추가됩니다.

        Args:
            documents: 분할할 Document 목록
            add_chunk_metadata: 청크 인덱스 등 메타데이터 추가 여부

        Returns:
            List[Document]: 분할된 청크 Document 목록

        Example:
            >>> chunks = chunker.split_documents(documents)
            >>> for chunk in chunks:
            ...     print(f"청크 {chunk.metadata.get('chunk_index')}: "
            ...           f"{len(chunk.page_content)}자")
        """
        if not documents:
            self.logger.warning("분할할 문서가 없습니다")
            return []

        self.logger.info(f"문서 분할 시작: {len(documents)}개 문서")

        # 분할 수행
        chunks = self._splitter.split_documents(documents)

        # 청크 메타데이터 추가
        # 주의: ChromaDB는 메타데이터 값으로 str, int, float, bool만 허용
        #       딕셔너리나 리스트는 저장할 수 없음
        if add_chunk_metadata:
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
                chunk.metadata["total_chunks"] = len(chunks)

                # 청킹 설정 정보 추가 (개별 필드로 저장)
                chunk.metadata["config_chunk_size"] = self.chunk_size
                chunk.metadata["config_chunk_overlap"] = self.chunk_overlap
                chunk.metadata["config_splitter_type"] = self.splitter_type.value

        self.logger.info(
            f"문서 분할 완료: {len(documents)}개 → {len(chunks)}개 청크"
        )

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        순수 텍스트를 청크로 분할

        Document 객체 없이 텍스트만 분할할 때 사용합니다.

        Args:
            text: 분할할 텍스트

        Returns:
            List[str]: 분할된 텍스트 청크 목록

        Example:
            >>> text = "긴 텍스트 내용..."
            >>> chunks = chunker.split_text(text)
            >>> for i, chunk in enumerate(chunks):
            ...     print(f"청크 {i}: {chunk[:50]}...")
        """
        if not text:
            return []

        return self._splitter.split_text(text)

    def split_text_to_documents(
        self,
        text: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        텍스트를 Document 청크로 변환

        순수 텍스트를 분할하고 Document 객체로 변환합니다.

        Args:
            text: 분할할 텍스트
            metadata: 모든 청크에 추가할 메타데이터

        Returns:
            List[Document]: Document 청크 목록

        Example:
            >>> chunks = chunker.split_text_to_documents(
            ...     text="긴 텍스트...",
            ...     metadata={"source": "user_input"}
            ... )
        """
        if not text:
            return []

        # 텍스트 분할
        text_chunks = self.split_text(text)

        # Document 생성
        documents = []
        for i, chunk in enumerate(text_chunks):
            doc_metadata = {
                "chunk_index": i,
                "chunk_size": len(chunk),
                "total_chunks": len(text_chunks),
            }
            if metadata:
                doc_metadata.update(metadata)

            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))

        return documents

    def estimate_chunk_count(
        self,
        documents: List[Document]
    ) -> int:
        """
        예상 청크 수 계산

        실제 분할 없이 대략적인 청크 수를 추정합니다.

        Args:
            documents: Document 목록

        Returns:
            int: 예상 청크 수

        Example:
            >>> count = chunker.estimate_chunk_count(documents)
            >>> print(f"약 {count}개의 청크가 생성될 예정입니다")
        """
        total_chars = sum(len(doc.page_content) for doc in documents)
        effective_chunk_size = self.chunk_size - self.chunk_overlap

        if effective_chunk_size <= 0:
            return len(documents)

        estimated = max(1, total_chars // effective_chunk_size)
        return estimated

    @property
    def config(self) -> dict:
        """
        현재 청킹 설정 반환

        Returns:
            dict: 청킹 설정 정보
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "splitter_type": self.splitter_type.value,
            "separators_count": len(self.separators),
            "keep_separator": self.keep_separator,
        }

    def with_config(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        splitter_type: Optional[SplitterType] = None
    ) -> "TextChunker":
        """
        새로운 설정으로 TextChunker 복사본 생성

        원본은 변경하지 않고 새 인스턴스를 생성합니다.

        Args:
            chunk_size: 새 청크 크기 (None이면 현재 값 유지)
            chunk_overlap: 새 오버랩 (None이면 현재 값 유지)
            splitter_type: 새 분할기 유형 (None이면 현재 값 유지)

        Returns:
            TextChunker: 새 설정이 적용된 인스턴스

        Example:
            >>> # 더 작은 청크로 분할
            >>> small_chunker = chunker.with_config(chunk_size=256)
        """
        return TextChunker(
            chunk_size=chunk_size or self.chunk_size,
            chunk_overlap=chunk_overlap or self.chunk_overlap,
            splitter_type=splitter_type or self.splitter_type,
            separators=self.separators,
            keep_separator=self.keep_separator,
        )

    @classmethod
    def for_markdown(
        cls,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> "TextChunker":
        """
        마크다운 문서용 청커 생성

        마크다운 헤더를 구분자로 사용하여 섹션별로 분할합니다.

        Args:
            chunk_size: 청크 크기
            chunk_overlap: 오버랩

        Returns:
            TextChunker: 마크다운 최적화 청커

        Example:
            >>> md_chunker = TextChunker.for_markdown()
            >>> chunks = md_chunker.split_documents(markdown_docs)
        """
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type=SplitterType.RECURSIVE,
            separators=cls.MARKDOWN_SEPARATORS,
        )

    @classmethod
    def for_code(
        cls,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> "TextChunker":
        """
        코드용 청커 생성

        코드 블록을 고려한 분할을 수행합니다.

        Args:
            chunk_size: 청크 크기 (코드는 더 큰 청크 권장)
            chunk_overlap: 오버랩

        Returns:
            TextChunker: 코드 최적화 청커

        Example:
            >>> code_chunker = TextChunker.for_code()
            >>> chunks = code_chunker.split_documents(code_docs)
        """
        code_separators = [
            "\nclass ",     # 클래스 정의
            "\ndef ",       # 함수 정의
            "\n\n",         # 빈 줄
            "\n",           # 줄바꿈
            " ",            # 공백
            "",             # 문자
        ]

        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type=SplitterType.RECURSIVE,
            separators=code_separators,
        )

    def __repr__(self) -> str:
        """청커의 문자열 표현"""
        return (
            f"TextChunker("
            f"size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, "
            f"type={self.splitter_type.value})"
        )
