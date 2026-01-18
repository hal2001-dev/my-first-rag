"""
=============================================================================
커스텀 예외 클래스 정의
=============================================================================

이 모듈은 RAG 시스템에서 사용하는 모든 커스텀 예외를 정의합니다.
예외 계층 구조를 통해 세밀한 에러 핸들링이 가능합니다.

예외 계층 구조:
    RAGException (기본 예외)
    ├── ConfigurationError     # 설정 관련 오류
    ├── DocumentLoadError      # 문서 로딩 오류
    │   └── UnsupportedSourceError  # 지원하지 않는 소스
    ├── EmbeddingError        # 임베딩 생성 오류
    ├── RetrievalError        # 문서 검색 오류
    └── LLMError              # LLM 응답 생성 오류

사용 예시:
    >>> from app.utils.exceptions import DocumentLoadError
    >>> try:
    ...     load_document("invalid.xyz")
    ... except DocumentLoadError as e:
    ...     print(f"문서 로딩 실패: {e}")

    >>> # 모든 RAG 관련 예외 처리
    >>> try:
    ...     process_query(question)
    ... except RAGException as e:
    ...     print(f"RAG 시스템 오류: {e}")
"""

from typing import Optional, Any


class RAGException(Exception):
    """
    RAG 시스템의 기본 예외 클래스

    모든 커스텀 예외는 이 클래스를 상속받습니다.
    이를 통해 RAG 관련 모든 예외를 한 번에 처리할 수 있습니다.

    Attributes:
        message (str): 사용자에게 표시할 에러 메시지
        details (Optional[Any]): 추가적인 에러 상세 정보
        original_error (Optional[Exception]): 원본 예외 (체이닝된 경우)

    Example:
        >>> raise RAGException("처리 중 오류 발생", details={"step": "embedding"})
    """

    def __init__(
        self,
        message: str,
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None
    ):
        """
        RAGException 초기화

        Args:
            message: 에러 메시지
            details: 추가 상세 정보 (딕셔너리, 리스트 등)
            original_error: 원본 예외 객체
        """
        super().__init__(message)
        self.message = message
        self.details = details
        self.original_error = original_error

    def __str__(self) -> str:
        """사람이 읽기 쉬운 에러 메시지 반환"""
        result = self.message
        if self.details:
            result += f" (상세: {self.details})"
        if self.original_error:
            result += f" [원인: {self.original_error}]"
        return result

    def to_dict(self) -> dict:
        """
        예외 정보를 딕셔너리로 변환 (API 응답이나 로깅에 유용)

        Returns:
            dict: 예외 정보가 담긴 딕셔너리
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None,
        }


class ConfigurationError(RAGException):
    """
    설정 관련 예외

    발생 상황:
        - 필수 환경 변수 누락
        - 잘못된 설정 값
        - API 키 미설정

    Example:
        >>> raise ConfigurationError(
        ...     "OpenAI API 키가 설정되지 않았습니다",
        ...     details={"required_env": "OPENAI_API_KEY"}
        ... )
    """
    pass


class DocumentLoadError(RAGException):
    """
    문서 로딩 예외

    발생 상황:
        - 파일을 찾을 수 없음
        - 파일 읽기 권한 없음
        - 파일 형식 파싱 실패
        - 네트워크 오류 (웹 문서)

    Example:
        >>> raise DocumentLoadError(
        ...     "PDF 파일을 로드할 수 없습니다",
        ...     details={"path": "/path/to/file.pdf", "reason": "file_not_found"}
        ... )
    """

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None
    ):
        """
        DocumentLoadError 초기화

        Args:
            message: 에러 메시지
            source: 로드 실패한 소스 (파일 경로 또는 URL)
            details: 추가 상세 정보
            original_error: 원본 예외
        """
        # details에 source 정보 추가
        if source:
            details = details or {}
            if isinstance(details, dict):
                details["source"] = source
            else:
                details = {"source": source, "additional": details}

        super().__init__(message, details, original_error)
        self.source = source


class UnsupportedSourceError(DocumentLoadError):
    """
    지원하지 않는 소스 형식 예외

    발생 상황:
        - 지원하지 않는 파일 확장자
        - 알 수 없는 URL 형식
        - 처리할 수 없는 콘텐츠 타입

    지원 형식 추가 방법:
        1. 새로운 Loader 클래스 구현
        2. LoaderFactory에 등록

    Example:
        >>> raise UnsupportedSourceError(
        ...     "지원하지 않는 파일 형식입니다",
        ...     source="document.xyz",
        ...     supported_formats=[".pdf", ".txt", ".md"]
        ... )
    """

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        supported_formats: Optional[list] = None,
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None
    ):
        """
        UnsupportedSourceError 초기화

        Args:
            message: 에러 메시지
            source: 처리 실패한 소스
            supported_formats: 지원하는 형식 목록
            details: 추가 상세 정보
            original_error: 원본 예외
        """
        details = details or {}
        if isinstance(details, dict) and supported_formats:
            details["supported_formats"] = supported_formats

        super().__init__(message, source, details, original_error)
        self.supported_formats = supported_formats


class EmbeddingError(RAGException):
    """
    임베딩 생성 예외

    발생 상황:
        - 임베딩 API 호출 실패
        - 토큰 수 초과
        - 모델 로딩 실패 (로컬 모델)
        - 네트워크 오류

    Example:
        >>> raise EmbeddingError(
        ...     "임베딩 생성에 실패했습니다",
        ...     details={"model": "text-embedding-3-small", "reason": "rate_limit"}
        ... )
    """

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None
    ):
        """
        EmbeddingError 초기화

        Args:
            message: 에러 메시지
            model: 사용 시도한 임베딩 모델명
            details: 추가 상세 정보
            original_error: 원본 예외
        """
        if model:
            details = details or {}
            if isinstance(details, dict):
                details["model"] = model

        super().__init__(message, details, original_error)
        self.model = model


class RetrievalError(RAGException):
    """
    문서 검색 예외

    발생 상황:
        - 벡터 스토어 연결 실패
        - 검색 쿼리 실행 오류
        - 컬렉션 없음

    Example:
        >>> raise RetrievalError(
        ...     "문서 검색에 실패했습니다",
        ...     details={"collection": "rag_documents", "query": "검색어"}
        ... )
    """

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None
    ):
        """
        RetrievalError 초기화

        Args:
            message: 에러 메시지
            query: 실패한 검색 쿼리
            details: 추가 상세 정보
            original_error: 원본 예외
        """
        if query:
            details = details or {}
            if isinstance(details, dict):
                # 쿼리가 너무 길면 잘라서 저장 (로그 가독성)
                details["query"] = query[:100] + "..." if len(query) > 100 else query

        super().__init__(message, details, original_error)
        self.query = query


class LLMError(RAGException):
    """
    LLM 응답 생성 예외

    발생 상황:
        - LLM API 호출 실패
        - 토큰 수 초과
        - 콘텐츠 필터링 (부적절한 콘텐츠)
        - 타임아웃
        - Rate Limit 초과

    Example:
        >>> raise LLMError(
        ...     "LLM 응답 생성에 실패했습니다",
        ...     model="gpt-4-turbo-preview",
        ...     details={"reason": "context_length_exceeded"}
        ... )
    """

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None
    ):
        """
        LLMError 초기화

        Args:
            message: 에러 메시지
            model: 사용 시도한 LLM 모델명
            provider: LLM 제공자 (openai, anthropic, ollama)
            details: 추가 상세 정보
            original_error: 원본 예외
        """
        details = details or {}
        if isinstance(details, dict):
            if model:
                details["model"] = model
            if provider:
                details["provider"] = provider

        super().__init__(message, details, original_error)
        self.model = model
        self.provider = provider


class VectorStoreError(RAGException):
    """
    벡터 스토어 관련 예외

    발생 상황:
        - ChromaDB 연결 실패
        - 컬렉션 생성/삭제 오류
        - 데이터 저장 실패
        - 영구 저장소 접근 오류

    Example:
        >>> raise VectorStoreError(
        ...     "ChromaDB에 연결할 수 없습니다",
        ...     details={"persist_directory": "./data/chroma_db"}
        ... )
    """

    def __init__(
        self,
        message: str,
        collection_name: Optional[str] = None,
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None
    ):
        """
        VectorStoreError 초기화

        Args:
            message: 에러 메시지
            collection_name: 관련 컬렉션 이름
            details: 추가 상세 정보
            original_error: 원본 예외
        """
        if collection_name:
            details = details or {}
            if isinstance(details, dict):
                details["collection_name"] = collection_name

        super().__init__(message, details, original_error)
        self.collection_name = collection_name
