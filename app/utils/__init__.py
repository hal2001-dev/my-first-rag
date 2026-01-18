"""
=============================================================================
유틸리티 모듈
=============================================================================

이 패키지는 공통으로 사용되는 유틸리티 함수들을 포함합니다.

모듈:
    - logger: 로깅 설정 및 로거 생성
    - exceptions: 커스텀 예외 클래스 정의
"""

from app.utils.logger import get_logger, setup_logging
from app.utils.exceptions import (
    RAGException,
    DocumentLoadError,
    UnsupportedSourceError,
    EmbeddingError,
    RetrievalError,
    LLMError,
    ConfigurationError,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "RAGException",
    "DocumentLoadError",
    "UnsupportedSourceError",
    "EmbeddingError",
    "RetrievalError",
    "LLMError",
    "ConfigurationError",
]
