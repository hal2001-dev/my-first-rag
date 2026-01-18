"""
=============================================================================
로깅 설정 모듈
=============================================================================

이 모듈은 애플리케이션 전체에서 사용할 로깅 시스템을 설정합니다.

로그 레벨 (심각도 순):
    - DEBUG: 상세한 디버깅 정보 (개발 시에만 사용)
    - INFO: 일반적인 정보 메시지 (기본값)
    - WARNING: 경고 메시지 (문제가 될 수 있는 상황)
    - ERROR: 에러 메시지 (처리 가능한 오류)
    - CRITICAL: 치명적 오류 (프로그램 종료가 필요한 수준)

로그 형식:
    2024-01-15 10:30:45 - app.core.rag_chain - INFO - 문서 3개 처리 완료

사용 예시:
    >>> from app.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("처리 시작")
    >>> logger.debug("상세 정보: %s", details)
    >>> logger.error("오류 발생: %s", error, exc_info=True)
"""

import logging
import sys
from typing import Optional
from pathlib import Path

# 로그 포맷 상수
# - asctime: 타임스탬프
# - name: 로거 이름 (보통 모듈 경로)
# - levelname: 로그 레벨
# - message: 실제 메시지
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 간단한 로그 포맷 (콘솔용)
SIMPLE_LOG_FORMAT = "%(levelname)s - %(message)s"

# 날짜/시간 포맷
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    use_simple_format: bool = False
) -> None:
    """
    애플리케이션 로깅 시스템 초기화

    이 함수는 애플리케이션 시작 시 한 번만 호출해야 합니다.
    여러 번 호출해도 안전하지만, 핸들러가 중복 추가될 수 있습니다.

    Args:
        level: 로그 레벨 문자열. None이면 설정 파일에서 읽음
               ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: 로그 파일 경로. None이면 콘솔에만 출력
        use_simple_format: True면 간단한 포맷 사용 (개발용)

    Example:
        >>> # 기본 설정으로 초기화
        >>> setup_logging()

        >>> # 디버그 레벨로 초기화
        >>> setup_logging(level="DEBUG")

        >>> # 파일과 콘솔 동시 출력
        >>> setup_logging(log_file=Path("app.log"))
    """
    # 설정에서 로그 레벨 가져오기
    if level is None:
        try:
            from app.config import get_settings
            settings = get_settings()
            level = settings.log_level
        except Exception:
            # 설정 로드 실패 시 기본값 사용
            level = "INFO"

    # 문자열을 로깅 레벨 상수로 변환
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 포맷터 선택
    log_format = SIMPLE_LOG_FORMAT if use_simple_format else DEFAULT_LOG_FORMAT

    # 루트 로거 설정
    # 루트 로거를 설정하면 모든 하위 로거에 적용됨
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 포맷터 생성
    formatter = logging.Formatter(log_format, datefmt=DATE_FORMAT)

    # 콘솔 핸들러 추가
    # StreamHandler는 기본적으로 stderr로 출력
    # stdout으로 변경하여 일반 출력과 함께 볼 수 있게 함
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 파일 핸들러 추가 (옵션)
    if log_file:
        # 로그 파일 디렉토리 생성
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # FileHandler는 기본적으로 append 모드
        # 로그 파일이 커지면 RotatingFileHandler 사용 권장
        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
            mode="a"  # append 모드
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 외부 라이브러리 로그 레벨 조정
    # 이 라이브러리들은 너무 많은 로그를 출력하므로 WARNING 이상만 표시
    noisy_loggers = [
        "httpx",              # HTTP 클라이언트 라이브러리
        "httpcore",           # HTTP 코어 라이브러리
        "urllib3",            # URL 라이브러리
        "openai",             # OpenAI SDK
        "chromadb",           # ChromaDB
        "langchain",          # LangChain
        "langchain_core",     # LangChain Core
        "langchain_community",# LangChain Community
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 인스턴스 반환

    각 모듈에서 이 함수를 호출하여 로거를 가져옵니다.
    로거 이름으로 모듈 경로를 사용하면 로그에서 출처를 쉽게 파악할 수 있습니다.

    Args:
        name: 로거 이름. 보통 __name__ 사용

    Returns:
        logging.Logger: 로거 인스턴스

    Example:
        >>> # 모듈 상단에서 로거 생성
        >>> logger = get_logger(__name__)

        >>> # 이후 로깅 사용
        >>> logger.info("작업 시작")
        >>> logger.debug("상세 정보: %s", data)
        >>> logger.warning("주의: %s", warning_message)
        >>> logger.error("오류 발생", exc_info=True)

    Tip:
        - __name__을 사용하면 모듈의 전체 경로가 로거 이름이 됨
          예: "app.core.rag_chain"
        - exc_info=True 옵션으로 스택 트레이스 포함 가능
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    로거 기능을 클래스에 추가하는 믹스인

    이 믹스인을 상속받으면 self.logger로 로거에 접근할 수 있습니다.
    로거 이름은 자동으로 "모듈명.클래스명" 형식이 됩니다.

    Example:
        >>> class MyProcessor(LoggerMixin):
        ...     def process(self):
        ...         self.logger.info("처리 시작")
        ...         # 작업 수행
        ...         self.logger.info("처리 완료")
    """

    @property
    def logger(self) -> logging.Logger:
        """
        클래스별 로거 반환 (지연 초기화)

        Returns:
            logging.Logger: 클래스명이 포함된 로거
        """
        # _logger가 없으면 생성
        if not hasattr(self, "_logger"):
            # 로거 이름: "모듈명.클래스명" 형식
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(logger_name)
        return self._logger


def log_function_call(logger: logging.Logger):
    """
    함수 호출을 로깅하는 데코레이터

    함수 진입과 종료를 자동으로 로깅합니다.
    디버깅 시 함수 호출 흐름을 추적하는 데 유용합니다.

    Args:
        logger: 사용할 로거 인스턴스

    Returns:
        데코레이터 함수

    Example:
        >>> logger = get_logger(__name__)
        >>>
        >>> @log_function_call(logger)
        ... def process_document(doc_path: str) -> dict:
        ...     # 처리 로직
        ...     return result
        >>>
        >>> # 호출 시 자동 로깅:
        >>> # DEBUG - Calling process_document with args=('doc.pdf',), kwargs={}
        >>> # DEBUG - process_document returned: {...}
    """
    import functools
    from typing import Callable, TypeVar, ParamSpec

    P = ParamSpec("P")
    T = TypeVar("T")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # 함수 진입 로깅
            logger.debug(
                "Calling %s with args=%s, kwargs=%s",
                func.__name__,
                args,
                kwargs
            )

            try:
                # 함수 실행
                result = func(*args, **kwargs)

                # 성공 로깅 (결과가 크면 일부만 표시)
                result_str = str(result)
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                logger.debug("%s returned: %s", func.__name__, result_str)

                return result

            except Exception as e:
                # 예외 로깅
                logger.error(
                    "%s raised %s: %s",
                    func.__name__,
                    type(e).__name__,
                    str(e),
                    exc_info=True
                )
                raise

        return wrapper
    return decorator
