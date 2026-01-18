"""
=============================================================================
설정 관리 모듈
=============================================================================

이 모듈은 애플리케이션의 모든 설정을 중앙에서 관리합니다.
pydantic-settings를 사용하여 환경 변수를 자동으로 로드하고 검증합니다.

설정 로드 순서:
    1. 기본값 (코드에 정의된 값)
    2. .env 파일의 값 (있는 경우)
    3. 환경 변수 (최우선)

모델 교체 방법:
    1. .env 파일에서 LLM_PROVIDER 변경 (openai, ollama, anthropic)
    2. 해당 제공자의 API 키 및 모델명 설정
    3. 애플리케이션 재시작

사용 예시:
    >>> from app.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.llm_provider)  # 'openai'
    >>> print(settings.openai_model)  # 'gpt-4-turbo-preview'
"""

from enum import Enum
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# 열거형 정의 (지원하는 제공자 목록)
# =============================================================================

class LLMProvider(str, Enum):
    """
    지원하는 LLM 제공자 목록

    새로운 LLM 제공자 추가 시:
        1. 여기에 새 값 추가
        2. app/core/llm.py에서 해당 제공자 구현
        3. .env.example에 필요한 환경 변수 추가
    """
    OPENAI = "openai"           # OpenAI GPT 모델 (GPT-4, GPT-3.5 등)
    OLLAMA = "ollama"           # Ollama 로컬 모델 (Llama, Mistral 등)
    ANTHROPIC = "anthropic"     # Anthropic Claude 모델


class EmbeddingProvider(str, Enum):
    """
    지원하는 임베딩 제공자 목록

    새로운 임베딩 제공자 추가 시:
        1. 여기에 새 값 추가
        2. app/core/embeddings.py에서 해당 제공자 구현
    """
    OPENAI = "openai"           # OpenAI 임베딩 (text-embedding-3-small 등)
    OLLAMA = "ollama"           # Ollama 로컬 임베딩 (nomic-embed-text 등)
    HUGGINGFACE = "huggingface" # HuggingFace 임베딩 모델


# =============================================================================
# 설정 클래스
# =============================================================================

class Settings(BaseSettings):
    """
    애플리케이션 전체 설정

    모든 설정은 환경 변수로 오버라이드 가능합니다.
    변수명은 대소문자를 구분하지 않습니다. (OPENAI_API_KEY = openai_api_key)

    Attributes:
        llm_provider: 사용할 LLM 제공자 (openai, ollama, anthropic)
        embedding_provider: 사용할 임베딩 제공자 (openai, ollama, huggingface)
        openai_api_key: OpenAI API 키
        openai_model: OpenAI 모델명
        ... (상세 내용은 각 속성의 docstring 참조)
    """

    # -------------------------------------------------------------------------
    # Pydantic 설정
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",                    # .env 파일 경로
        env_file_encoding="utf-8",          # .env 파일 인코딩
        case_sensitive=False,               # 환경 변수 대소문자 무시
        extra="ignore",                     # 알 수 없는 환경 변수 무시
    )

    # -------------------------------------------------------------------------
    # 제공자 선택 (모델 교체의 핵심 설정)
    # -------------------------------------------------------------------------
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="사용할 LLM 제공자. openai, ollama, anthropic 중 선택"
    )

    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="사용할 임베딩 제공자. openai, ollama, huggingface 중 선택"
    )

    # -------------------------------------------------------------------------
    # OpenAI 설정
    # -------------------------------------------------------------------------
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API 키. https://platform.openai.com/api-keys 에서 발급"
    )

    openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="사용할 OpenAI 모델. gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo 등"
    )

    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI 임베딩 모델. text-embedding-3-small(저렴), text-embedding-3-large(고성능)"
    )

    # -------------------------------------------------------------------------
    # Ollama 설정 (로컬 LLM)
    # -------------------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama 서버 URL. 기본값은 로컬 서버"
    )

    ollama_model: str = Field(
        default="llama3",
        description="Ollama 모델명. ollama list 명령으로 확인 가능"
    )

    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama 임베딩 모델명"
    )

    # -------------------------------------------------------------------------
    # Anthropic 설정 (Claude)
    # -------------------------------------------------------------------------
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API 키"
    )

    anthropic_model: str = Field(
        default="claude-3-sonnet-20240229",
        description="Claude 모델명. claude-3-opus, claude-3-sonnet 등"
    )

    # -------------------------------------------------------------------------
    # ChromaDB 설정
    # -------------------------------------------------------------------------
    chroma_persist_directory: Path = Field(
        default=Path("./data/chroma_db"),
        description="ChromaDB 데이터 영구 저장 경로"
    )

    chroma_collection_name: str = Field(
        default="rag_documents",
        description="ChromaDB 컬렉션(테이블) 이름. 문서 그룹을 구분하는 식별자"
    )

    # -------------------------------------------------------------------------
    # 청킹 설정
    # -------------------------------------------------------------------------
    chunk_size: int = Field(
        default=512,
        ge=100,         # 최소 100자
        le=2000,        # 최대 2000자
        description="청크 크기(문자 수). 너무 작으면 문맥 손실, 너무 크면 검색 정확도 저하"
    )

    chunk_overlap: int = Field(
        default=50,
        ge=0,           # 최소 0 (오버랩 없음)
        le=500,         # 최대 500자
        description="청크 간 겹치는 문자 수. 문맥 유지를 위해 청크 크기의 10-20% 권장"
    )

    # -------------------------------------------------------------------------
    # 검색 설정
    # -------------------------------------------------------------------------
    retrieval_top_k: int = Field(
        default=4,
        ge=1,           # 최소 1개
        le=20,          # 최대 20개
        description="검색 시 반환할 문서 수. 너무 많으면 노이즈 증가, 적으면 정보 부족"
    )

    # -------------------------------------------------------------------------
    # 앱 설정
    # -------------------------------------------------------------------------
    upload_directory: Path = Field(
        default=Path("./data/uploads"),
        description="업로드된 파일 임시 저장 경로"
    )

    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="최대 업로드 파일 크기 (MB)"
    )

    # -------------------------------------------------------------------------
    # 로깅 설정
    # -------------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="로그 레벨. DEBUG, INFO, WARNING, ERROR 중 선택"
    )

    # -------------------------------------------------------------------------
    # 검증 메서드
    # -------------------------------------------------------------------------
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """
        청크 오버랩이 청크 크기보다 작은지 검증

        오버랩이 청크 크기보다 크거나 같으면 무한 루프 발생 가능
        """
        # info.data에서 chunk_size 가져오기 (이미 검증된 경우)
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap({v})은 chunk_size({chunk_size})보다 작아야 합니다"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """로그 레벨이 유효한 값인지 검증"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"log_level은 {valid_levels} 중 하나여야 합니다. 입력값: {v}"
            )
        return v_upper

    # -------------------------------------------------------------------------
    # 유틸리티 메서드
    # -------------------------------------------------------------------------
    def get_openai_api_key(self) -> str:
        """
        OpenAI API 키를 안전하게 반환

        Returns:
            str: OpenAI API 키 문자열

        Raises:
            ValueError: API 키가 설정되지 않은 경우
        """
        if self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 OPENAI_API_KEY=sk-xxx 형식으로 추가하세요."
            )
        return self.openai_api_key.get_secret_value()

    def get_anthropic_api_key(self) -> str:
        """
        Anthropic API 키를 안전하게 반환

        Returns:
            str: Anthropic API 키 문자열

        Raises:
            ValueError: API 키가 설정되지 않은 경우
        """
        if self.anthropic_api_key is None:
            raise ValueError(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 ANTHROPIC_API_KEY=sk-ant-xxx 형식으로 추가하세요."
            )
        return self.anthropic_api_key.get_secret_value()

    def ensure_directories(self) -> None:
        """
        필요한 디렉토리들이 존재하는지 확인하고 없으면 생성

        앱 시작 시 호출하여 필수 디렉토리 구조 보장
        """
        self.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
        self.upload_directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 설정 인스턴스 생성 함수
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    설정 인스턴스를 반환 (싱글톤 패턴)

    lru_cache 데코레이터로 인해 한 번만 생성되고 이후에는 캐시된 인스턴스 반환
    이렇게 하면 매번 .env 파일을 읽지 않아도 됨

    Returns:
        Settings: 애플리케이션 설정 인스턴스

    사용 예시:
        >>> settings = get_settings()
        >>> print(settings.openai_model)
        'gpt-4-turbo-preview'
    """
    return Settings()


def clear_settings_cache() -> None:
    """
    설정 캐시 초기화

    테스트나 설정 변경 후 새로운 설정을 로드해야 할 때 사용

    사용 예시:
        >>> clear_settings_cache()
        >>> new_settings = get_settings()  # 새로 로드됨
    """
    get_settings.cache_clear()
