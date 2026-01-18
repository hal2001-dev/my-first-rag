"""
=============================================================================
임베딩 생성 모듈
=============================================================================

이 모듈은 텍스트를 벡터(임베딩)로 변환하는 기능을 제공합니다.
임베딩은 텍스트의 의미를 수치 벡터로 표현한 것으로,
유사한 의미의 텍스트는 유사한 벡터를 가집니다.

임베딩 활용:
    - 유사 문서 검색: 질문과 가장 유사한 문서 청크 찾기
    - 클러스터링: 의미적으로 비슷한 문서 그룹화
    - 분류: 문서 카테고리 자동 분류

지원 제공자:
    - OpenAI: text-embedding-3-small, text-embedding-3-large (클라우드)
    - Ollama: nomic-embed-text 등 (로컬)
    - HuggingFace: sentence-transformers 모델들 (로컬/클라우드)

제공자 교체 방법:
    1. .env에서 EMBEDDING_PROVIDER 변경
    2. 해당 제공자의 설정 추가
    3. 애플리케이션 재시작

설계 원칙:
    - 추상 베이스 클래스로 제공자 독립적인 인터페이스
    - 팩토리 함수로 설정 기반 자동 선택
    - 각 제공자는 독립적으로 테스트 가능

사용 예시:
    >>> from app.core.embeddings import get_embedding_generator
    >>> generator = get_embedding_generator()  # 설정 기반 자동 선택
    >>> embedding = generator.embed_query("검색어")
    >>> print(len(embedding))  # 벡터 차원 수 출력
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from app.config import get_settings, EmbeddingProvider
from app.utils.logger import get_logger, LoggerMixin
from app.utils.exceptions import EmbeddingError, ConfigurationError

# 모듈 레벨 로거
logger = get_logger(__name__)


# =============================================================================
# 추상 베이스 클래스
# =============================================================================

class BaseEmbeddingGenerator(ABC, LoggerMixin):
    """
    임베딩 생성기의 추상 베이스 클래스

    모든 임베딩 제공자는 이 클래스를 상속받아 구현해야 합니다.
    이를 통해 제공자에 관계없이 동일한 인터페이스로 사용할 수 있습니다.

    새로운 임베딩 제공자 추가 방법:
        1. 이 클래스를 상속받는 새 클래스 생성
        2. embed_documents()와 embed_query() 메서드 구현
        3. get_embedding_generator() 팩토리 함수에 추가

    Attributes:
        model (str): 사용 중인 임베딩 모델명
        dimension (int): 임베딩 벡터의 차원 수

    Example:
        >>> class MyEmbeddingGenerator(BaseEmbeddingGenerator):
        ...     def embed_documents(self, texts):
        ...         # 구현
        ...         pass
        ...     def embed_query(self, text):
        ...         # 구현
        ...         pass
    """

    def __init__(self, model: str):
        """
        BaseEmbeddingGenerator 초기화

        Args:
            model: 사용할 임베딩 모델명
        """
        self.model = model
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> Optional[int]:
        """
        임베딩 벡터의 차원 수

        Returns:
            int: 벡터 차원 수. 아직 임베딩을 생성하지 않았으면 None
        """
        return self._dimension

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서의 임베딩 생성

        문서 저장 시 사용됩니다. 배치 처리로 효율성을 높입니다.

        Args:
            texts: 임베딩할 텍스트 목록

        Returns:
            List[List[float]]: 각 텍스트의 임베딩 벡터 목록

        Raises:
            EmbeddingError: 임베딩 생성 실패 시
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리의 임베딩 생성

        검색 시 질문을 임베딩할 때 사용됩니다.

        Args:
            text: 임베딩할 쿼리 텍스트

        Returns:
            List[float]: 쿼리의 임베딩 벡터

        Raises:
            EmbeddingError: 임베딩 생성 실패 시
        """
        pass

    @property
    @abstractmethod
    def embedding_function(self):
        """
        LangChain/ChromaDB 호환 임베딩 함수 반환

        벡터 스토어에 전달하기 위한 임베딩 함수입니다.

        Returns:
            임베딩 함수 객체 (LangChain Embeddings 인터페이스)
        """
        pass


# =============================================================================
# OpenAI 임베딩 구현
# =============================================================================

class OpenAIEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    OpenAI 임베딩 생성기

    OpenAI의 text-embedding 모델을 사용하여 임베딩을 생성합니다.

    지원 모델:
        - text-embedding-3-small: 저렴하고 빠름 (1536차원)
        - text-embedding-3-large: 고성능, 더 정확함 (3072차원)
        - text-embedding-ada-002: 레거시 모델 (1536차원)

    가격 (2024년 기준):
        - text-embedding-3-small: $0.02 / 1M tokens
        - text-embedding-3-large: $0.13 / 1M tokens

    Attributes:
        model (str): OpenAI 임베딩 모델명
        _embeddings: LangChain OpenAIEmbeddings 인스턴스

    Example:
        >>> generator = OpenAIEmbeddingGenerator(
        ...     api_key="sk-xxx",
        ...     model="text-embedding-3-small"
        ... )
        >>> embedding = generator.embed_query("Hello, world!")
        >>> print(len(embedding))  # 1536
    """

    # 모델별 벡터 차원 정보 (참조용)
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small"
    ):
        """
        OpenAIEmbeddingGenerator 초기화

        Args:
            api_key: OpenAI API 키
            model: 사용할 임베딩 모델명

        Raises:
            ConfigurationError: API 키가 유효하지 않은 경우
        """
        super().__init__(model)

        # API 키 검증
        if not api_key or not api_key.startswith("sk-"):
            raise ConfigurationError(
                "유효하지 않은 OpenAI API 키입니다. "
                "'sk-'로 시작하는 키를 사용하세요."
            )

        self.logger.info(f"OpenAI 임베딩 생성기 초기화: model={model}")

        try:
            # LangChain OpenAI 임베딩 인스턴스 생성
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key,
            )

            # 알려진 모델이면 차원 정보 설정
            self._dimension = self.MODEL_DIMENSIONS.get(model)

        except ImportError as e:
            raise ConfigurationError(
                "langchain-openai 패키지가 설치되지 않았습니다. "
                "pip install langchain-openai 를 실행하세요.",
                original_error=e
            )
        except Exception as e:
            raise EmbeddingError(
                "OpenAI 임베딩 초기화에 실패했습니다",
                model=model,
                original_error=e
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서의 임베딩 생성 (OpenAI)

        Args:
            texts: 임베딩할 텍스트 목록

        Returns:
            List[List[float]]: 임베딩 벡터 목록

        Raises:
            EmbeddingError: API 호출 실패 시
        """
        if not texts:
            return []

        self.logger.debug(f"문서 {len(texts)}개의 임베딩 생성 중...")

        try:
            embeddings = self._embeddings.embed_documents(texts)

            # 첫 번째 임베딩의 차원으로 dimension 설정
            if embeddings and self._dimension is None:
                self._dimension = len(embeddings[0])

            self.logger.debug(
                f"임베딩 생성 완료: {len(embeddings)}개, 차원={self._dimension}"
            )
            return embeddings

        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            raise EmbeddingError(
                "문서 임베딩 생성에 실패했습니다",
                model=self.model,
                details={"document_count": len(texts)},
                original_error=e
            )

    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리의 임베딩 생성 (OpenAI)

        Args:
            text: 임베딩할 쿼리 텍스트

        Returns:
            List[float]: 쿼리의 임베딩 벡터

        Raises:
            EmbeddingError: API 호출 실패 시
        """
        if not text:
            raise EmbeddingError("빈 텍스트는 임베딩할 수 없습니다")

        self.logger.debug(f"쿼리 임베딩 생성 중: '{text[:50]}...'")

        try:
            embedding = self._embeddings.embed_query(text)

            # 차원 정보 업데이트
            if self._dimension is None:
                self._dimension = len(embedding)

            return embedding

        except Exception as e:
            self.logger.error(f"쿼리 임베딩 실패: {e}")
            raise EmbeddingError(
                "쿼리 임베딩 생성에 실패했습니다",
                model=self.model,
                details={"query_length": len(text)},
                original_error=e
            )

    @property
    def embedding_function(self):
        """LangChain 호환 임베딩 함수 반환"""
        return self._embeddings


# =============================================================================
# Ollama 임베딩 구현 (로컬 모델)
# =============================================================================

class OllamaEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Ollama 임베딩 생성기 (로컬 모델)

    로컬에서 실행되는 Ollama 서버를 통해 임베딩을 생성합니다.
    인터넷 연결 없이 사용 가능하며, 비용이 발생하지 않습니다.

    사전 요구사항:
        1. Ollama 설치: https://ollama.com/download
        2. 임베딩 모델 다운로드: ollama pull nomic-embed-text
        3. Ollama 서버 실행: ollama serve

    지원 모델:
        - nomic-embed-text: 높은 성능의 범용 임베딩 (768차원)
        - mxbai-embed-large: 대형 임베딩 모델
        - all-minilm: 빠르고 가벼운 모델

    Attributes:
        model (str): Ollama 임베딩 모델명
        base_url (str): Ollama 서버 URL

    Example:
        >>> generator = OllamaEmbeddingGenerator(
        ...     model="nomic-embed-text",
        ...     base_url="http://localhost:11434"
        ... )
        >>> embedding = generator.embed_query("Hello!")
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ):
        """
        OllamaEmbeddingGenerator 초기화

        Args:
            model: Ollama 임베딩 모델명
            base_url: Ollama 서버 URL

        Raises:
            ConfigurationError: Ollama 연결 실패 시
        """
        super().__init__(model)
        self.base_url = base_url

        self.logger.info(
            f"Ollama 임베딩 생성기 초기화: model={model}, url={base_url}"
        )

        try:
            # LangChain Ollama 임베딩 인스턴스 생성
            from langchain_community.embeddings import OllamaEmbeddings

            self._embeddings = OllamaEmbeddings(
                model=model,
                base_url=base_url,
            )

        except ImportError as e:
            raise ConfigurationError(
                "langchain-community 패키지가 설치되지 않았습니다. "
                "pip install langchain-community 를 실행하세요.",
                original_error=e
            )
        except Exception as e:
            raise EmbeddingError(
                f"Ollama 연결에 실패했습니다. "
                f"Ollama가 실행 중인지 확인하세요 (ollama serve). "
                f"URL: {base_url}",
                model=model,
                original_error=e
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서의 임베딩 생성 (Ollama)

        Args:
            texts: 임베딩할 텍스트 목록

        Returns:
            List[List[float]]: 임베딩 벡터 목록
        """
        if not texts:
            return []

        self.logger.debug(f"Ollama로 문서 {len(texts)}개 임베딩 생성 중...")

        try:
            embeddings = self._embeddings.embed_documents(texts)

            if embeddings and self._dimension is None:
                self._dimension = len(embeddings[0])

            return embeddings

        except Exception as e:
            raise EmbeddingError(
                "Ollama 문서 임베딩 생성에 실패했습니다. "
                "Ollama 서버 상태를 확인하세요.",
                model=self.model,
                original_error=e
            )

    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리의 임베딩 생성 (Ollama)

        Args:
            text: 임베딩할 쿼리 텍스트

        Returns:
            List[float]: 쿼리의 임베딩 벡터
        """
        if not text:
            raise EmbeddingError("빈 텍스트는 임베딩할 수 없습니다")

        try:
            embedding = self._embeddings.embed_query(text)

            if self._dimension is None:
                self._dimension = len(embedding)

            return embedding

        except Exception as e:
            raise EmbeddingError(
                "Ollama 쿼리 임베딩 생성에 실패했습니다",
                model=self.model,
                original_error=e
            )

    @property
    def embedding_function(self):
        """LangChain 호환 임베딩 함수 반환"""
        return self._embeddings


# =============================================================================
# HuggingFace 임베딩 구현
# =============================================================================

class HuggingFaceEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    HuggingFace 임베딩 생성기

    HuggingFace의 sentence-transformers 모델을 사용합니다.
    로컬에서 실행되므로 API 비용이 발생하지 않습니다.

    인기 모델:
        - all-MiniLM-L6-v2: 빠르고 가벼움 (384차원)
        - all-mpnet-base-v2: 균형 잡힌 성능 (768차원)
        - paraphrase-multilingual-MiniLM-L12-v2: 다국어 지원

    주의:
        - 첫 실행 시 모델 다운로드에 시간이 걸릴 수 있음
        - GPU 사용 시 CUDA 설치 필요

    Attributes:
        model (str): HuggingFace 모델명

    Example:
        >>> generator = HuggingFaceEmbeddingGenerator(
        ...     model="all-MiniLM-L6-v2"
        ... )
        >>> embedding = generator.embed_query("Hello!")
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """
        HuggingFaceEmbeddingGenerator 초기화

        Args:
            model: HuggingFace 모델명 (sentence-transformers 호환)

        Raises:
            ConfigurationError: 의존성 누락 시
        """
        super().__init__(model)

        self.logger.info(f"HuggingFace 임베딩 생성기 초기화: model={model}")

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},  # GPU 사용 시 "cuda"
                encode_kwargs={"normalize_embeddings": True},
            )

        except ImportError as e:
            raise ConfigurationError(
                "HuggingFace 임베딩을 사용하려면 추가 패키지가 필요합니다. "
                "pip install sentence-transformers 를 실행하세요.",
                original_error=e
            )
        except Exception as e:
            raise EmbeddingError(
                f"HuggingFace 모델 로드에 실패했습니다: {model}",
                model=model,
                original_error=e
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서의 임베딩 생성 (HuggingFace)"""
        if not texts:
            return []

        try:
            embeddings = self._embeddings.embed_documents(texts)

            if embeddings and self._dimension is None:
                self._dimension = len(embeddings[0])

            return embeddings

        except Exception as e:
            raise EmbeddingError(
                "HuggingFace 문서 임베딩 생성에 실패했습니다",
                model=self.model,
                original_error=e
            )

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리의 임베딩 생성 (HuggingFace)"""
        if not text:
            raise EmbeddingError("빈 텍스트는 임베딩할 수 없습니다")

        try:
            embedding = self._embeddings.embed_query(text)

            if self._dimension is None:
                self._dimension = len(embedding)

            return embedding

        except Exception as e:
            raise EmbeddingError(
                "HuggingFace 쿼리 임베딩 생성에 실패했습니다",
                model=self.model,
                original_error=e
            )

    @property
    def embedding_function(self):
        """LangChain 호환 임베딩 함수 반환"""
        return self._embeddings


# =============================================================================
# 팩토리 함수
# =============================================================================

def get_embedding_generator(
    provider: Optional[EmbeddingProvider] = None
) -> BaseEmbeddingGenerator:
    """
    설정 기반으로 적절한 임베딩 생성기 반환

    이 함수는 팩토리 패턴을 사용하여 설정에 따라
    적절한 임베딩 생성기를 자동으로 선택합니다.

    Args:
        provider: 임베딩 제공자. None이면 설정 파일에서 읽음

    Returns:
        BaseEmbeddingGenerator: 임베딩 생성기 인스턴스

    Raises:
        ConfigurationError: 알 수 없는 제공자이거나 설정 오류 시

    Example:
        >>> # 설정 파일 기반 자동 선택
        >>> generator = get_embedding_generator()

        >>> # 명시적으로 제공자 지정
        >>> generator = get_embedding_generator(EmbeddingProvider.OLLAMA)
    """
    settings = get_settings()
    provider = provider or settings.embedding_provider

    logger.info(f"임베딩 생성기 생성: provider={provider.value}")

    # 제공자별 인스턴스 생성
    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddingGenerator(
            api_key=settings.get_openai_api_key(),
            model=settings.openai_embedding_model,
        )

    elif provider == EmbeddingProvider.OLLAMA:
        return OllamaEmbeddingGenerator(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    elif provider == EmbeddingProvider.HUGGINGFACE:
        return HuggingFaceEmbeddingGenerator(
            model="all-MiniLM-L6-v2"  # 기본 모델
        )

    else:
        raise ConfigurationError(
            f"지원하지 않는 임베딩 제공자입니다: {provider}. "
            f"지원 목록: {[p.value for p in EmbeddingProvider]}"
        )


# =============================================================================
# 편의 클래스 (하위 호환성)
# =============================================================================

class EmbeddingGenerator:
    """
    임베딩 생성기 래퍼 클래스 (하위 호환성 유지)

    직접 사용하기보다 get_embedding_generator() 함수 사용을 권장합니다.
    이 클래스는 기존 코드와의 호환성을 위해 유지됩니다.

    Example:
        >>> generator = EmbeddingGenerator()  # 기본 설정 사용
        >>> embedding = generator.embed_query("Hello")
    """

    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        """래퍼 초기화 - 내부적으로 팩토리 함수 사용"""
        self._generator = get_embedding_generator(provider)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩 생성"""
        return self._generator.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩 생성"""
        return self._generator.embed_query(text)

    @property
    def embedding_function(self):
        """LangChain 호환 임베딩 함수"""
        return self._generator.embedding_function

    @property
    def model(self) -> str:
        """사용 중인 모델명"""
        return self._generator.model

    @property
    def dimension(self) -> Optional[int]:
        """임베딩 벡터 차원"""
        return self._generator.dimension
