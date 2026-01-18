"""
=============================================================================
LLM (Large Language Model) 클라이언트 모듈
=============================================================================

이 모듈은 다양한 LLM 제공자와의 통합을 담당합니다.
질문에 대한 답변을 생성하는 핵심 기능을 제공합니다.

지원 제공자:
    - OpenAI: GPT-4, GPT-3.5-turbo 등 (클라우드)
    - Ollama: Llama, Mistral 등 (로컬)
    - Anthropic: Claude 3 시리즈 (클라우드)

LLM 선택 가이드:
    - OpenAI GPT-4: 최고 품질, 높은 비용
    - OpenAI GPT-3.5: 빠르고 저렴, 적당한 품질
    - Ollama: 무료, 프라이버시 보장, 로컬 GPU 필요
    - Claude: GPT-4급 품질, 긴 컨텍스트 지원

제공자 교체 방법:
    1. .env에서 LLM_PROVIDER 변경 (openai, ollama, anthropic)
    2. 해당 제공자의 API 키/모델 설정
    3. 애플리케이션 재시작

사용 예시:
    >>> from app.core.llm import get_llm_client
    >>> client = get_llm_client()  # 설정 기반 자동 선택
    >>> response = client.generate("질문 내용", context="관련 문서 내용")
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config import get_settings, LLMProvider
from app.utils.logger import get_logger, LoggerMixin
from app.utils.exceptions import LLMError, ConfigurationError

# 모듈 레벨 로거
logger = get_logger(__name__)


# =============================================================================
# 기본 시스템 프롬프트
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """당신은 제공된 문맥(context)을 기반으로 질문에 답변하는 도움이 되는 AI 어시스턴트입니다.

규칙:
1. 반드시 제공된 문맥만을 기반으로 답변하세요
2. 문맥에 답변할 정보가 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요
3. 가능하면 출처 문서를 언급하세요
4. 간결하되 충분한 정보를 포함하세요
5. 확실하지 않은 경우 불확실성을 표현하세요
6. 한국어로 답변하세요 (질문이 한국어인 경우)"""


# =============================================================================
# 추상 베이스 클래스
# =============================================================================

class BaseLLMClient(ABC, LoggerMixin):
    """
    LLM 클라이언트의 추상 베이스 클래스

    모든 LLM 제공자는 이 클래스를 상속받아 구현해야 합니다.
    이를 통해 제공자에 관계없이 동일한 인터페이스로 사용할 수 있습니다.

    새로운 LLM 제공자 추가 방법:
        1. 이 클래스를 상속받는 새 클래스 생성
        2. generate() 메서드 구현
        3. get_llm_client() 팩토리 함수에 추가

    Attributes:
        model (str): 사용 중인 모델명
        temperature (float): 응답의 무작위성 (0=결정적, 1=창의적)
        system_prompt (str): 시스템 프롬프트

    Example:
        >>> class MyLLMClient(BaseLLMClient):
        ...     def generate(self, prompt, context=""):
        ...         # 구현
        ...         return response
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None
    ):
        """
        BaseLLMClient 초기화

        Args:
            model: 사용할 LLM 모델명
            temperature: 응답 무작위성 (0.0~1.0)
            system_prompt: 시스템 프롬프트 (None이면 기본값 사용)
        """
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: str = "",
        **kwargs
    ) -> str:
        """
        프롬프트에 대한 응답 생성

        Args:
            prompt: 사용자 질문/프롬프트
            context: 검색된 문서 내용 (RAG에서 사용)
            **kwargs: 추가 생성 파라미터

        Returns:
            str: LLM이 생성한 응답 텍스트

        Raises:
            LLMError: 응답 생성 실패 시
        """
        pass

    @abstractmethod
    def generate_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        context: str = ""
    ) -> str:
        """
        대화 히스토리를 포함한 응답 생성

        Args:
            prompt: 현재 사용자 질문
            history: 이전 대화 히스토리
                    [{"role": "user", "content": "..."}, ...]
            context: 검색된 문서 내용

        Returns:
            str: LLM이 생성한 응답

        Raises:
            LLMError: 응답 생성 실패 시
        """
        pass

    @property
    @abstractmethod
    def llm(self):
        """
        LangChain 호환 LLM 인스턴스 반환

        RAG 체인 구성에 사용됩니다.

        Returns:
            LangChain BaseChatModel 인스턴스
        """
        pass

    def _build_prompt_with_context(self, prompt: str, context: str) -> str:
        """
        컨텍스트를 포함한 프롬프트 구성

        Args:
            prompt: 사용자 질문
            context: 검색된 문서 내용

        Returns:
            str: 구성된 전체 프롬프트
        """
        if context:
            return f"""다음 문맥을 참고하여 질문에 답변해주세요.

문맥:
{context}

질문: {prompt}

답변:"""
        return prompt


# =============================================================================
# OpenAI LLM 구현
# =============================================================================

class OpenAILLMClient(BaseLLMClient):
    """
    OpenAI GPT 모델 클라이언트

    OpenAI의 GPT 모델을 사용하여 응답을 생성합니다.

    지원 모델:
        - gpt-4-turbo-preview: 최신 GPT-4 Turbo (128K 컨텍스트)
        - gpt-4: GPT-4 기본 (8K 컨텍스트)
        - gpt-3.5-turbo: 빠르고 저렴 (16K 컨텍스트)

    가격 (2024년 기준, 1M 토큰당):
        - gpt-4-turbo: 입력 $10, 출력 $30
        - gpt-3.5-turbo: 입력 $0.50, 출력 $1.50

    Attributes:
        model (str): OpenAI 모델명
        _llm: LangChain ChatOpenAI 인스턴스

    Example:
        >>> client = OpenAILLMClient(
        ...     api_key="sk-xxx",
        ...     model="gpt-4-turbo-preview"
        ... )
        >>> response = client.generate("인공지능이란?", context="...")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """
        OpenAILLMClient 초기화

        Args:
            api_key: OpenAI API 키
            model: 사용할 GPT 모델명
            temperature: 응답 무작위성 (0.0~1.0)
            system_prompt: 시스템 프롬프트
            max_tokens: 최대 응답 토큰 수 (None이면 제한 없음)

        Raises:
            ConfigurationError: API 키가 유효하지 않은 경우
        """
        super().__init__(model, temperature, system_prompt)

        # API 키 검증
        if not api_key or not api_key.startswith("sk-"):
            raise ConfigurationError(
                "유효하지 않은 OpenAI API 키입니다. "
                "'sk-'로 시작하는 키를 사용하세요."
            )

        self.logger.info(f"OpenAI LLM 클라이언트 초기화: model={model}")

        try:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key,
                max_tokens=max_tokens,
            )

        except ImportError as e:
            raise ConfigurationError(
                "langchain-openai 패키지가 설치되지 않았습니다.",
                original_error=e
            )
        except Exception as e:
            raise LLMError(
                "OpenAI LLM 초기화에 실패했습니다",
                model=model,
                provider="openai",
                original_error=e
            )

    def generate(
        self,
        prompt: str,
        context: str = "",
        **kwargs
    ) -> str:
        """
        프롬프트에 대한 응답 생성 (OpenAI)

        Args:
            prompt: 사용자 질문
            context: 검색된 문서 내용
            **kwargs: 추가 파라미터 (무시됨)

        Returns:
            str: GPT가 생성한 응답
        """
        if not prompt:
            raise LLMError("프롬프트가 비어있습니다")

        self.logger.debug(f"응답 생성 중: prompt='{prompt[:50]}...'")

        try:
            # 메시지 구성
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._build_prompt_with_context(prompt, context))
            ]

            # LLM 호출
            response = self._llm.invoke(messages)

            self.logger.debug(f"응답 생성 완료: {len(response.content)}자")
            return response.content

        except Exception as e:
            self.logger.error(f"응답 생성 실패: {e}")
            raise LLMError(
                "OpenAI 응답 생성에 실패했습니다",
                model=self.model,
                provider="openai",
                original_error=e
            )

    def generate_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        context: str = ""
    ) -> str:
        """
        대화 히스토리를 포함한 응답 생성 (OpenAI)

        Args:
            prompt: 현재 질문
            history: 이전 대화 히스토리
            context: 검색된 문서 내용

        Returns:
            str: GPT가 생성한 응답
        """
        try:
            # 메시지 구성
            messages = [SystemMessage(content=self.system_prompt)]

            # 히스토리 추가
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

            # 현재 질문 추가
            messages.append(
                HumanMessage(content=self._build_prompt_with_context(prompt, context))
            )

            # LLM 호출
            response = self._llm.invoke(messages)
            return response.content

        except Exception as e:
            raise LLMError(
                "OpenAI 대화 응답 생성에 실패했습니다",
                model=self.model,
                provider="openai",
                original_error=e
            )

    @property
    def llm(self):
        """LangChain 호환 LLM 인스턴스"""
        return self._llm


# =============================================================================
# Ollama LLM 구현 (로컬 모델)
# =============================================================================

class OllamaLLMClient(BaseLLMClient):
    """
    Ollama 로컬 LLM 클라이언트

    로컬에서 실행되는 Ollama 서버를 통해 LLM을 사용합니다.
    인터넷 연결 없이 사용 가능하며, 비용이 발생하지 않습니다.

    사전 요구사항:
        1. Ollama 설치: https://ollama.com/download
        2. 모델 다운로드: ollama pull llama3
        3. 서버 실행: ollama serve (자동 실행될 수도 있음)

    인기 모델:
        - llama3: Meta의 최신 Llama 모델 (8B/70B)
        - mistral: 뛰어난 성능의 7B 모델
        - codellama: 코드 특화 모델
        - phi3: Microsoft의 소형 모델 (빠름)

    Attributes:
        model (str): Ollama 모델명
        base_url (str): Ollama 서버 URL

    Example:
        >>> client = OllamaLLMClient(model="llama3")
        >>> response = client.generate("인공지능이란?")
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        system_prompt: Optional[str] = None
    ):
        """
        OllamaLLMClient 초기화

        Args:
            model: Ollama 모델명
            base_url: Ollama 서버 URL
            temperature: 응답 무작위성
            system_prompt: 시스템 프롬프트

        Raises:
            ConfigurationError: Ollama 연결 실패 시
        """
        super().__init__(model, temperature, system_prompt)
        self.base_url = base_url

        self.logger.info(f"Ollama LLM 클라이언트 초기화: model={model}")

        try:
            from langchain_community.chat_models import ChatOllama

            self._llm = ChatOllama(
                model=model,
                base_url=base_url,
                temperature=temperature,
            )

        except ImportError as e:
            raise ConfigurationError(
                "langchain-community 패키지가 설치되지 않았습니다.",
                original_error=e
            )
        except Exception as e:
            raise LLMError(
                f"Ollama 연결에 실패했습니다. "
                f"Ollama가 실행 중인지 확인하세요. URL: {base_url}",
                model=model,
                provider="ollama",
                original_error=e
            )

    def generate(
        self,
        prompt: str,
        context: str = "",
        **kwargs
    ) -> str:
        """프롬프트에 대한 응답 생성 (Ollama)"""
        if not prompt:
            raise LLMError("프롬프트가 비어있습니다")

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._build_prompt_with_context(prompt, context))
            ]

            response = self._llm.invoke(messages)
            return response.content

        except Exception as e:
            raise LLMError(
                "Ollama 응답 생성에 실패했습니다. "
                "Ollama 서버 상태를 확인하세요.",
                model=self.model,
                provider="ollama",
                original_error=e
            )

    def generate_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        context: str = ""
    ) -> str:
        """대화 히스토리를 포함한 응답 생성 (Ollama)"""
        try:
            messages = [SystemMessage(content=self.system_prompt)]

            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

            messages.append(
                HumanMessage(content=self._build_prompt_with_context(prompt, context))
            )

            response = self._llm.invoke(messages)
            return response.content

        except Exception as e:
            raise LLMError(
                "Ollama 대화 응답 생성에 실패했습니다",
                model=self.model,
                provider="ollama",
                original_error=e
            )

    @property
    def llm(self):
        """LangChain 호환 LLM 인스턴스"""
        return self._llm


# =============================================================================
# Anthropic LLM 구현 (Claude)
# =============================================================================

class AnthropicLLMClient(BaseLLMClient):
    """
    Anthropic Claude 모델 클라이언트

    Anthropic의 Claude 모델을 사용하여 응답을 생성합니다.

    지원 모델:
        - claude-3-opus-20240229: 최고 성능
        - claude-3-sonnet-20240229: 균형 잡힌 성능
        - claude-3-haiku-20240307: 빠르고 저렴

    특징:
        - 200K 토큰 컨텍스트 윈도우
        - 안전하고 도움이 되는 응답
        - 긴 문서 처리에 적합

    Attributes:
        model (str): Claude 모델명

    Example:
        >>> client = AnthropicLLMClient(
        ...     api_key="sk-ant-xxx",
        ...     model="claude-3-sonnet-20240229"
        ... )
        >>> response = client.generate("질문", context="문서 내용")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096
    ):
        """
        AnthropicLLMClient 초기화

        Args:
            api_key: Anthropic API 키
            model: Claude 모델명
            temperature: 응답 무작위성
            system_prompt: 시스템 프롬프트
            max_tokens: 최대 응답 토큰 수

        Raises:
            ConfigurationError: API 키가 유효하지 않은 경우
        """
        super().__init__(model, temperature, system_prompt)

        if not api_key:
            raise ConfigurationError("Anthropic API 키가 필요합니다")

        self.logger.info(f"Anthropic LLM 클라이언트 초기화: model={model}")

        try:
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(
                model=model,
                temperature=temperature,
                anthropic_api_key=api_key,
                max_tokens=max_tokens,
            )

        except ImportError as e:
            raise ConfigurationError(
                "langchain-anthropic 패키지가 설치되지 않았습니다. "
                "pip install langchain-anthropic 를 실행하세요.",
                original_error=e
            )
        except Exception as e:
            raise LLMError(
                "Anthropic LLM 초기화에 실패했습니다",
                model=model,
                provider="anthropic",
                original_error=e
            )

    def generate(
        self,
        prompt: str,
        context: str = "",
        **kwargs
    ) -> str:
        """프롬프트에 대한 응답 생성 (Claude)"""
        if not prompt:
            raise LLMError("프롬프트가 비어있습니다")

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._build_prompt_with_context(prompt, context))
            ]

            response = self._llm.invoke(messages)
            return response.content

        except Exception as e:
            raise LLMError(
                "Anthropic 응답 생성에 실패했습니다",
                model=self.model,
                provider="anthropic",
                original_error=e
            )

    def generate_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        context: str = ""
    ) -> str:
        """대화 히스토리를 포함한 응답 생성 (Claude)"""
        try:
            messages = [SystemMessage(content=self.system_prompt)]

            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

            messages.append(
                HumanMessage(content=self._build_prompt_with_context(prompt, context))
            )

            response = self._llm.invoke(messages)
            return response.content

        except Exception as e:
            raise LLMError(
                "Anthropic 대화 응답 생성에 실패했습니다",
                model=self.model,
                provider="anthropic",
                original_error=e
            )

    @property
    def llm(self):
        """LangChain 호환 LLM 인스턴스"""
        return self._llm


# =============================================================================
# 팩토리 함수
# =============================================================================

def get_llm_client(
    provider: Optional[LLMProvider] = None,
    **kwargs
) -> BaseLLMClient:
    """
    설정 기반으로 적절한 LLM 클라이언트 반환

    이 함수는 팩토리 패턴을 사용하여 설정에 따라
    적절한 LLM 클라이언트를 자동으로 선택합니다.

    Args:
        provider: LLM 제공자. None이면 설정 파일에서 읽음
        **kwargs: 클라이언트 생성에 전달할 추가 파라미터

    Returns:
        BaseLLMClient: LLM 클라이언트 인스턴스

    Raises:
        ConfigurationError: 알 수 없는 제공자이거나 설정 오류 시

    Example:
        >>> # 설정 파일 기반 자동 선택
        >>> client = get_llm_client()

        >>> # 명시적으로 제공자 지정
        >>> client = get_llm_client(LLMProvider.OLLAMA)

        >>> # 추가 파라미터 전달
        >>> client = get_llm_client(temperature=0.7)
    """
    settings = get_settings()
    provider = provider or settings.llm_provider

    logger.info(f"LLM 클라이언트 생성: provider={provider.value}")

    # 제공자별 인스턴스 생성
    if provider == LLMProvider.OPENAI:
        return OpenAILLMClient(
            api_key=settings.get_openai_api_key(),
            model=settings.openai_model,
            **kwargs
        )

    elif provider == LLMProvider.OLLAMA:
        return OllamaLLMClient(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            **kwargs
        )

    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicLLMClient(
            api_key=settings.get_anthropic_api_key(),
            model=settings.anthropic_model,
            **kwargs
        )

    else:
        raise ConfigurationError(
            f"지원하지 않는 LLM 제공자입니다: {provider}. "
            f"지원 목록: {[p.value for p in LLMProvider]}"
        )


# =============================================================================
# 편의 클래스 (하위 호환성)
# =============================================================================

class LLMClient:
    """
    LLM 클라이언트 래퍼 클래스 (하위 호환성 유지)

    직접 사용하기보다 get_llm_client() 함수 사용을 권장합니다.

    Example:
        >>> client = LLMClient()  # 기본 설정 사용
        >>> response = client.generate("질문")
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ):
        """래퍼 초기화 - 내부적으로 팩토리 함수 사용"""
        self._client = get_llm_client(provider, **kwargs)

    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        """응답 생성"""
        return self._client.generate(prompt, context, **kwargs)

    def generate_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        context: str = ""
    ) -> str:
        """대화 히스토리 포함 응답 생성"""
        return self._client.generate_with_history(prompt, history, context)

    @property
    def llm(self):
        """LangChain 호환 LLM 인스턴스"""
        return self._client.llm

    @property
    def model(self) -> str:
        """사용 중인 모델명"""
        return self._client.model
