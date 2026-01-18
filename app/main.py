"""
=============================================================================
RAG 시스템 메인 애플리케이션
=============================================================================

이 모듈은 Streamlit 기반 RAG 웹 애플리케이션의 진입점입니다.

실행 방법:
    $ streamlit run app/main.py

    또는 가상환경 활성화 후:
    $ venv\\Scripts\\activate  (Windows)
    $ source venv/bin/activate  (Linux/Mac)
    $ streamlit run app/main.py

기능:
    - 문서 업로드 (PDF, TXT, MD, 웹 URL)
    - 문서 기반 질문 응답
    - 채팅 인터페이스
    - 출처 문서 확인

필수 환경 변수:
    - OPENAI_API_KEY: OpenAI API 키 (OpenAI 사용 시)
    - 또는 다른 LLM 제공자 설정 (.env 파일 참조)
"""

import streamlit as st
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가 (임포트 문제 해결)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.config import get_settings
from app.core.rag_chain import RAGPipeline
from app.ui.sidebar import render_sidebar
from app.ui.document_upload import render_document_upload
from app.ui.chat_interface import render_chat_interface
from app.utils.logger import setup_logging, get_logger

# 로깅 설정
setup_logging()
logger = get_logger(__name__)


def configure_page() -> None:
    """
    Streamlit 페이지 기본 설정

    페이지 제목, 아이콘, 레이아웃 등을 설정합니다.
    """
    st.set_page_config(
        page_title="RAG Document Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": """
            # RAG Document Assistant

            ChromaDB와 LLM을 활용한 문서 기반 질의응답 시스템입니다.

            **지원 기능:**
            - PDF, 텍스트, 마크다운 문서 업로드
            - 웹 페이지 콘텐츠 수집
            - 자연어 질문 응답
            - 출처 문서 확인

            **기술 스택:**
            - LangChain
            - ChromaDB
            - OpenAI GPT / Ollama / Claude

            ---
            Version 0.1.0
            """
        }
    )


def initialize_session_state() -> None:
    """
    Streamlit 세션 상태 초기화

    앱 전체에서 공유되는 상태를 초기화합니다:
        - rag_pipeline: RAG 파이프라인 인스턴스
        - messages: 채팅 히스토리
        - initialized: 초기화 완료 여부
    """
    # RAG 파이프라인 초기화
    if "rag_pipeline" not in st.session_state:
        try:
            logger.info("RAG 파이프라인 초기화 시작...")
            st.session_state.rag_pipeline = RAGPipeline()
            logger.info("RAG 파이프라인 초기화 완료")
        except Exception as e:
            logger.error(f"RAG 파이프라인 초기화 실패: {e}")
            st.session_state.rag_pipeline = None
            st.session_state.init_error = str(e)

    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 초기화 완료 플래그
    if "initialized" not in st.session_state:
        st.session_state.initialized = True


def check_configuration() -> bool:
    """
    필수 설정 확인

    API 키 등 필수 설정이 되어 있는지 확인합니다.

    Returns:
        bool: 설정이 올바르면 True
    """
    try:
        settings = get_settings()

        # LLM 제공자별 필수 설정 확인
        from app.config import LLMProvider

        if settings.llm_provider == LLMProvider.OPENAI:
            if not settings.openai_api_key:
                return False

        elif settings.llm_provider == LLMProvider.ANTHROPIC:
            if not settings.anthropic_api_key:
                return False

        # Ollama는 API 키 불필요

        return True

    except Exception as e:
        logger.error(f"설정 확인 실패: {e}")
        return False


def show_configuration_error() -> None:
    """설정 오류 화면 표시"""
    st.error("⚠️ 설정 오류")
    st.markdown("""
    ### API 키가 설정되지 않았습니다

    `.env` 파일을 생성하고 필요한 API 키를 설정해주세요.

    **OpenAI 사용 시:**
    ```
    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-your-api-key-here
    ```

    **Ollama 사용 시 (로컬):**
    ```
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=llama3
    ```

    **Claude 사용 시:**
    ```
    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=sk-ant-your-api-key-here
    ```

    `.env.example` 파일을 참고하여 설정하세요.
    """)

    # .env.example 파일 내용 표시
    env_example_path = project_root / ".env.example"
    if env_example_path.exists():
        with st.expander(".env.example 파일 내용 보기"):
            st.code(env_example_path.read_text(encoding="utf-8"), language="bash")


def show_initialization_error() -> None:
    """초기화 오류 화면 표시"""
    error_msg = st.session_state.get("init_error", "알 수 없는 오류")
    st.error(f"⚠️ 초기화 실패: {error_msg}")
    st.markdown("""
    ### 문제 해결 방법

    1. **API 키 확인**: `.env` 파일에 올바른 API 키가 설정되어 있는지 확인
    2. **의존성 설치**: `pip install -r requirements.txt` 실행
    3. **Ollama 서버**: Ollama 사용 시 서버가 실행 중인지 확인

    오류가 계속되면 로그를 확인하세요.
    """)

    if st.button("다시 시도"):
        # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def render_main_content() -> None:
    """
    메인 콘텐츠 영역 렌더링

    탭 기반 인터페이스를 제공합니다:
        - 채팅: 문서 기반 질의응답
        - 문서 업로드: 파일/URL 추가
        - 통계: 시스템 상태
    """
    # 메인 타이틀
    st.title("📚 RAG Document Assistant")
    st.markdown("문서를 업로드하고 자연어로 질문하세요.")
    st.markdown("---")

    # 탭 생성
    tab_chat, tab_upload, tab_stats = st.tabs([
        "💬 채팅",
        "📤 문서 업로드",
        "📊 통계"
    ])

    # 채팅 탭
    with tab_chat:
        render_chat_interface()

    # 문서 업로드 탭
    with tab_upload:
        render_document_upload()

    # 통계 탭
    with tab_stats:
        render_statistics()


def render_statistics() -> None:
    """통계 정보 탭 렌더링"""
    st.header("시스템 통계")

    if "rag_pipeline" not in st.session_state or st.session_state.rag_pipeline is None:
        st.warning("파이프라인이 초기화되지 않았습니다")
        return

    pipeline = st.session_state.rag_pipeline

    # 새로고침 버튼
    if st.button("통계 새로고침", key="refresh_stats"):
        st.rerun()

    try:
        stats = pipeline.get_stats()

        # 벡터 스토어 통계
        st.subheader("벡터 스토어")
        vs_stats = stats.get("vector_store", {})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("저장된 청크 수", f"{vs_stats.get('count', 0):,}개")
        with col2:
            st.metric("컬렉션 이름", vs_stats.get("name", "-"))

        st.caption(f"저장 경로: {vs_stats.get('persist_directory', '-')}")

        # 청킹 설정
        st.subheader("청킹 설정")
        chunk_config = stats.get("chunking", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("청크 크기", f"{chunk_config.get('chunk_size', '-')}자")
        with col2:
            st.metric("오버랩", f"{chunk_config.get('chunk_overlap', '-')}자")
        with col3:
            st.metric("분할기 유형", chunk_config.get("splitter_type", "-"))

        # LLM 정보
        st.subheader("LLM 설정")
        llm_info = stats.get("llm", {})
        st.write(f"**모델:** {llm_info.get('model', '-')}")

        # 지원 형식
        st.subheader("지원 형식")
        formats = stats.get("supported_formats", [])
        if formats:
            st.write(", ".join(formats))

    except Exception as e:
        st.error(f"통계 로드 실패: {e}")
        logger.error(f"통계 로드 실패: {e}")


def main() -> None:
    """
    메인 애플리케이션 진입점

    앱의 전체 흐름을 관리합니다:
        1. 페이지 설정
        2. 세션 상태 초기화
        3. 설정 검증
        4. UI 렌더링
    """
    # 1. 페이지 설정
    configure_page()

    # 2. 세션 상태 초기화
    initialize_session_state()

    # 3. 설정 검증
    if not check_configuration():
        show_configuration_error()
        return

    # 4. 초기화 오류 확인
    if st.session_state.get("init_error"):
        show_initialization_error()
        return

    if st.session_state.rag_pipeline is None:
        show_initialization_error()
        return

    # 5. UI 렌더링
    render_sidebar()
    render_main_content()


# =============================================================================
# 애플리케이션 실행
# =============================================================================

if __name__ == "__main__":
    main()
