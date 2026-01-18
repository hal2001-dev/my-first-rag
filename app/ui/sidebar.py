"""
=============================================================================
사이드바 컴포넌트
=============================================================================

이 모듈은 Streamlit 앱의 사이드바 UI를 제공합니다.

사이드바 기능:
    - 설정 표시 (모델, 청크 크기 등)
    - 통계 정보 표시
    - 채팅 히스토리 초기화
    - 벡터 스토어 초기화

사용 예시:
    >>> import streamlit as st
    >>> from app.ui.sidebar import render_sidebar
    >>> render_sidebar()
"""

import streamlit as st
from typing import Optional

from app.config import get_settings, LLMProvider
from app.utils.logger import get_logger

# 모듈 레벨 로거
logger = get_logger(__name__)


def render_sidebar() -> None:
    """
    사이드바 렌더링

    앱 설정, 통계, 액션 버튼을 포함하는 사이드바를 렌더링합니다.

    사이드바 섹션:
        1. 설정 정보 표시
        2. 파이프라인 통계
        3. 액션 버튼 (초기화 등)
        4. 도움말
    """
    with st.sidebar:
        # 앱 제목
        st.title("RAG 시스템")
        st.markdown("---")

        # 1. 현재 설정 표시
        _render_settings_section()

        st.markdown("---")

        # 2. 통계 정보
        _render_stats_section()

        st.markdown("---")

        # 3. 액션 버튼
        _render_actions_section()

        st.markdown("---")

        # 4. 도움말
        _render_help_section()


def _render_settings_section() -> None:
    """설정 정보 섹션 렌더링"""
    st.subheader("현재 설정")

    settings = get_settings()

    # LLM 정보
    st.markdown("**LLM 설정**")
    st.caption(f"제공자: {settings.llm_provider.value}")
    if settings.llm_provider == LLMProvider.OPENAI:
        st.caption(f"모델: {settings.openai_model}")
    elif settings.llm_provider == LLMProvider.OLLAMA:
        st.caption(f"모델: {settings.ollama_model}")
    elif settings.llm_provider == LLMProvider.ANTHROPIC:
        st.caption(f"모델: {settings.anthropic_model}")

    # 청킹 설정
    st.markdown("**청킹 설정**")
    st.caption(f"청크 크기: {settings.chunk_size}자")
    st.caption(f"오버랩: {settings.chunk_overlap}자")

    # 검색 설정
    st.markdown("**검색 설정**")
    st.caption(f"Top-K: {settings.retrieval_top_k}개")


def _render_stats_section() -> None:
    """통계 정보 섹션 렌더링"""
    st.subheader("통계")

    # 파이프라인이 초기화되어 있는지 확인
    if "rag_pipeline" not in st.session_state:
        st.info("파이프라인이 초기화되지 않았습니다")
        return

    pipeline = st.session_state.rag_pipeline

    try:
        stats = pipeline.get_stats()

        # 문서 수
        doc_count = stats.get("vector_store", {}).get("count", 0)
        st.metric("저장된 문서 청크", f"{doc_count:,}개")

        # 지원 형식
        formats = stats.get("supported_formats", [])
        if formats:
            st.caption(f"지원 형식: {', '.join(formats[:5])}")

    except Exception as e:
        st.error(f"통계 로드 실패: {e}")
        logger.error(f"통계 로드 실패: {e}")


def _render_actions_section() -> None:
    """액션 버튼 섹션 렌더링"""
    st.subheader("액션")

    # 채팅 히스토리 초기화
    if st.button(
        "채팅 기록 삭제",
        key="clear_chat",
        help="대화 히스토리를 삭제합니다",
        use_container_width=True
    ):
        if "messages" in st.session_state:
            st.session_state.messages = []
            st.success("채팅 기록이 삭제되었습니다")
            st.rerun()

    # 벡터 스토어 초기화 (확인 절차 포함)
    st.markdown("---")
    st.markdown("**위험 영역**")

    # 확인 체크박스
    confirm_reset = st.checkbox(
        "벡터 스토어 초기화 확인",
        key="confirm_reset_checkbox",
        help="체크 후 초기화 버튼을 누르면 모든 문서가 삭제됩니다"
    )

    if st.button(
        "벡터 스토어 초기화",
        key="reset_vector_store",
        disabled=not confirm_reset,
        type="secondary",
        use_container_width=True
    ):
        if confirm_reset and "rag_pipeline" in st.session_state:
            try:
                st.session_state.rag_pipeline.clear()
                st.session_state.messages = []
                st.success("벡터 스토어가 초기화되었습니다")
                st.rerun()
            except Exception as e:
                st.error(f"초기화 실패: {e}")
                logger.error(f"벡터 스토어 초기화 실패: {e}")


def _render_help_section() -> None:
    """도움말 섹션 렌더링"""
    st.subheader("도움말")

    with st.expander("사용 방법", expanded=False):
        st.markdown("""
        **1. 문서 업로드**
        - "문서 업로드" 탭에서 파일 업로드 또는 URL 입력
        - 지원 형식: PDF, TXT, MD, 웹 페이지

        **2. 질문하기**
        - "채팅" 탭에서 질문 입력
        - 업로드한 문서 기반으로 답변 생성

        **3. 출처 확인**
        - 답변 아래 "출처 보기"에서 참조 문서 확인
        """)

    with st.expander("문제 해결", expanded=False):
        st.markdown("""
        **답변이 "정보를 찾을 수 없습니다"인 경우:**
        - 관련 문서가 업로드되었는지 확인
        - 질문을 더 구체적으로 작성

        **문서 업로드 실패:**
        - 파일 형식 확인 (PDF, TXT, MD)
        - 파일 크기 확인 (최대 50MB)
        - URL이 접근 가능한지 확인

        **느린 응답:**
        - 대용량 문서는 처리 시간이 길 수 있음
        - 인터넷 연결 상태 확인
        """)

    # 버전 정보
    st.caption("RAG System v0.1.0")
