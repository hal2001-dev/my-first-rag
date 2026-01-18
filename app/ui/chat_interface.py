"""
=============================================================================
채팅 인터페이스 컴포넌트
=============================================================================

이 모듈은 RAG 시스템의 채팅 UI를 제공합니다.

기능:
    - 채팅 메시지 표시
    - 질문 입력 및 응답 생성
    - 출처 문서 표시
    - 대화 히스토리 관리

사용 예시:
    >>> import streamlit as st
    >>> from app.ui.chat_interface import render_chat_interface
    >>> render_chat_interface()
"""

import streamlit as st
from typing import List, Dict, Any

from app.utils.logger import get_logger

# 모듈 레벨 로거
logger = get_logger(__name__)


def render_chat_interface() -> None:
    """
    채팅 인터페이스 렌더링

    RAG 시스템과 대화할 수 있는 채팅 UI를 제공합니다.
    이전 대화 내용을 표시하고 새 질문을 입력받습니다.
    """
    st.header("문서와 대화하기")

    # 파이프라인 확인
    if "rag_pipeline" not in st.session_state:
        st.error("RAG 파이프라인이 초기화되지 않았습니다")
        return

    pipeline = st.session_state.rag_pipeline

    # 문서 로드 여부 확인
    if not pipeline.is_ready():
        st.warning(
            "아직 문서가 업로드되지 않았습니다. "
            "'문서 업로드' 탭에서 먼저 문서를 추가해주세요."
        )
        _show_demo_message()
        return

    # 문서 수 표시
    doc_count = pipeline.get_document_count()
    st.info(f"현재 {doc_count:,}개의 문서 청크가 로드되어 있습니다.")

    # 채팅 히스토리 표시
    _display_chat_history()

    # 입력 처리
    _handle_user_input()


def _show_demo_message() -> None:
    """문서 없을 때 안내 메시지"""
    st.markdown("---")
    st.markdown("### 시작 방법")
    st.markdown("""
    1. **문서 업로드** 탭으로 이동
    2. PDF, TXT, MD 파일 업로드 또는 웹 URL 입력
    3. '처리 시작' 버튼 클릭
    4. 처리 완료 후 이 탭에서 질문하기

    **지원 형식:**
    - PDF 문서
    - 텍스트 파일 (.txt, .md)
    - 웹 페이지 (URL)
    """)


def _display_chat_history() -> None:
    """채팅 히스토리 표시"""
    # 세션 상태에서 메시지 가져오기
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 각 메시지 렌더링
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

            # 어시스턴트 메시지인 경우 출처 표시
            if role == "assistant" and message.get("sources"):
                _display_sources(message["sources"])


def _display_sources(sources: List[Dict[str, Any]]) -> None:
    """
    출처 문서 표시

    Args:
        sources: 출처 정보 목록
    """
    if not sources:
        return

    with st.expander("출처 보기", expanded=False):
        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            source_name = metadata.get("source", "알 수 없음")
            page = metadata.get("page", "")

            # 출처 헤더
            header = f"**[출처 {i}]** {source_name}"
            if page:
                header += f" (페이지 {page})"
            st.markdown(header)

            # 내용 미리보기
            content = source.get("content", "")
            if content:
                st.caption(content[:300] + "..." if len(content) > 300 else content)

            if i < len(sources):
                st.markdown("---")


def _handle_user_input() -> None:
    """사용자 입력 처리"""
    # 채팅 입력
    if prompt := st.chat_input(
        "문서에 대해 질문하세요...",
        key="chat_input"
    ):
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)

        # 응답 생성
        _generate_response(prompt)


def _generate_response(question: str) -> None:
    """
    RAG 응답 생성

    Args:
        question: 사용자 질문
    """
    pipeline = st.session_state.rag_pipeline

    # 어시스턴트 메시지 영역
    with st.chat_message("assistant"):
        # 로딩 표시
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                # 대화 히스토리 준비 (최근 5개)
                history = _prepare_history(st.session_state.messages[:-1])

                # RAG 쿼리 실행
                if history:
                    result = pipeline.query_with_history(
                        question=question,
                        history=history,
                        return_sources=True
                    )
                else:
                    result = pipeline.query(
                        question=question,
                        return_sources=True
                    )

                answer = result["answer"]
                sources = result.get("sources", [])

                # 응답 표시
                st.markdown(answer)

                # 출처 표시
                if sources:
                    _display_sources(sources)

                # 세션에 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                logger.error(f"응답 생성 실패: {e}")

                # 에러도 히스토리에 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })


def _prepare_history(
    messages: List[Dict[str, Any]],
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """
    LLM용 대화 히스토리 준비

    Args:
        messages: 전체 메시지 목록
        max_messages: 최대 포함할 메시지 수

    Returns:
        List[Dict[str, str]]: LLM에 전달할 히스토리
    """
    # 최근 메시지만 사용
    recent_messages = messages[-max_messages:]

    history = []
    for msg in recent_messages:
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    return history


def clear_chat_history() -> None:
    """
    채팅 히스토리 초기화

    외부에서 호출 가능한 헬퍼 함수입니다.
    """
    if "messages" in st.session_state:
        st.session_state.messages = []
        logger.info("채팅 히스토리 초기화됨")
