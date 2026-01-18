"""
=============================================================================
문서 업로드 컴포넌트
=============================================================================

이 모듈은 문서 업로드 UI를 제공합니다.

지원 기능:
    - 파일 업로드 (PDF, TXT, MD)
    - URL 입력 (웹 페이지)
    - 처리 상태 표시
    - 결과 통계 표시

사용 예시:
    >>> import streamlit as st
    >>> from app.ui.document_upload import render_document_upload
    >>> render_document_upload()
"""

import streamlit as st
from pathlib import Path
from typing import List, Optional

from app.config import get_settings
from app.utils.logger import get_logger

# 모듈 레벨 로거
logger = get_logger(__name__)


def render_document_upload() -> None:
    """
    문서 업로드 인터페이스 렌더링

    파일 업로드와 URL 입력을 위한 UI를 제공합니다.
    업로드된 문서는 RAG 파이프라인을 통해 처리됩니다.
    """
    st.header("문서 업로드")
    st.markdown("PDF, 텍스트 파일 또는 웹 URL을 추가하여 RAG 시스템에 문서를 등록하세요.")

    # 파이프라인 확인
    if "rag_pipeline" not in st.session_state:
        st.error("RAG 파이프라인이 초기화되지 않았습니다. 앱을 새로고침하세요.")
        return

    # 탭으로 구분
    tab1, tab2 = st.tabs(["파일 업로드", "URL 입력"])

    with tab1:
        _render_file_upload()

    with tab2:
        _render_url_input()


def _render_file_upload() -> None:
    """파일 업로드 섹션 렌더링"""
    st.subheader("파일 업로드")

    settings = get_settings()

    # 파일 업로더
    uploaded_files = st.file_uploader(
        "파일을 선택하세요",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        help=f"지원 형식: PDF, TXT, MD (최대 {settings.max_upload_size_mb}MB)",
        key="file_uploader"
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)}개 파일이 선택되었습니다")

        # 파일 목록 표시
        with st.expander("선택된 파일 목록", expanded=True):
            for file in uploaded_files:
                file_size_mb = file.size / (1024 * 1024)
                st.caption(f"- {file.name} ({file_size_mb:.2f} MB)")

        # 처리 버튼
        if st.button(
            "문서 처리 시작",
            key="process_files",
            type="primary",
            use_container_width=True
        ):
            _process_uploaded_files(uploaded_files)


def _render_url_input() -> None:
    """URL 입력 섹션 렌더링"""
    st.subheader("웹 페이지 추가")

    # URL 입력
    url_input = st.text_area(
        "URL을 입력하세요 (줄바꿈으로 여러 개 입력)",
        placeholder="https://example.com/page1\nhttps://example.com/page2",
        height=150,
        key="url_input"
    )

    if url_input:
        urls = [url.strip() for url in url_input.strip().split("\n") if url.strip()]

        if urls:
            st.info(f"{len(urls)}개 URL이 입력되었습니다")

            # URL 목록 표시
            with st.expander("입력된 URL 목록", expanded=True):
                for url in urls:
                    st.caption(f"- {url}")

            # 처리 버튼
            if st.button(
                "웹 페이지 처리 시작",
                key="process_urls",
                type="primary",
                use_container_width=True
            ):
                _process_urls(urls)


def _process_uploaded_files(uploaded_files) -> None:
    """
    업로드된 파일 처리

    Args:
        uploaded_files: Streamlit 업로드 파일 객체 목록
    """
    settings = get_settings()
    pipeline = st.session_state.rag_pipeline

    # 임시 파일 저장 경로
    upload_dir = settings.upload_directory
    upload_dir.mkdir(parents=True, exist_ok=True)

    sources = []

    # 프로그레스 바
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. 파일 저장
        status_text.text("파일 저장 중...")
        for i, uploaded_file in enumerate(uploaded_files):
            # 파일 크기 확인
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > settings.max_upload_size_mb:
                st.warning(
                    f"'{uploaded_file.name}' 파일이 너무 큽니다 "
                    f"({file_size_mb:.1f}MB > {settings.max_upload_size_mb}MB). 건너뜁니다."
                )
                continue

            # 파일 저장
            file_path = upload_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            sources.append(str(file_path))
            progress_bar.progress((i + 1) / len(uploaded_files) * 0.3)

        if not sources:
            st.error("처리할 파일이 없습니다")
            return

        # 2. RAG 파이프라인으로 처리
        status_text.text("문서 처리 중... (시간이 걸릴 수 있습니다)")
        progress_bar.progress(0.5)

        stats = pipeline.ingest_documents(sources)

        progress_bar.progress(1.0)
        status_text.empty()

        # 결과 표시
        _display_processing_result(stats)

    except Exception as e:
        st.error(f"처리 중 오류 발생: {e}")
        logger.error(f"파일 처리 실패: {e}")

    finally:
        progress_bar.empty()


def _process_urls(urls: List[str]) -> None:
    """
    URL 처리

    Args:
        urls: URL 문자열 목록
    """
    pipeline = st.session_state.rag_pipeline

    # 프로그레스 바
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("웹 페이지 로딩 중... (시간이 걸릴 수 있습니다)")
        progress_bar.progress(0.3)

        stats = pipeline.ingest_documents(urls)

        progress_bar.progress(1.0)
        status_text.empty()

        # 결과 표시
        _display_processing_result(stats)

    except Exception as e:
        st.error(f"처리 중 오류 발생: {e}")
        logger.error(f"URL 처리 실패: {e}")

    finally:
        progress_bar.empty()


def _display_processing_result(stats: dict) -> None:
    """
    처리 결과 표시

    Args:
        stats: 처리 통계 딕셔너리
    """
    # 성공 메시지
    if stats["chunks_stored"] > 0:
        st.success("문서 처리가 완료되었습니다!")

        # 통계 메트릭
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("처리된 소스", f"{stats['sources_processed']}개")
        with col2:
            st.metric("로드된 문서", f"{stats['documents_loaded']}개")
        with col3:
            st.metric("생성된 청크", f"{stats['chunks_stored']}개")

    else:
        st.warning("처리된 문서가 없습니다")

    # 오류 표시
    if stats.get("errors"):
        with st.expander("오류 상세", expanded=False):
            for error in stats["errors"]:
                st.error(f"- {error['source']}: {error['error']}")
