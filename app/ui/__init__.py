"""
=============================================================================
Streamlit 웹 UI 모듈
=============================================================================

이 패키지는 RAG 시스템의 웹 인터페이스 컴포넌트들을 포함합니다.

컴포넌트:
    - sidebar: 설정 및 옵션 사이드바
    - document_upload: 문서 업로드 인터페이스
    - chat_interface: 채팅 기반 질의응답 UI

Streamlit 세션 상태 관리:
    - st.session_state.rag_pipeline: RAG 파이프라인 인스턴스
    - st.session_state.messages: 채팅 히스토리
    - st.session_state.documents_ingested: 문서 로드 여부
"""

from app.ui.sidebar import render_sidebar
from app.ui.document_upload import render_document_upload
from app.ui.chat_interface import render_chat_interface

__all__ = [
    "render_sidebar",
    "render_document_upload",
    "render_chat_interface",
]
