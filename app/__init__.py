"""
=============================================================================
RAG (Retrieval-Augmented Generation) 애플리케이션 패키지
=============================================================================

이 패키지는 ChromaDB와 LLM을 사용한 문서 기반 질의응답 시스템을 제공합니다.

주요 모듈:
    - core: 핵심 RAG 기능 (임베딩, 벡터 스토어, LLM, RAG 체인)
    - loaders: 다양한 문서 형식 로더 (PDF, 텍스트, 웹)
    - chunking: 텍스트 청킹 전략
    - ui: Streamlit 웹 인터페이스 컴포넌트
    - utils: 유틸리티 함수 및 헬퍼

사용 예시:
    >>> from app.core.rag_chain import RAGPipeline
    >>> pipeline = RAGPipeline()
    >>> pipeline.ingest_documents(["document.pdf"])
    >>> response = pipeline.query("문서 내용에 대해 질문하세요")
"""

__version__ = "0.1.0"
__author__ = "RAG System Developer"
