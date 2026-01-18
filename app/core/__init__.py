"""
=============================================================================
핵심 RAG 기능 모듈
=============================================================================

이 패키지는 RAG 시스템의 핵심 컴포넌트들을 포함합니다.

모듈:
    - embeddings: 텍스트 임베딩 생성 (벡터 변환)
    - vector_store: ChromaDB 벡터 데이터베이스 통합
    - llm: LLM(Large Language Model) 클라이언트
    - rag_chain: RAG 파이프라인 오케스트레이션

아키텍처 개요:
    1. 문서 → 청킹 → 임베딩 → 벡터 스토어 저장
    2. 질의 → 임베딩 → 유사 문서 검색 → LLM 응답 생성
"""

from app.core.embeddings import EmbeddingGenerator, get_embedding_generator
from app.core.vector_store import VectorStore
from app.core.llm import LLMClient, get_llm_client
from app.core.rag_chain import RAGPipeline

__all__ = [
    "EmbeddingGenerator",
    "get_embedding_generator",
    "VectorStore",
    "LLMClient",
    "get_llm_client",
    "RAGPipeline",
]
