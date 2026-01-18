"""
=============================================================================
RAG (Retrieval-Augmented Generation) 파이프라인
=============================================================================

이 모듈은 RAG 시스템의 핵심 파이프라인을 구현합니다.
문서 수집, 저장, 검색, 응답 생성의 전체 흐름을 관리합니다.

RAG 아키텍처:
    1. 수집 (Ingestion):
       문서 로드 → 청킹 → 임베딩 → 벡터 스토어 저장

    2. 검색 (Retrieval):
       질문 → 임베딩 → 유사 문서 검색 → 관련 문서 반환

    3. 생성 (Generation):
       질문 + 관련 문서 → LLM → 답변 생성

사용 예시:
    >>> from app.core.rag_chain import RAGPipeline
    >>> pipeline = RAGPipeline()

    >>> # 문서 수집
    >>> stats = pipeline.ingest_documents(["document.pdf", "https://example.com"])
    >>> print(f"총 {stats['chunks_created']}개 청크 생성됨")

    >>> # 질문 응답
    >>> response = pipeline.query("문서 내용에 대한 질문")
    >>> print(response["answer"])
"""

from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.config import get_settings
from app.core.vector_store import VectorStore
from app.core.llm import get_llm_client, BaseLLMClient
from app.core.embeddings import get_embedding_generator
from app.loaders.loader_factory import DocumentLoaderFactory
from app.chunking.text_splitter import TextChunker
from app.utils.logger import get_logger, LoggerMixin
from app.utils.exceptions import RAGException

# 모듈 레벨 로거
logger = get_logger(__name__)


# =============================================================================
# RAG 프롬프트 템플릿
# =============================================================================

RAG_PROMPT_TEMPLATE = """다음 문맥을 참고하여 질문에 답변해주세요.

문맥:
{context}

질문: {question}

답변 지침:
1. 반드시 제공된 문맥만을 기반으로 답변하세요
2. 문맥에 해당 정보가 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요
3. 가능하면 출처를 언급하세요
4. 간결하되 충분한 정보를 포함하세요

답변:"""


# =============================================================================
# RAG 파이프라인 클래스
# =============================================================================

class RAGPipeline(LoggerMixin):
    """
    RAG 파이프라인 클래스

    문서 수집부터 질문 응답까지 전체 RAG 흐름을 관리합니다.
    LangChain Expression Language (LCEL)을 사용하여
    효율적인 체인을 구성합니다.

    Attributes:
        vector_store (VectorStore): 벡터 데이터베이스
        llm_client (BaseLLMClient): LLM 클라이언트
        loader_factory (DocumentLoaderFactory): 문서 로더 팩토리
        text_chunker (TextChunker): 텍스트 청커

    Methods:
        ingest_documents: 문서 수집 및 저장
        query: 질문에 대한 답변 생성
        get_stats: 파이프라인 통계 정보

    Example:
        >>> pipeline = RAGPipeline()

        >>> # 문서 수집
        >>> pipeline.ingest_documents([
        ...     "report.pdf",
        ...     "https://example.com/article"
        ... ])

        >>> # 질문
        >>> result = pipeline.query("주요 내용을 요약해주세요")
        >>> print(result["answer"])
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[BaseLLMClient] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        RAGPipeline 초기화

        Args:
            vector_store: 벡터 스토어 인스턴스. None이면 자동 생성
            llm_client: LLM 클라이언트. None이면 설정 기반 자동 생성
            chunk_size: 청크 크기. None이면 설정에서 읽음
            chunk_overlap: 청크 오버랩. None이면 설정에서 읽음

        Example:
            >>> # 기본 설정 사용
            >>> pipeline = RAGPipeline()

            >>> # 커스텀 설정
            >>> pipeline = RAGPipeline(
            ...     chunk_size=1000,
            ...     chunk_overlap=100
            ... )
        """
        self.logger.info("RAGPipeline 초기화 시작...")

        # 설정 로드
        settings = get_settings()
        settings.ensure_directories()

        # 컴포넌트 초기화
        self.vector_store = vector_store or VectorStore()
        self.llm_client = llm_client or get_llm_client()
        self.loader_factory = DocumentLoaderFactory()
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # RAG 체인 구성
        self._chain = self._build_chain()

        self.logger.info(
            f"RAGPipeline 초기화 완료: "
            f"LLM={self.llm_client.model}, "
            f"chunk_size={self.text_chunker.chunk_size}"
        )

    def _build_chain(self):
        """
        LCEL을 사용한 RAG 체인 구성

        체인 구조:
            1. 질문 → Retriever → 관련 문서
            2. 관련 문서 → 포맷팅 → 컨텍스트
            3. 컨텍스트 + 질문 → 프롬프트 → LLM → 응답

        Returns:
            Runnable: 실행 가능한 RAG 체인
        """
        # 프롬프트 템플릿
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # Retriever
        retriever = self.vector_store.as_retriever()

        # 체인 구성 (LCEL)
        chain = (
            {
                "context": retriever | self._format_documents,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm_client.llm
            | StrOutputParser()
        )

        return chain

    @staticmethod
    def _format_documents(documents: List[Document]) -> str:
        """
        검색된 문서들을 컨텍스트 문자열로 포맷팅

        각 문서에 출처 정보를 포함하여 가독성 있게 포맷팅합니다.

        Args:
            documents: 검색된 Document 목록

        Returns:
            str: 포맷팅된 컨텍스트 문자열
        """
        if not documents:
            return "관련 문서를 찾을 수 없습니다."

        formatted_parts = []
        for i, doc in enumerate(documents, 1):
            # 출처 정보 추출
            source = doc.metadata.get("source", "알 수 없음")
            page = doc.metadata.get("page", "")

            # 출처 레이블 생성
            source_label = f"[출처 {i}: {source}"
            if page:
                source_label += f", 페이지 {page}"
            source_label += "]"

            # 포맷팅
            formatted_parts.append(f"{source_label}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_parts)

    # =========================================================================
    # 문서 수집 (Ingestion)
    # =========================================================================

    def ingest_documents(
        self,
        sources: List[str],
        fail_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        여러 소스에서 문서를 수집하여 벡터 스토어에 저장

        지원 소스:
            - PDF 파일 (.pdf)
            - 텍스트 파일 (.txt, .md)
            - 웹 URL (http://, https://)

        처리 과정:
            1. 소스별 적절한 로더로 문서 로드
            2. 문서를 청크로 분할
            3. 청크를 벡터 스토어에 저장

        Args:
            sources: 문서 소스 목록 (파일 경로 또는 URL)
            fail_on_error: True면 오류 발생 시 즉시 중단

        Returns:
            dict: 수집 결과 통계
                - sources_processed: 처리된 소스 수
                - documents_loaded: 로드된 문서 수
                - chunks_created: 생성된 청크 수
                - chunks_stored: 저장된 청크 수
                - errors: 오류 목록

        Example:
            >>> stats = pipeline.ingest_documents([
            ...     "report.pdf",
            ...     "notes.txt",
            ...     "https://example.com/article"
            ... ])
            >>> print(f"처리: {stats['sources_processed']}개 소스")
            >>> print(f"청크: {stats['chunks_created']}개 생성됨")
        """
        self.logger.info(f"문서 수집 시작: {len(sources)}개 소스")

        stats = {
            "sources_processed": 0,
            "documents_loaded": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "errors": [],
        }

        all_documents = []

        # 1. 문서 로드
        for source in sources:
            try:
                documents = self.loader_factory.load(source)
                all_documents.extend(documents)
                stats["sources_processed"] += 1
                stats["documents_loaded"] += len(documents)

                self.logger.info(
                    f"로드 완료: {source} ({len(documents)}개 문서)"
                )

            except Exception as e:
                error_info = {
                    "source": source,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                stats["errors"].append(error_info)
                self.logger.error(f"로드 실패: {source}, 오류: {e}")

                if fail_on_error:
                    raise RAGException(
                        f"문서 로드 실패: {source}",
                        details=error_info,
                        original_error=e
                    )

        if not all_documents:
            self.logger.warning("로드된 문서가 없습니다")
            return stats

        # 2. 문서 청킹
        self.logger.info(f"문서 청킹 시작: {len(all_documents)}개 문서")

        chunks = self.text_chunker.split_documents(all_documents)
        stats["chunks_created"] = len(chunks)

        self.logger.info(f"청킹 완료: {len(chunks)}개 청크 생성")

        # 3. 벡터 스토어에 저장
        if chunks:
            self.logger.info(f"벡터 스토어 저장 시작: {len(chunks)}개 청크")

            try:
                ids = self.vector_store.add_documents(chunks)
                stats["chunks_stored"] = len(ids)

                self.logger.info(f"저장 완료: {len(ids)}개 청크")

            except Exception as e:
                stats["errors"].append({
                    "source": "vector_store",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                self.logger.error(f"벡터 스토어 저장 실패: {e}")

                if fail_on_error:
                    raise

        self.logger.info(
            f"문서 수집 완료: "
            f"소스 {stats['sources_processed']}/{len(sources)}, "
            f"청크 {stats['chunks_stored']}/{stats['chunks_created']}"
        )

        return stats

    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        순수 텍스트를 직접 수집

        파일 없이 텍스트를 직접 벡터 스토어에 추가합니다.
        사용자 입력이나 API 응답 등을 저장할 때 유용합니다.

        Args:
            text: 저장할 텍스트
            metadata: 추가 메타데이터 (출처 정보 등)

        Returns:
            dict: 수집 결과 통계

        Example:
            >>> stats = pipeline.ingest_text(
            ...     "이것은 직접 입력한 텍스트입니다...",
            ...     metadata={"source": "user_input", "topic": "example"}
            ... )
        """
        self.logger.info(f"텍스트 직접 수집: {len(text)}자")

        # Document 생성
        doc_metadata = metadata or {}
        doc_metadata["source"] = doc_metadata.get("source", "direct_input")

        # 청킹
        chunks = self.text_chunker.split_text_to_documents(
            text, metadata=doc_metadata
        )

        stats = {
            "sources_processed": 1,
            "documents_loaded": 1,
            "chunks_created": len(chunks),
            "chunks_stored": 0,
            "errors": [],
        }

        # 저장
        if chunks:
            ids = self.vector_store.add_documents(chunks)
            stats["chunks_stored"] = len(ids)

        return stats

    # =========================================================================
    # 질문 응답 (Query)
    # =========================================================================

    def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성

        RAG 체인을 실행하여 질문에 대한 답변을 생성합니다.
        관련 문서를 검색하고 LLM을 통해 답변을 생성합니다.

        Args:
            question: 사용자 질문
            return_sources: True면 출처 문서 정보 포함

        Returns:
            dict: 응답 결과
                - question: 원본 질문
                - answer: 생성된 답변
                - sources: 참조된 문서 정보 (return_sources=True 시)

        Raises:
            RAGException: 답변 생성 실패 시

        Example:
            >>> result = pipeline.query("주요 내용을 요약해주세요")
            >>> print(result["answer"])

            >>> # 출처 확인
            >>> for source in result["sources"]:
            ...     print(f"- {source['metadata']['source']}")
        """
        if not question or not question.strip():
            return {
                "question": question,
                "answer": "질문을 입력해주세요.",
                "sources": [],
            }

        self.logger.info(f"질문 처리 시작: '{question[:50]}...'")

        try:
            # 관련 문서 검색 (출처용)
            if return_sources:
                retrieved_docs = self.vector_store.similarity_search(
                    question,
                    k=get_settings().retrieval_top_k
                )
            else:
                retrieved_docs = []

            # RAG 체인 실행
            answer = self._chain.invoke(question)

            # 출처 정보 구성
            sources = []
            if return_sources:
                for doc in retrieved_docs:
                    sources.append({
                        "content": doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content,
                        "metadata": doc.metadata,
                    })

            self.logger.info(f"답변 생성 완료: {len(answer)}자")

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
            }

        except Exception as e:
            self.logger.error(f"답변 생성 실패: {e}")
            raise RAGException(
                "답변 생성에 실패했습니다",
                details={"question": question[:100]},
                original_error=e
            )

    def query_with_history(
        self,
        question: str,
        history: List[Dict[str, str]],
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        대화 히스토리를 고려한 질문 응답

        이전 대화 내용을 컨텍스트로 활용하여 답변을 생성합니다.

        Args:
            question: 현재 질문
            history: 이전 대화 히스토리
                    [{"role": "user", "content": "..."}, ...]
            return_sources: 출처 포함 여부

        Returns:
            dict: 응답 결과

        Example:
            >>> history = [
            ...     {"role": "user", "content": "RAG가 뭐야?"},
            ...     {"role": "assistant", "content": "RAG는..."}
            ... ]
            >>> result = pipeline.query_with_history(
            ...     "더 자세히 설명해줘",
            ...     history
            ... )
        """
        self.logger.info(f"히스토리 포함 질문: {len(history)}개 이전 대화")

        # 관련 문서 검색
        retrieved_docs = self.vector_store.similarity_search(
            question,
            k=get_settings().retrieval_top_k
        )

        # 컨텍스트 구성
        context = self._format_documents(retrieved_docs)

        # 히스토리 포함 답변 생성
        answer = self.llm_client.generate_with_history(
            prompt=question,
            history=history,
            context=context
        )

        # 출처 구성
        sources = []
        if return_sources:
            for doc in retrieved_docs:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    def search(
        self,
        query: str,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        관련 문서만 검색 (답변 생성 없이)

        LLM 호출 없이 유사 문서만 검색합니다.
        검색 결과 확인이나 디버깅에 유용합니다.

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            List[dict]: 검색된 문서 정보 목록

        Example:
            >>> results = pipeline.search("인공지능 기술")
            >>> for r in results:
            ...     print(f"점수: {r['score']:.4f}")
            ...     print(f"내용: {r['content'][:100]}...")
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]

    # =========================================================================
    # 유틸리티 메서드
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        파이프라인 통계 정보 반환

        Returns:
            dict: 파이프라인 상태 정보
                - vector_store: 벡터 스토어 통계
                - chunking: 청킹 설정
                - llm: LLM 정보

        Example:
            >>> stats = pipeline.get_stats()
            >>> print(f"저장된 청크: {stats['vector_store']['count']}")
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "chunking": self.text_chunker.config,
            "llm": {
                "model": self.llm_client.model,
            },
            "supported_formats": self.loader_factory.get_supported_formats(),
        }

    def clear(self) -> None:
        """
        벡터 스토어의 모든 데이터 삭제

        주의: 이 작업은 되돌릴 수 없습니다!

        Example:
            >>> pipeline.clear()
            >>> # 모든 문서가 삭제됨
        """
        self.logger.warning("벡터 스토어 초기화 중...")
        self.vector_store.delete_collection()
        self.logger.warning("벡터 스토어 초기화 완료")

    def is_ready(self) -> bool:
        """
        파이프라인이 질문에 응답할 준비가 되었는지 확인

        문서가 수집되어 있어야 질문에 답변할 수 있습니다.

        Returns:
            bool: 준비 완료 여부 (문서가 있으면 True)

        Example:
            >>> if pipeline.is_ready():
            ...     result = pipeline.query("질문")
            ... else:
            ...     print("먼저 문서를 수집하세요")
        """
        stats = self.vector_store.get_collection_stats()
        return stats.get("count", 0) > 0

    def get_document_count(self) -> int:
        """
        저장된 문서(청크) 수 반환

        Returns:
            int: 저장된 청크 수
        """
        stats = self.vector_store.get_collection_stats()
        return stats.get("count", 0)
