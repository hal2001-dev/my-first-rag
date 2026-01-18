"""
=============================================================================
벡터 스토어 모듈 (ChromaDB 통합)
=============================================================================

이 모듈은 ChromaDB 벡터 데이터베이스와의 통합을 담당합니다.
문서 임베딩을 저장하고 유사도 기반 검색을 수행합니다.

ChromaDB 특징:
    - 로컬 영구 저장 지원 (SQLite 기반)
    - 메타데이터 필터링 가능
    - LangChain과 완벽 통합
    - 무료 오픈소스

주요 개념:
    - Collection: 문서 그룹 (RDBMS의 테이블과 유사)
    - Document: 텍스트와 메타데이터를 포함하는 문서
    - Embedding: 문서의 벡터 표현

사용 예시:
    >>> from app.core.vector_store import VectorStore
    >>> store = VectorStore()
    >>> store.add_documents(documents)
    >>> results = store.similarity_search("검색어", k=3)

아키텍처:
    1. 문서 추가: 문서 → 임베딩 생성 → ChromaDB 저장
    2. 검색: 쿼리 → 임베딩 → 유사 문서 검색 → 결과 반환
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.core.embeddings import get_embedding_generator, BaseEmbeddingGenerator
from app.utils.logger import get_logger, LoggerMixin
from app.utils.exceptions import VectorStoreError, ConfigurationError

# 모듈 레벨 로거
logger = get_logger(__name__)


class VectorStore(LoggerMixin):
    """
    ChromaDB 벡터 스토어 래퍼 클래스

    이 클래스는 ChromaDB와의 모든 상호작용을 캡슐화합니다.
    문서 추가, 검색, 삭제 등의 기능을 제공합니다.

    Attributes:
        collection_name (str): ChromaDB 컬렉션 이름
        persist_directory (Path): 데이터 영구 저장 경로
        _embedding_generator: 임베딩 생성기 인스턴스
        _vector_store: LangChain Chroma 인스턴스

    Example:
        >>> # 기본 설정으로 초기화
        >>> store = VectorStore()

        >>> # 커스텀 설정으로 초기화
        >>> store = VectorStore(
        ...     collection_name="my_docs",
        ...     persist_directory=Path("./my_data")
        ... )

        >>> # 문서 추가
        >>> docs = [Document(page_content="내용", metadata={"source": "test"})]
        >>> store.add_documents(docs)

        >>> # 유사 문서 검색
        >>> results = store.similarity_search("검색어", k=5)
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[Path] = None,
        embedding_generator: Optional[BaseEmbeddingGenerator] = None,
    ):
        """
        VectorStore 초기화

        Args:
            collection_name: ChromaDB 컬렉션 이름.
                            None이면 설정 파일에서 읽음
            persist_directory: 데이터 저장 경로.
                              None이면 설정 파일에서 읽음
            embedding_generator: 임베딩 생성기.
                                None이면 설정 기반으로 자동 생성

        Raises:
            VectorStoreError: ChromaDB 초기화 실패 시
            ConfigurationError: 의존성 누락 시
        """
        # 설정 로드
        settings = get_settings()

        # 파라미터 또는 설정에서 값 가져오기
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory

        # 디렉토리 생성 (없으면)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"VectorStore 초기화: collection={self.collection_name}, "
            f"path={self.persist_directory}"
        )

        # 임베딩 생성기 설정
        self._embedding_generator = embedding_generator or get_embedding_generator()

        # ChromaDB 인스턴스 생성
        self._initialize_chroma()

    def _initialize_chroma(self) -> None:
        """
        ChromaDB 인스턴스 초기화

        LangChain의 Chroma 클래스를 사용하여 벡터 스토어를 설정합니다.
        기존 컬렉션이 있으면 로드하고, 없으면 새로 생성합니다.

        Raises:
            VectorStoreError: 초기화 실패 시
        """
        try:
            from langchain_chroma import Chroma

            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embedding_generator.embedding_function,
                persist_directory=str(self.persist_directory),
            )

            # 컬렉션 정보 로깅
            stats = self.get_collection_stats()
            self.logger.info(
                f"ChromaDB 연결 완료: {stats['count']}개 문서 로드됨"
            )

        except ImportError as e:
            raise ConfigurationError(
                "langchain-chroma 패키지가 설치되지 않았습니다. "
                "pip install langchain-chroma chromadb 를 실행하세요.",
                original_error=e
            )
        except Exception as e:
            raise VectorStoreError(
                "ChromaDB 초기화에 실패했습니다",
                collection_name=self.collection_name,
                details={"persist_directory": str(self.persist_directory)},
                original_error=e
            )

    # =========================================================================
    # 문서 추가/삭제 메서드
    # =========================================================================

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        문서들을 벡터 스토어에 추가

        문서는 자동으로 임베딩되어 저장됩니다.
        이미 존재하는 ID의 문서는 업데이트됩니다.

        Args:
            documents: 추가할 Document 객체 목록
            ids: 문서 ID 목록. None이면 자동 생성

        Returns:
            List[str]: 추가된 문서들의 ID 목록

        Raises:
            VectorStoreError: 문서 추가 실패 시

        Example:
            >>> docs = [
            ...     Document(page_content="내용1", metadata={"source": "file1.txt"}),
            ...     Document(page_content="내용2", metadata={"source": "file2.txt"}),
            ... ]
            >>> ids = store.add_documents(docs)
            >>> print(f"{len(ids)}개 문서 추가됨")
        """
        if not documents:
            self.logger.warning("추가할 문서가 없습니다")
            return []

        self.logger.info(f"문서 {len(documents)}개 추가 중...")

        try:
            # LangChain Chroma의 add_documents 사용
            result_ids = self._vector_store.add_documents(
                documents=documents,
                ids=ids
            )

            self.logger.info(f"문서 {len(result_ids)}개 추가 완료")
            return result_ids

        except Exception as e:
            raise VectorStoreError(
                "문서 추가에 실패했습니다",
                collection_name=self.collection_name,
                details={"document_count": len(documents)},
                original_error=e
            )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        텍스트 문자열들을 직접 벡터 스토어에 추가

        Document 객체 대신 순수 텍스트를 추가할 때 사용합니다.

        Args:
            texts: 추가할 텍스트 목록
            metadatas: 각 텍스트의 메타데이터 목록
            ids: 문서 ID 목록

        Returns:
            List[str]: 추가된 문서들의 ID 목록

        Example:
            >>> texts = ["첫 번째 문장", "두 번째 문장"]
            >>> metadatas = [{"source": "a"}, {"source": "b"}]
            >>> ids = store.add_texts(texts, metadatas)
        """
        if not texts:
            return []

        try:
            result_ids = self._vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )

            self.logger.info(f"텍스트 {len(result_ids)}개 추가 완료")
            return result_ids

        except Exception as e:
            raise VectorStoreError(
                "텍스트 추가에 실패했습니다",
                collection_name=self.collection_name,
                original_error=e
            )

    def delete(self, ids: List[str]) -> None:
        """
        ID로 문서 삭제

        Args:
            ids: 삭제할 문서 ID 목록

        Raises:
            VectorStoreError: 삭제 실패 시

        Example:
            >>> store.delete(["doc_1", "doc_2"])
        """
        if not ids:
            return

        try:
            self._vector_store.delete(ids=ids)
            self.logger.info(f"문서 {len(ids)}개 삭제 완료")

        except Exception as e:
            raise VectorStoreError(
                "문서 삭제에 실패했습니다",
                collection_name=self.collection_name,
                details={"ids": ids},
                original_error=e
            )

    def delete_collection(self) -> None:
        """
        전체 컬렉션 삭제

        주의: 이 작업은 되돌릴 수 없습니다!
        모든 문서와 임베딩이 삭제됩니다.

        Raises:
            VectorStoreError: 삭제 실패 시

        Example:
            >>> store.delete_collection()
            >>> # 컬렉션이 완전히 삭제됨
        """
        self.logger.warning(f"컬렉션 삭제 중: {self.collection_name}")

        try:
            self._vector_store.delete_collection()
            self.logger.warning(f"컬렉션 삭제 완료: {self.collection_name}")

            # 새 컬렉션으로 재초기화
            self._initialize_chroma()

        except Exception as e:
            raise VectorStoreError(
                "컬렉션 삭제에 실패했습니다",
                collection_name=self.collection_name,
                original_error=e
            )

    # =========================================================================
    # 검색 메서드
    # =========================================================================

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        유사도 기반 문서 검색

        쿼리와 가장 유사한 문서들을 반환합니다.
        코사인 유사도를 기준으로 정렬됩니다.

        Args:
            query: 검색 쿼리 텍스트
            k: 반환할 문서 수 (기본값: 4)
            filter: 메타데이터 필터 조건 (선택)
                   예: {"source": "file.pdf"}

        Returns:
            List[Document]: 유사한 문서 목록 (유사도 높은 순)

        Raises:
            VectorStoreError: 검색 실패 시

        Example:
            >>> # 기본 검색
            >>> results = store.similarity_search("인공지능이란?", k=3)

            >>> # 필터 적용 검색
            >>> results = store.similarity_search(
            ...     "인공지능",
            ...     filter={"source": "ai_book.pdf"}
            ... )
        """
        if not query:
            self.logger.warning("빈 쿼리로 검색 시도")
            return []

        self.logger.debug(f"유사도 검색: query='{query[:50]}...', k={k}")

        try:
            results = self._vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )

            self.logger.debug(f"검색 결과: {len(results)}개 문서")
            return results

        except Exception as e:
            raise VectorStoreError(
                "유사도 검색에 실패했습니다",
                collection_name=self.collection_name,
                details={"query": query[:100], "k": k},
                original_error=e
            )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        유사도 점수와 함께 문서 검색

        각 문서와 함께 유사도 점수(거리)를 반환합니다.
        점수가 낮을수록 더 유사합니다 (코사인 거리 기준).

        Args:
            query: 검색 쿼리 텍스트
            k: 반환할 문서 수
            filter: 메타데이터 필터 조건

        Returns:
            List[tuple]: (Document, score) 튜플 목록
                        score는 거리값 (낮을수록 유사)

        Example:
            >>> results = store.similarity_search_with_score("질문", k=3)
            >>> for doc, score in results:
            ...     print(f"점수: {score:.4f}, 내용: {doc.page_content[:50]}")
        """
        if not query:
            return []

        try:
            results = self._vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )

            self.logger.debug(
                f"검색 결과 (점수 포함): {len(results)}개, "
                f"최고 점수: {results[0][1]:.4f}" if results else "결과 없음"
            )
            return results

        except Exception as e:
            raise VectorStoreError(
                "유사도 검색(점수 포함)에 실패했습니다",
                collection_name=self.collection_name,
                original_error=e
            )

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[tuple]:
        """
        관련성 점수와 함께 문서 검색

        0~1 사이의 정규화된 관련성 점수를 반환합니다.
        1에 가까울수록 더 관련성이 높습니다.

        Args:
            query: 검색 쿼리 텍스트
            k: 반환할 문서 수
            score_threshold: 최소 관련성 점수 (선택)
                            이 점수 미만의 결과는 제외

        Returns:
            List[tuple]: (Document, relevance_score) 튜플 목록
                        relevance_score는 0~1 (높을수록 관련성 높음)

        Example:
            >>> results = store.similarity_search_with_relevance_scores(
            ...     "중요한 질문",
            ...     k=5,
            ...     score_threshold=0.7  # 70% 이상만 반환
            ... )
        """
        try:
            results = self._vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                score_threshold=score_threshold
            )

            return results

        except Exception as e:
            raise VectorStoreError(
                "관련성 검색에 실패했습니다",
                collection_name=self.collection_name,
                original_error=e
            )

    # =========================================================================
    # Retriever 변환
    # =========================================================================

    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        LangChain Retriever로 변환

        RAG 체인에서 사용할 수 있는 Retriever 객체를 반환합니다.

        Args:
            search_type: 검색 유형
                - "similarity": 기본 유사도 검색
                - "mmr": 다양성을 고려한 검색 (Maximum Marginal Relevance)
                - "similarity_score_threshold": 임계값 기반 검색
            search_kwargs: 검색 파라미터
                - k: 반환할 문서 수
                - score_threshold: 최소 점수 (similarity_score_threshold 시)
                - fetch_k: MMR에서 초기 후보 수
                - lambda_mult: MMR 다양성 파라미터

        Returns:
            VectorStoreRetriever: LangChain 호환 Retriever

        Example:
            >>> # 기본 Retriever
            >>> retriever = store.as_retriever()

            >>> # 커스텀 설정
            >>> retriever = store.as_retriever(
            ...     search_type="mmr",
            ...     search_kwargs={"k": 5, "fetch_k": 20}
            ... )

            >>> # RAG 체인에서 사용
            >>> from langchain_core.runnables import RunnablePassthrough
            >>> chain = retriever | format_docs | llm
        """
        settings = get_settings()

        # 기본 검색 파라미터
        default_kwargs = {"k": settings.retrieval_top_k}

        if search_kwargs:
            default_kwargs.update(search_kwargs)

        self.logger.debug(
            f"Retriever 생성: type={search_type}, kwargs={default_kwargs}"
        )

        return self._vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=default_kwargs
        )

    # =========================================================================
    # 유틸리티 메서드
    # =========================================================================

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        컬렉션 통계 정보 반환

        Returns:
            Dict[str, Any]: 컬렉션 정보
                - name: 컬렉션 이름
                - count: 저장된 문서 수
                - persist_directory: 저장 경로

        Example:
            >>> stats = store.get_collection_stats()
            >>> print(f"문서 수: {stats['count']}")
        """
        try:
            # LangChain Chroma의 내부 컬렉션 접근
            collection = self._vector_store._collection
            count = collection.count()

            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": str(self.persist_directory),
            }

        except Exception as e:
            self.logger.warning(f"컬렉션 통계 조회 실패: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "persist_directory": str(self.persist_directory),
                "error": str(e),
            }

    def get_all_documents(
        self,
        limit: Optional[int] = None
    ) -> List[Document]:
        """
        저장된 모든 문서 조회

        주의: 문서가 많으면 메모리 사용량이 높아질 수 있습니다.

        Args:
            limit: 최대 반환 문서 수 (선택)

        Returns:
            List[Document]: 저장된 모든 문서

        Example:
            >>> all_docs = store.get_all_documents(limit=100)
        """
        try:
            collection = self._vector_store._collection

            # 문서 조회
            results = collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )

            # Document 객체로 변환
            documents = []
            for i, content in enumerate(results.get("documents", [])):
                metadata = results.get("metadatas", [{}])[i] or {}
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

            return documents

        except Exception as e:
            self.logger.error(f"문서 조회 실패: {e}")
            return []

    def document_exists(self, doc_id: str) -> bool:
        """
        특정 ID의 문서 존재 여부 확인

        Args:
            doc_id: 확인할 문서 ID

        Returns:
            bool: 문서 존재 여부
        """
        try:
            collection = self._vector_store._collection
            result = collection.get(ids=[doc_id])
            return len(result.get("ids", [])) > 0

        except Exception:
            return False

    @property
    def embedding_dimension(self) -> Optional[int]:
        """
        임베딩 벡터 차원 수

        Returns:
            Optional[int]: 벡터 차원 수 (알 수 없으면 None)
        """
        return self._embedding_generator.dimension
