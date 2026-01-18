"""
=============================================================================
웹 페이지 로더
=============================================================================

이 모듈은 URL에서 웹 페이지 콘텐츠를 로드합니다.

기능:
    - HTTP/HTTPS URL 지원
    - HTML을 텍스트로 변환
    - 메인 콘텐츠 추출 (광고, 네비게이션 제거)
    - 메타데이터 수집 (제목, 도메인 등)

사용 라이브러리:
    - requests: HTTP 요청
    - BeautifulSoup: HTML 파싱
    - LangChain WebBaseLoader: 통합 로더

사용 예시:
    >>> loader = WebDocumentLoader()
    >>> documents = loader.load("https://example.com/article")
    >>> print(documents[0].metadata["title"])
"""

from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import re

from langchain_core.documents import Document

from app.loaders.base_loader import BaseDocumentLoader
from app.utils.logger import get_logger
from app.utils.exceptions import DocumentLoadError

# 모듈 레벨 로거
logger = get_logger(__name__)


class WebDocumentLoader(BaseDocumentLoader):
    """
    웹 페이지 로더

    URL에서 웹 페이지 콘텐츠를 추출하여 Document로 변환합니다.
    HTML 태그를 제거하고 본문 텍스트만 추출합니다.

    Attributes:
        timeout (int): HTTP 요청 타임아웃 (초)
        user_agent (str): 사용자 에이전트 문자열

    Features:
        - HTTP/HTTPS URL 지원
        - HTML 본문 텍스트 추출
        - 불필요한 요소 (스크립트, 스타일) 제거
        - 도메인, 제목 등 메타데이터 자동 추출

    Limitations:
        - JavaScript 렌더링 필요 페이지는 콘텐츠 누락 가능
        - 로그인 필요 페이지 미지원
        - robots.txt 차단 페이지 미지원

    Example:
        >>> loader = WebDocumentLoader(timeout=30)

        >>> # 단일 URL 로드
        >>> docs = loader.load("https://example.com/article")

        >>> # 여러 URL 로드
        >>> docs = loader.load_multiple([
        ...     "https://example.com/page1",
        ...     "https://example.com/page2"
        ... ])
    """

    # 웹 로더는 확장자 기반이 아닌 URL 기반
    SUPPORTED_EXTENSIONS = set()

    # 기본 User-Agent
    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        timeout: int = 30,
        user_agent: Optional[str] = None,
        verify_ssl: bool = True
    ):
        """
        WebDocumentLoader 초기화

        Args:
            timeout: HTTP 요청 타임아웃 (초, 기본값: 30)
            user_agent: 커스텀 User-Agent (None이면 기본값 사용)
            verify_ssl: SSL 인증서 검증 여부 (기본값: True)
        """
        super().__init__()
        self.timeout = timeout
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.verify_ssl = verify_ssl
        self.logger.info(f"WebDocumentLoader 초기화됨, 타임아웃: {timeout}초")

    def load(self, source: str) -> List[Document]:
        """
        URL에서 웹 페이지 콘텐츠 로드

        Args:
            source: 웹 페이지 URL (http:// 또는 https://)

        Returns:
            List[Document]: Document 객체 목록 (보통 1개)

        Raises:
            DocumentLoadError: URL 접근 실패 또는 파싱 실패 시

        Example:
            >>> docs = loader.load("https://news.example.com/article/123")
            >>> print(docs[0].metadata["domain"])  # "news.example.com"
            >>> print(docs[0].metadata["title"])   # 페이지 제목
        """
        self.logger.info(f"웹 페이지 로딩 시작: {source}")

        # URL 유효성 검증
        if not self._is_valid_url(source):
            raise DocumentLoadError(
                f"유효하지 않은 URL입니다: {source}",
                source=source,
                details={"reason": "invalid_url_format"}
            )

        try:
            # URL 접근 가능 여부 확인
            self._check_url_accessible(source)

            # LangChain WebBaseLoader 사용
            from langchain_community.document_loaders import WebBaseLoader

            web_loader = WebBaseLoader(
                web_paths=[source],
                bs_kwargs={
                    "features": "lxml",
                    "parse_only": None,
                },
                requests_kwargs={
                    "timeout": self.timeout,
                    "headers": {"User-Agent": self.user_agent},
                    "verify": self.verify_ssl,
                },
            )

            documents = web_loader.load()

            # 콘텐츠 정제 및 메타데이터 추가
            processed_docs = []
            for doc in documents:
                content = self._clean_web_content(doc.page_content)

                if content:
                    # URL 파싱하여 메타데이터 추가
                    parsed_url = urlparse(source)

                    doc.page_content = content
                    doc.metadata.update({
                        "source": source,
                        "loader_type": self.__class__.__name__,
                        "domain": parsed_url.netloc,
                        "scheme": parsed_url.scheme,
                        "path": parsed_url.path,
                        "content_type": "web_page",
                    })

                    processed_docs.append(doc)

            self.logger.info(
                f"웹 페이지 로딩 완료: {source}, "
                f"콘텐츠 크기: {len(processed_docs[0].page_content) if processed_docs else 0}자"
            )

            return processed_docs

        except ImportError as e:
            raise DocumentLoadError(
                "웹 로더 라이브러리가 설치되지 않았습니다. "
                "pip install beautifulsoup4 lxml 를 실행하세요.",
                source=source,
                original_error=e
            )
        except DocumentLoadError:
            raise
        except Exception as e:
            self.logger.error(f"웹 페이지 로딩 실패: {source}, 오류: {e}")
            raise DocumentLoadError(
                f"웹 페이지 로딩에 실패했습니다",
                source=source,
                details={"error_type": type(e).__name__},
                original_error=e
            )

    def supports(self, source: str) -> bool:
        """
        이 로더가 주어진 소스를 지원하는지 확인

        Args:
            source: 확인할 소스 (URL)

        Returns:
            bool: 유효한 HTTP/HTTPS URL이면 True

        Example:
            >>> loader.supports("https://example.com")  # True
            >>> loader.supports("http://example.com")   # True
            >>> loader.supports("ftp://example.com")    # False
            >>> loader.supports("document.pdf")         # False
        """
        return self._is_valid_url(source)

    def _is_valid_url(self, url: str) -> bool:
        """
        URL 형식 검증

        Args:
            url: 검증할 URL

        Returns:
            bool: 유효한 HTTP/HTTPS URL이면 True
        """
        try:
            result = urlparse(url)
            return all([
                result.scheme in ("http", "https"),
                result.netloc,
            ])
        except Exception:
            return False

    def _check_url_accessible(self, url: str) -> None:
        """
        URL 접근 가능 여부 확인

        HEAD 요청으로 URL에 접근할 수 있는지 확인합니다.

        Args:
            url: 확인할 URL

        Raises:
            DocumentLoadError: URL 접근 불가 시
        """
        import requests

        try:
            response = requests.head(
                url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                verify=self.verify_ssl,
                allow_redirects=True,
            )
            response.raise_for_status()

        except requests.exceptions.Timeout:
            raise DocumentLoadError(
                f"URL 요청 타임아웃 ({self.timeout}초 초과)",
                source=url,
                details={"reason": "timeout"}
            )
        except requests.exceptions.SSLError as e:
            raise DocumentLoadError(
                "SSL 인증서 검증 실패. verify_ssl=False로 시도해보세요.",
                source=url,
                details={"reason": "ssl_error"},
                original_error=e
            )
        except requests.exceptions.ConnectionError as e:
            raise DocumentLoadError(
                "URL에 연결할 수 없습니다. 네트워크 상태를 확인하세요.",
                source=url,
                details={"reason": "connection_error"},
                original_error=e
            )
        except requests.exceptions.HTTPError as e:
            raise DocumentLoadError(
                f"HTTP 오류: {e.response.status_code}",
                source=url,
                details={"status_code": e.response.status_code},
                original_error=e
            )

    def _clean_web_content(self, content: str) -> str:
        """
        웹 콘텐츠 정제

        HTML 잔여물, 과도한 공백 등을 제거합니다.

        Args:
            content: 원본 콘텐츠

        Returns:
            str: 정제된 콘텐츠
        """
        if not content:
            return ""

        # 연속된 공백/줄바꿈 정리
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)

        # 각 줄 앞뒤 공백 제거
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(lines)

        return content.strip()

    def load_multiple(
        self,
        urls: List[str],
        fail_on_error: bool = False
    ) -> List[Document]:
        """
        여러 URL에서 콘텐츠 로드

        Args:
            urls: URL 목록
            fail_on_error: True면 하나라도 실패 시 예외 발생
                          False면 실패한 URL 건너뛰고 계속 진행

        Returns:
            List[Document]: 로드된 모든 Document 목록

        Example:
            >>> urls = [
            ...     "https://example.com/page1",
            ...     "https://example.com/page2",
            ...     "https://example.com/page3",
            ... ]
            >>> docs = loader.load_multiple(urls)
            >>> print(f"총 {len(docs)}개 페이지 로드됨")
        """
        self.logger.info(f"다중 URL 로딩: {len(urls)}개")

        all_documents = []
        errors = []

        for url in urls:
            try:
                docs = self.load(url)
                all_documents.extend(docs)
            except DocumentLoadError as e:
                self.logger.warning(f"URL 로딩 실패: {url}, 오류: {e}")
                errors.append({"url": url, "error": str(e)})

                if fail_on_error:
                    raise

        if errors:
            self.logger.warning(
                f"일부 URL 로딩 실패: {len(errors)}/{len(urls)}개"
            )

        self.logger.info(
            f"다중 URL 로딩 완료: {len(all_documents)}개 문서"
        )

        return all_documents

    def extract_links(self, url: str) -> List[str]:
        """
        페이지에서 링크 추출

        페이지 내의 모든 하이퍼링크를 추출합니다.

        Args:
            url: 웹 페이지 URL

        Returns:
            List[str]: 추출된 링크 목록 (절대 URL)

        Example:
            >>> links = loader.extract_links("https://example.com")
            >>> for link in links[:5]:
            ...     print(link)
        """
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                verify=self.verify_ssl,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "lxml")

            links = []
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                # 상대 URL을 절대 URL로 변환
                absolute_url = urljoin(url, href)
                # HTTP/HTTPS URL만 포함
                if absolute_url.startswith(("http://", "https://")):
                    links.append(absolute_url)

            # 중복 제거
            return list(set(links))

        except Exception as e:
            self.logger.warning(f"링크 추출 실패: {url}, 오류: {e}")
            return []

    def get_page_metadata(self, url: str) -> Dict[str, Any]:
        """
        페이지 메타데이터 추출

        페이지의 제목, 설명, 키워드 등 메타 정보를 추출합니다.

        Args:
            url: 웹 페이지 URL

        Returns:
            dict: 페이지 메타데이터
                 - title: 페이지 제목
                 - description: 메타 설명
                 - keywords: 키워드
                 - language: 언어
                 - author: 저자

        Example:
            >>> metadata = loader.get_page_metadata("https://example.com")
            >>> print(f"제목: {metadata.get('title')}")
        """
        import requests
        from bs4 import BeautifulSoup

        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                verify=self.verify_ssl,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "lxml")

            metadata = {
                "url": url,
                "domain": urlparse(url).netloc,
            }

            # 제목
            title_tag = soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text().strip()

            # 메타 태그
            for meta in soup.find_all("meta"):
                name = meta.get("name", "").lower()
                content = meta.get("content", "")

                if name == "description":
                    metadata["description"] = content
                elif name == "keywords":
                    metadata["keywords"] = content
                elif name == "author":
                    metadata["author"] = content

            # HTML lang 속성
            html_tag = soup.find("html")
            if html_tag and html_tag.get("lang"):
                metadata["language"] = html_tag.get("lang")

            return metadata

        except Exception as e:
            self.logger.warning(f"메타데이터 추출 실패: {url}, 오류: {e}")
            return {"url": url, "error": str(e)}
