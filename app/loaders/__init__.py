"""
=============================================================================
문서 로더 모듈
=============================================================================

이 패키지는 다양한 형식의 문서를 로드하는 기능을 제공합니다.

지원 형식:
    - PDF: PDF 문서 텍스트 추출
    - 텍스트: txt, md, markdown 파일
    - 웹: URL에서 웹 페이지 콘텐츠 추출

설계 원칙:
    - 팩토리 패턴을 사용하여 적절한 로더 자동 선택
    - 추상 베이스 클래스로 새로운 로더 쉽게 추가 가능
    - 각 로더는 독립적으로 테스트 가능

새로운 문서 형식 추가 방법:
    1. BaseDocumentLoader를 상속받는 새 클래스 생성
    2. load()와 supports() 메서드 구현
    3. LoaderFactory에 새 로더 등록
"""

from app.loaders.base_loader import BaseDocumentLoader
from app.loaders.pdf_loader import PDFDocumentLoader
from app.loaders.text_loader import TextDocumentLoader
from app.loaders.web_loader import WebDocumentLoader
from app.loaders.loader_factory import DocumentLoaderFactory

__all__ = [
    "BaseDocumentLoader",
    "PDFDocumentLoader",
    "TextDocumentLoader",
    "WebDocumentLoader",
    "DocumentLoaderFactory",
]
