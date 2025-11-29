"""
RAG System v4.0 - Модульная мультимодальная система
"""

from src.rag.ingestion import DocumentProcessor, RSTTableParser
from src.rag.main import RAGSystem
from src.rag.query_processing import QueryProcessor
from src.rag.ranking import CrossEncoderRanker
from src.rag.schema import DocumentChunk, ImageAnalysis, SearchResult
from src.rag.search import HybridRetriever, KeywordSearchEngine
from src.rag.storage import VectorStorage

__version__ = "4.0.0"

__all__ = [
    "RAGSystem",
    "DocumentProcessor",
    "VectorStorage",
    "KeywordSearchEngine",
    "HybridRetriever",
    "QueryProcessor",
    "CrossEncoderRanker",
    "DocumentChunk",
    "SearchResult",
    "ImageAnalysis",
    "RSTTableParser",
]
