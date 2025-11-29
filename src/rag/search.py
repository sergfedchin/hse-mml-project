"""
Поисковые движки: BM25, Hybrid Retrieval
"""

import gc
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from src.rag.schema import SearchResult
from src.rag.storage import VectorStorage
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class KeywordSearchEngine:
    """BM25 keyword search с кэшированием и lazy loading."""

    def __init__(self, cache_dir: Optional[str] = None, auto_unload: bool = False):
        """
        Args:
            cache_dir: Директория для кэша BM25 индекса (если None, кэш отключен)
            auto_unload: Автоматически выгружать индекс после использования
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.auto_unload = auto_unload

        # Индекс и данные (загружаются лениво)
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_ids: List[str] = []
        self._corpus_contents: List[str] = []
        self._corpus_metadata: List[Dict] = []

        # Хэш для проверки актуальности кэша
        self._corpus_hash: Optional[str] = None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"KeywordSearchEngine: cache enabled at {self.cache_dir}")
        else:
            logger.info("KeywordSearchEngine: cache disabled")

    @property
    def is_loaded(self) -> bool:
        """Проверка, загружен ли индекс в память."""
        return self._bm25 is not None

    def initialize(self, documents: List[Dict]):
        """
        Инициализирует BM25 индекс из документов.

        Args:
            documents: List[{"id": str, "content": str, "metadata": dict}]
        """
        if not documents:
            logger.warning("No documents to initialize BM25")
            return

        # Вычисляем хэш корпуса (для проверки актуальности кэша)
        corpus_hash = self._compute_corpus_hash(documents)

        # Проверяем, есть ли актуальный кэш
        if self.cache_dir and self._load_from_cache(corpus_hash):
            logger.info(
                f"✓ BM25 index loaded from cache ({len(self._corpus_ids)} documents)"
            )
            return

        # Строим индекс с нуля
        logger.info(f"Building BM25 index from {len(documents)} documents...")

        self._corpus_ids = [doc["id"] for doc in documents]
        self._corpus_contents = [doc["content"] for doc in documents]
        self._corpus_metadata = [doc["metadata"] for doc in documents]

        # Токенизация
        tokenized_corpus = [
            content.lower().split() for content in self._corpus_contents
        ]

        self._bm25 = BM25Okapi(tokenized_corpus)
        self._corpus_hash = corpus_hash

        logger.info(f"✓ BM25 index built ({len(self._corpus_ids)} documents)")

        # Сохраняем в кэш
        if self.cache_dir:
            self._save_to_cache()

    def _compute_corpus_hash(self, documents: List[Dict]) -> str:
        """Вычисляет хэш корпуса для проверки актуальности кэша."""
        import hashlib

        # Хэшируем ID всех документов (порядок важен)
        ids_string = "".join(sorted([doc["id"] for doc in documents]))
        return hashlib.md5(ids_string.encode()).hexdigest()

    def _get_cache_path(self) -> Path:
        """Путь к файлу кэша."""
        return self.cache_dir / "bm25_index.pkl"

    def _save_to_cache(self):
        """Сохраняет индекс в кэш."""
        try:
            cache_path = self._get_cache_path()

            cache_data = {
                "corpus_hash": self._corpus_hash,
                "corpus_ids": self._corpus_ids,
                "corpus_contents": self._corpus_contents,
                "corpus_metadata": self._corpus_metadata,
                "bm25": self._bm25,
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)

            logger.info(f"✓ BM25 index saved to cache: {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save BM25 cache: {e}")

    def _load_from_cache(self, expected_hash: str) -> bool:
        """
        Загружает индекс из кэша.

        Returns:
            True если загрузка успешна и кэш актуален
        """
        try:
            cache_path = self._get_cache_path()

            if not cache_path.exists():
                return False

            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Проверяем актуальность кэша
            if cache_data["corpus_hash"] != expected_hash:
                logger.info("BM25 cache is outdated, rebuilding...")
                return False

            # Загружаем данные из кэша
            self._corpus_hash = cache_data["corpus_hash"]
            self._corpus_ids = cache_data["corpus_ids"]
            self._corpus_contents = cache_data["corpus_contents"]
            self._corpus_metadata = cache_data["corpus_metadata"]
            self._bm25 = cache_data["bm25"]

            return True

        except Exception as e:
            logger.warning(f"Failed to load BM25 cache: {e}")
            return False

    def invalidate_cache(self):
        """Удаляет кэш (вызывается при add/delete документа)."""
        if self.cache_dir:
            cache_path = self._get_cache_path()
            if cache_path.exists():
                cache_path.unlink()
                logger.info("✓ BM25 cache invalidated")

    def unload(self):
        """Выгружает индекс из памяти (но сохраняет в кэш)."""
        if self._bm25 is not None:
            logger.info("Unloading BM25 index from memory...")
            self._bm25 = None
            self._corpus_ids = []
            self._corpus_contents = []
            self._corpus_metadata = []
            gc.collect()
            logger.info("✓ BM25 index unloaded")

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """BM25 поиск."""
        if not self._bm25:
            logger.warning("BM25 index not loaded")
            return []

        try:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)

            # Сортируем индексы по убыванию скора
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    chunk_type = self._corpus_metadata[idx].get("type", "text")

                    # Для таблиц берем Markdown из metadata
                    if chunk_type == "table":
                        content = self._corpus_metadata[idx].get(
                            "markdown", self._corpus_contents[idx]
                        )
                    else:
                        content = self._corpus_contents[idx]

                    result = SearchResult(
                        chunk_id=self._corpus_ids[idx],
                        score=float(scores[idx]),
                        content=content,
                        metadata=self._corpus_metadata[idx],
                        source_engine="keyword",
                        chunk_type=chunk_type,
                    )
                    results.append(result)

            # Автоматическая выгрузка после использования
            if self.auto_unload:
                self.unload()

            return results

        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []


class HybridRetriever:
    """Гибридный поиск: Vector + BM25."""

    def __init__(
        self, vector_storage: VectorStorage, keyword_engine: KeywordSearchEngine
    ):
        self.vector_storage = vector_storage
        self.keyword_engine = keyword_engine

    async def search(self, queries: List[str], top_k: int = 50) -> List[SearchResult]:
        """
        Гибридный поиск по нескольким запросам.

        Args:
            queries: Список поисковых запросов
            top_k: Количество уникальных результатов

        Returns:
            Список SearchResult без нормализации скоров (это сделает Reranker)
        """
        all_results = {}  # chunk_id -> SearchResult

        for query in queries:
            # Векторный поиск
            vector_results = await self.vector_storage.search(query, top_k=top_k // 2)

            for result in vector_results:
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                else:
                    # Берем максимальный скор
                    if result.score > all_results[result.chunk_id].score:
                        all_results[result.chunk_id] = result

            # Keyword поиск
            keyword_results = self.keyword_engine.search(query, top_k=top_k // 2)

            for result in keyword_results:
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                else:
                    if result.score > all_results[result.chunk_id].score:
                        all_results[result.chunk_id] = result

        # Возвращаем уникальные результаты
        unique_results = list(all_results.values())

        # Сортируем по скору
        unique_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Hybrid search returned {len(unique_results)} unique candidates")

        return unique_results[:top_k]
