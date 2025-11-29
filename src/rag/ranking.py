"""
Реранкинг результатов поиска через Cross-Encoder
"""

import gc
from typing import List, Optional

from sentence_transformers import CrossEncoder

from src.rag.schema import SearchResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoderRanker:
    """Реранкер на основе Cross-Encoder модели с ленивой загрузкой."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", auto_unload: bool = True):
        """
        Args:
            model_name: Название модели Cross-Encoder
                Рекомендуемые:
                - BAAI/bge-reranker-v2-m3 (multilingual, 568M params)
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (English, lighter)
            auto_unload: Автоматически выгружать модель после использования
        """
        self.model_name = model_name
        self.auto_unload = auto_unload
        self._model: Optional[CrossEncoder] = None  # Модель НЕ загружена
        
        logger.info(f"CrossEncoderRanker created (model will be loaded on first use): {model_name}")
    
    def _load_model(self):
        """Ленивая загрузка модели при первом вызове."""
        if self._model is None:
            logger.info(f"Loading CrossEncoder model: {self.model_name}...")
            try:
                self._model = CrossEncoder(self.model_name, max_length=512)
                logger.info(f"✓ CrossEncoder model loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder: {e}")
                raise
    
    def _unload_model(self):
        """Выгрузка модели из памяти."""
        if self._model is not None:
            logger.info("Unloading CrossEncoder model from memory...")
            del self._model
            self._model = None
            gc.collect()
            logger.info("✓ CrossEncoder model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель в память."""
        return self._model is not None
    
    def rank(
        self,
        user_query: str,
        candidates: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Реранкинг кандидатов по релевантности к исходному запросу.
        
        Модель загружается автоматически при первом вызове и может быть
        выгружена после использования (если auto_unload=True).
        
        Args:
            user_query: Оригинальный запрос пользователя (до расширения)
            candidates: Список кандидатов из гибридного поиска
            top_k: Количество лучших результатов
        
        Returns:
            Отсортированный список топ-K результатов
        """
        if not candidates:
            return []
        
        try:
            # Загружаем модель если не загружена
            self._load_model()
            
            # Формируем пары (query, candidate_content)
            pairs = [[user_query, candidate.content] for candidate in candidates]
            
            # Предсказываем скоры релевантности
            scores = self._model.predict(pairs)
            
            # Обновляем скоры в SearchResult
            for i, candidate in enumerate(candidates):
                candidate.score = float(scores[i])
            
            # Сортируем по убыванию скора
            candidates.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Reranked {len(candidates)} candidates, returning top-{top_k}")
            
            result = candidates[:top_k]
            
            # Автоматическая выгрузка после использования
            if self.auto_unload:
                self._unload_model()
            
            return result
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: возвращаем первые top_k без реранкинга
            return candidates[:top_k]
    
    def __del__(self):
        """Выгрузка модели при удалении объекта."""
        self._unload_model()
