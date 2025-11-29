"""
Унифицированные дата-классы для RAG системы v4.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    """Универсальный чанк данных (текст, таблица или изображение)."""
    
    id: str
    content: str  # Текст, Markdown таблицы или описание изображения
    type: str  # "text", "table", "image_content"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    breadcrumbs: Optional[str] = None  # Путь: "Глава 1 > Раздел 2 > Подраздел"
    
    def __post_init__(self):
        """Добавляем breadcrumbs в content для лучшего контекста."""
        if self.breadcrumbs and self.type == "text":
            # Префикс для улучшения векторного поиска
            self.content = f"[{self.breadcrumbs}]\n\n{self.content}"


@dataclass
class SearchResult:
    """Результат поиска из одного источника."""
    
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    source_engine: str  # "vector", "keyword", "hybrid"
    chunk_type: str  # "text", "table", "image_content"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
            "source_engine": self.source_engine,
            "chunk_type": self.chunk_type,
        }


@dataclass
class ImageAnalysis:
    """Результат анализа изображения VLM."""
    
    ocr_text: str = ""
    ui_description: str = ""
    
    def to_text(self, alt_text: str = "") -> str:
        """Конвертирует анализ изображения в текст для embedding."""
        parts = []
        if alt_text:
            parts.append(f"Изображение: {alt_text}")
        if self.ocr_text:
            parts.append(f"Текст на изображении: {self.ocr_text}")
        if self.ui_description:
            parts.append(f"Описание: {self.ui_description}")
        return "\n".join(parts)
