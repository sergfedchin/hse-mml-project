"""
Конфигурация системы с поддержкой гибридного поиска.
"""

import logging
from pathlib import Path
from typing import List
import toml

logger = logging.getLogger(__name__)


class HybridSearchConfig:
    """Конфигурация гибридного поиска."""

    def __init__(self, config_dict: dict):
        hybrid = config_dict.get('hybrid_search', {})

        self.weight_semantic = hybrid.get('weight_semantic', 0.7)
        self.weight_lexical = hybrid.get('weight_lexical', 0.3)
        self.top_k_retrieve = hybrid.get('top_k_retrieve', 100)
        self.top_k_rerank = hybrid.get('top_k_rerank', 50)
        self.rrf_k = hybrid.get('rrf_k', 60)

    def validate(self) -> bool:
        """Валидация параметров гибридного поиска."""
        try:
            assert 0 <= self.weight_semantic <= 1, "weight_semantic должен быть в [0, 1]"
            assert 0 <= self.weight_lexical <= 1, "weight_lexical должен быть в [0, 1]"
            assert abs((self.weight_semantic + self.weight_lexical) - 1.0) < 0.01,                 "Сумма весов должна быть ~1.0"
            assert self.top_k_retrieve > 0, "top_k_retrieve должен быть положительным"
            assert self.top_k_rerank > 0, "top_k_rerank должен быть положительным"
            assert self.rrf_k > 0, "rrf_k должен быть положительным"
            return True
        except AssertionError as e:
            logger.error(f"Hybrid search config validation failed: {e}")
            return False


class Config:
    """Класс конфигурации системы с поддержкой всех новых параметров."""

    def __init__(self, config_dict: dict):
        # Ollama настройки
        self.ollama_base_url = config_dict.get('ollama', {}).get(
            'base_url', 'http://localhost:11434'
        )
        self.text_embedding_model = config_dict.get('ollama', {}).get(
            'text_embedding_model', 'nomic-embed-text:v1.5'
        )
        self.vlm_model = config_dict.get('ollama', {}).get(
            'vlm_model', 'qwen3-vl:8b'
        )
        self.ollama_timeout = config_dict.get('ollama', {}).get(
            'timeout', 180.0
        )

        # Векторная БД
        self.vector_db_path = config_dict.get('vector_db', {}).get(
            'path', './data/chromadb'
        )

        # Хранилище изображений
        self.images_storage_path = config_dict.get('storage', {}).get(
            'images_path', './data/images'
        )

        # Параметры chunking
        self.chunk_size = config_dict.get('chunking', {}).get('chunk_size', 1024)
        self.chunk_overlap = config_dict.get('chunking', {}).get('chunk_overlap', 128)

        # Параметры поиска
        self.max_search_results = config_dict.get('search', {}).get('max_results', 5)
        self.similarity_threshold = config_dict.get('search', {}).get(
            'similarity_threshold', 0.7
        )

        # Гибридный поиск
        self.hybrid_search = HybridSearchConfig(config_dict)

        # OCR настройки (PaddleOCR)
        self.ocr_enabled = config_dict.get('ocr', {}).get('enabled', True)
        self.ocr_use_gpu = config_dict.get('ocr', {}).get('use_gpu', False)

        # VLM настройки
        self.vlm_enabled = config_dict.get('vlm', {}).get('enabled', True)
        # Промпты теперь в коде RAGEngine, не в конфиге

        # Обработка таблиц
        self.table_max_chunk_size = config_dict.get('table_processing', {}).get(
            'max_chunk_size', 1000
        )

        # Логирование
        self.log_level = config_dict.get('logging', {}).get('level', 'INFO')

        # CORS
        self.cors_enabled = config_dict.get('cors', {}).get('enabled', False)
        self.cors_origins = config_dict.get('cors', {}).get('origins', ['*'])

    def validate(self) -> bool:
        """
        Валидация конфигурации.

        Returns:
            True если конфигурация корректна, иначе False
        """
        try:
            # Проверка путей
            Path(self.vector_db_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.images_storage_path).parent.mkdir(parents=True, exist_ok=True)

            # Проверка числовых параметров
            assert self.chunk_size > 0, "chunk_size должен быть положительным"
            assert self.chunk_overlap >= 0, "chunk_overlap не может быть отрицательным"
            assert self.chunk_overlap < self.chunk_size,                 "chunk_overlap должен быть меньше chunk_size"
            assert self.max_search_results > 0,                 "max_search_results должен быть положительным"
            assert 0 <= self.similarity_threshold <= 1,                 "similarity_threshold должен быть в диапазоне [0, 1]"
            assert self.ollama_timeout > 0, "ollama_timeout должен быть положительным"
            assert self.table_max_chunk_size > 0,                 "table_max_chunk_size должен быть положительным"

            # Валидация гибридного поиска
            if not self.hybrid_search.validate():
                return False

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def to_dict(self) -> dict:
        """
        Конвертация конфигурации в словарь для сохранения.

        Returns:
            Словарь с параметрами конфигурации
        """
        return {
            'ollama': {
                'base_url': self.ollama_base_url,
                'text_embedding_model': self.text_embedding_model,
                'vlm_model': self.vlm_model,
                'timeout': self.ollama_timeout
            },
            'vector_db': {
                'path': self.vector_db_path
            },
            'storage': {
                'images_path': self.images_storage_path
            },
            'chunking': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            },
            'search': {
                'max_results': self.max_search_results,
                'similarity_threshold': self.similarity_threshold
            },
            'hybrid_search': {
                'weight_semantic': self.hybrid_search.weight_semantic,
                'weight_lexical': self.hybrid_search.weight_lexical,
                'top_k_retrieve': self.hybrid_search.top_k_retrieve,
                'top_k_rerank': self.hybrid_search.top_k_rerank,
                'rrf_k': self.hybrid_search.rrf_k
            },
            'ocr': {
                'enabled': self.ocr_enabled,
                'use_gpu': self.ocr_use_gpu
            },
            'vlm': {
                'enabled': self.vlm_enabled
            },
            'table_processing': {
                'max_chunk_size': self.table_max_chunk_size
            },
            'logging': {
                'level': self.log_level
            },
            'cors': {
                'enabled': self.cors_enabled,
                'origins': self.cors_origins
            }
        }

    def get_device_info(self) -> str:
        """Получение информации об используемом устройстве."""
        if self.ocr_use_gpu:
            return "GPU (CUDA)"
        else:
            return "CPU (optimized for Apple Silicon or x86)"

    def print_summary(self):
        """Вывод краткой информации о конфигурации."""
        print("=" * 60)
        print("RAG Engine Configuration Summary")
        print("=" * 60)
        print(f"Ollama URL: {self.ollama_base_url}")
        print(f"Text Embedding Model: {self.text_embedding_model}")
        print(f"VLM Model: {self.vlm_model}")
        print(f"Device: {self.get_device_info()}")
        print(f"Chunk Size: {self.chunk_size} (overlap: {self.chunk_overlap})")
        print(f"Hybrid Search Weights: {self.hybrid_search.weight_semantic:.1f} semantic / "
              f"{self.hybrid_search.weight_lexical:.1f} lexical")
        print(f"OCR: {'Enabled' if self.ocr_enabled else 'Disabled'} (GPU: {self.ocr_use_gpu})")
        print(f"VLM: {'Enabled' if self.vlm_enabled else 'Disabled'}")
        print("=" * 60)


def load_config(config_path: str) -> Config:
    """
    Загрузить конфигурацию из TOML файла.

    Args:
        config_path: Путь к конфигурационному файлу

    Returns:
        Объект Config
    """
    try:
        config_dict = toml.load(Path(config_path))
        logger.info(f"Configuration loaded from {config_path}")

        config = Config(config_dict)

        # Валидация загруженной конфигурации
        if not config.validate():
            logger.warning("Configuration validation failed, but continuing with loaded values")

        return config

    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return Config({})

    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def save_config(config: Config, config_path: str) -> None:
    """
    Сохранить конфигурацию в TOML файл.

    Args:
        config: Объект Config для сохранения
        config_path: Путь к конфигурационному файлу
    """
    try:
        config_dict = config.to_dict()

        # Создание директории если не существует
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            toml.dump(config_dict, f)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise


# Пример использования
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Загрузка конфигурации
    config = load_config('config.toml')

    # Вывод информации
    config.print_summary()

    # Тест валидации
    if config.validate():
        print("\n✓ Configuration is valid")
    else:
        print("\n✗ Configuration validation failed")
