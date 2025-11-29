"""
Конфигурация RAG-системы v4.0
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""

    level: str = "INFO"
    use_colors: bool = True
    use_emoji: bool = True
    module_width: int = 25
    log_file: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LoggingConfig":
        """Создает LoggingConfig из словаря."""
        logging_data = config_dict.get("logging", {})
        return cls(
            level=logging_data.get("level", "INFO"),
            use_colors=logging_data.get("use_colors", True),
            use_emoji=logging_data.get("use_emoji", True),
            module_width=logging_data.get("module_width", 25),
            log_file=logging_data.get("log_file", None),
        )

    def to_dict(self) -> dict:
        """Конвертирует в словарь для сохранения."""
        result = {
            "level": self.level,
            "use_colors": self.use_colors,
            "use_emoji": self.use_emoji,
            "module_width": self.module_width,
        }
        # log_file не добавляем если None
        if self.log_file:
            result["log_file"] = self.log_file
        return result

    def validate(self) -> bool:
        """Валидация параметров логирования."""
        try:
            assert self.level.upper() in [
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ], f"Invalid log level: {self.level}"
            assert isinstance(self.use_colors, bool), "use_colors должен быть bool"
            assert isinstance(self.use_emoji, bool), "use_emoji должен быть bool"
            assert self.module_width > 0, "module_width должен быть положительным"
            assert self.module_width <= 50, "module_width не должен превышать 50"
            return True
        except AssertionError as e:
            logger.error(f"Logging config validation failed: {e}")
            return False


class Config:
    """Класс конфигурации RAG-системы v4.0."""

    def __init__(self, config_dict: dict = None):
        if config_dict is None:
            config_dict = {}

        # ====================================================================
        # Ollama настройки
        # ====================================================================
        ollama = config_dict.get("ollama", {})
        self.ollama_base_url = ollama.get("base_url", "http://localhost:11434")
        self.text_embedding_model = ollama.get(
            "text_embedding_model", "nomic-embed-text:v1.5"
        )
        self.vlm_model = ollama.get("vlm_model", "qwen3-vl:4b-instruct")
        self.text_generation_model = ollama.get("text_generation_model", "gemma3:4b")
        self.ollama_timeout = ollama.get("timeout", 180.0)

        # ====================================================================
        # Векторная БД
        # ====================================================================
        vector_db = config_dict.get("vector_db", {})
        self.vector_db_path = vector_db.get("path", "./data/rag_engine/chromadb")

        # ====================================================================
        # Хранилище изображений
        # ====================================================================
        storage = config_dict.get("storage", {})
        self.images_storage_path = storage.get(
            "images_path", "./data/rag_engine/images/"
        )

        # ====================================================================
        # Параметры chunking
        # ====================================================================
        chunking = config_dict.get("chunking", {})
        self.chunk_size = chunking.get("chunk_size", 1024)
        self.chunk_overlap = chunking.get("chunk_overlap", 128)

        # ====================================================================
        # Параметры поиска
        # ====================================================================
        search = config_dict.get("search", {})
        self.max_search_results = search.get("max_results", 5)
        self.retrieval_top_k = search.get("retrieval_top_k", 50)
        self.bm25_cache_enabled = search.get("bm25_cache_enabled", True)
        self.bm25_auto_unload = search.get("bm25_auto_unload", False)

        # ====================================================================
        # Query Expansion
        # ====================================================================
        query_expansion = config_dict.get("query_expansion", {})
        self.query_expansion_enabled = query_expansion.get("enabled", True)
        self.query_expansion_max_queries = query_expansion.get("max_queries", 3)

        # ====================================================================
        # Reranker
        # ====================================================================
        reranker = config_dict.get("reranker", {})
        self.reranker_enabled = reranker.get("enabled", True)
        self.reranker_model_name = reranker.get("model_name", "BAAI/bge-reranker-v2-m3")
        self.reranker_auto_unload = reranker.get("auto_unload", True)

        # ====================================================================
        # Логирование (отдельный класс)
        # ====================================================================
        self.logging = LoggingConfig.from_dict(config_dict)

    def validate(self) -> bool:
        """
        Валидация конфигурации.

        Returns:
            True если конфигурация корректна, иначе False
        """
        try:
            # Проверка путей
            Path(self.vector_db_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.images_storage_path).mkdir(parents=True, exist_ok=True)

            # Проверка числовых параметров
            assert self.chunk_size > 0, "chunk_size должен быть положительным"
            assert self.chunk_overlap >= 0, "chunk_overlap не может быть отрицательным"
            assert self.chunk_overlap < self.chunk_size, (
                "chunk_overlap должен быть меньше chunk_size"
            )

            assert self.max_search_results > 0, (
                "max_search_results должен быть положительным"
            )
            assert self.retrieval_top_k > 0, "retrieval_top_k должен быть положительным"
            assert self.retrieval_top_k >= self.max_search_results, (
                "retrieval_top_k должен быть >= max_search_results"
            )

            assert self.ollama_timeout > 0, "ollama_timeout должен быть положительным"

            assert self.query_expansion_max_queries > 0, (
                "query_expansion_max_queries должен быть положительным"
            )
            assert self.query_expansion_max_queries <= 5, (
                "query_expansion_max_queries не должен превышать 5"
            )

            # Проверка строковых параметров
            assert self.ollama_base_url, "ollama_base_url не может быть пустым"
            assert self.text_embedding_model, (
                "text_embedding_model не может быть пустым"
            )
            assert self.vlm_model, "vlm_model не может быть пустым"
            assert self.text_generation_model, (
                "text_generation_model не может быть пустым"
            )

            if self.reranker_enabled:
                assert self.reranker_model_name, (
                    "reranker_model_name не может быть пустым"
                )

            # Валидация логирования
            if not self.logging.validate():
                return False

            logger.info("✓ Configuration validation passed")
            return True

        except AssertionError as e:
            logger.error(f"✗ Configuration validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Configuration validation error: {e}")
            return False

    def to_dict(self) -> dict:
        """
        Конвертация конфигурации в словарь для сохранения.

        Returns:
            Словарь с параметрами конфигурации
        """
        return {
            "ollama": {
                "base_url": self.ollama_base_url,
                "text_embedding_model": self.text_embedding_model,
                "vlm_model": self.vlm_model,
                "text_generation_model": self.text_generation_model,
                "timeout": self.ollama_timeout,
            },
            "vector_db": {
                "path": self.vector_db_path,
            },
            "storage": {
                "images_path": self.images_storage_path,
            },
            "chunking": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            "search": {
                "max_results": self.max_search_results,
                "retrieval_top_k": self.retrieval_top_k,
                "bm25_cache_enabled": self.bm25_cache_enabled,
                "bm25_auto_unload": self.bm25_auto_unload,
            },
            "query_expansion": {
                "enabled": self.query_expansion_enabled,
                "max_queries": self.query_expansion_max_queries,
            },
            "reranker": {
                "enabled": self.reranker_enabled,
                "model_name": self.reranker_model_name,
                "auto_unload": self.reranker_auto_unload,
            },
            "logging": self.logging.to_dict(),
        }

    def print_summary(self):
        """Вывод краткой информации о конфигурации."""
        print("=" * 70)
        print("RAG System v4.0 - Configuration Summary")
        print("=" * 70)
        print(f"Ollama URL:            {self.ollama_base_url}")
        print(f"Text Embedding Model:  {self.text_embedding_model}")
        print(f"VLM Model:             {self.vlm_model}")
        print(f"Text Generation Model: {self.text_generation_model}")
        print("")
        print(
            f"Chunk Size:            {self.chunk_size} chars (overlap: {self.chunk_overlap})"
        )
        print("")
        print("Search Pipeline:")
        print(
            f"  - Query Expansion:   {'Enabled' if self.query_expansion_enabled else 'Disabled'} "
            f"(max {self.query_expansion_max_queries} queries)"
        )
        print(f"  - Retrieval:         Top-{self.retrieval_top_k} candidates")
        print(
            f"  - BM25 Keyword:      Cache: {'Yes' if self.bm25_cache_enabled else 'No'}, "
            f"Auto-unload: {'Yes' if self.bm25_auto_unload else 'No'}"
        )
        print(
            f"  - Reranker:          {'Enabled' if self.reranker_enabled else 'Disabled'}"
        )
        if self.reranker_enabled:
            print(f"    Model: {self.reranker_model_name}")
            print(
                f"    Auto-unload: {'Yes' if self.reranker_auto_unload else 'No (kept in memory)'}"
            )
        print(f"  - Final Results:     Top-{self.max_search_results}")
        print("")
        print("Storage:")
        print(f"  - Vector DB:         {self.vector_db_path}")
        print(f"  - Images:            {self.images_storage_path}")
        print("")
        print("Logging:")
        print(f"  - Level:             {self.logging.level}")
        print(f"  - Colors:            {'Yes' if self.logging.use_colors else 'No'}")
        print(f"  - Emoji:             {'Yes' if self.logging.use_emoji else 'No'}")
        print(f"  - Module Width:      {self.logging.module_width}")
        if self.logging.log_file:
            print(f"  - Log File:          {self.logging.log_file}")
        print("=" * 70)


def load_config(config_path: str = "config.toml") -> Config:
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
            logger.warning("Configuration has validation warnings, but continuing...")

        return config

    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return Config({})

    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def save_config(config: Config, config_path: str = "config.toml") -> None:
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

        with open(config_path, "w", encoding="utf-8") as f:
            toml.dump(config_dict, f)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise


# Пример использования
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Загрузка конфигурации
    config = load_config("config.toml")

    # Вывод информации
    config.print_summary()

    # Тест валидации
    print()
    if config.validate():
        print("✓ Configuration is valid and ready to use")
    else:
        print("✗ Configuration validation failed")
