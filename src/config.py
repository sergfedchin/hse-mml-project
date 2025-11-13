"""
Конфигурация системы.
"""

import logging
from pathlib import Path
from typing import List
import toml

logger = logging.getLogger(__name__)


class Config:
    """Класс конфигурации системы"""
    
    def __init__(self, config_dict: dict):
        # Ollama настройки
        self.ollama_base_url = config_dict.get('ollama', {}).get(
            'base_url', 'http://localhost:11434'
        )
        self.text_embedding_model = config_dict.get('ollama', {}).get(
            'text_embedding_model', 'nomic-embed-text'
        )
        self.vision_embedding_model = config_dict.get('ollama', {}).get(
            'vision_embedding_model', 'llava'
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
        self.chunk_size = config_dict.get('chunking', {}).get('chunk_size', 512)
        self.chunk_overlap = config_dict.get('chunking', {}).get('chunk_overlap', 50)
        
        # Параметры поиска
        self.max_search_results = config_dict.get('search', {}).get(
            'max_results', 5
        )
        
        # Логирование
        self.log_level = config_dict.get('logging', {}).get('level', 'INFO')
        
        # CORS
        self.cors_enabled = config_dict.get('cors', {}).get('enabled', False)
        self.cors_origins = config_dict.get('cors', {}).get(
            'origins', ['*']
        )


def load_config(config_path: str) -> Config:
    """
    Загрузить конфигурацию из TOML файла.
    
    Args:
        config_path: Путь к конфигурационному файлу
        
    Returns:
        Объект Config
    """
    try:
        with open(config_path, 'rb') as f:
            config_dict = toml.load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return Config(config_dict)
        
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return Config({})
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise
