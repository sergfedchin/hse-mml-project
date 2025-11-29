"""
Красивое цветное логирование для RAG System v4.0
"""

import logging
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.config import LoggingConfig


class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для логов с консистентным форматированием."""
    
    # ANSI escape codes для цветов
    COLORS = {
        'DEBG': '\033[36m',      # Cyan
        'INFO': '\033[32m',      # Green
        'WARN': '\033[33m',      # Yellow
        'ERRO': '\033[31m',      # Red
        'CRIT': '\033[35m',      # Magenta
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    CYAN = '\033[96m'      # Light Cyan для имени модуля
    
    # Эмодзи для уровней
    EMOJI = {
        'DEBG': '🔍',
        'INFO': '✓',
        'WARN': '⚠️',
        'ERRO': '✗',
        'CRIT': '🔥',
    }
    
    # Маппинг стандартных уровней в 4-буквенные
    LEVEL_MAPPING = {
        'DEBUG': 'DEBG',
        'INFO': 'INFO',
        'WARNING': 'WARN',
        'ERROR': 'ERRO',
        'CRITICAL': 'CRIT',
    }
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        use_emoji: bool = True,
        module_width: int = 25
    ):
        """
        Args:
            fmt: Формат сообщения (если None, используется стандартный)
            use_emoji: Использовать ли эмодзи в логах
            module_width: Ширина поля имени модуля (символов)
        """
        self.use_emoji = use_emoji
        self.module_width = module_width
        
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        """Форматирует лог-запись с цветами и консистентным форматом."""
        
        # Конвертируем уровень в 4-буквенный
        short_level = self.LEVEL_MAPPING.get(record.levelname, record.levelname[:4])
        
        # Цвет уровня
        level_color = self.COLORS.get(short_level, self.RESET)
        
        # Эмодзи для уровня
        emoji = self.EMOJI.get(short_level, '') if self.use_emoji else ''
        emoji_str = f"{emoji} " if emoji else ""
        
        # Форматируем время (с датой, dim grey)
        timestamp = self.formatTime(record, self.datefmt)
        colored_timestamp = f"{self.DIM}{timestamp}{self.RESET}"
        
        # Форматируем имя модуля (фиксированная ширина, светло-голубой)
        module_name = record.name
        
        # Обрезаем или дополняем до нужной ширины
        if len(module_name) > self.module_width:
            # Сокращаем: "src.rag.very.long.module" -> "s.r.v.l.module"
            parts = module_name.split('.')
            if len(parts) > 2:
                # Сокращаем все части кроме последней до 1 буквы
                shortened = '.'.join([p[0] for p in parts[:-1]] + [parts[-1]])
                if len(shortened) > self.module_width:
                    # Если всё ещё длинное, обрезаем последнюю часть
                    module_name = shortened[:self.module_width-2] + '..'
                else:
                    module_name = shortened
            else:
                module_name = module_name[:self.module_width-2] + '..'
        
        # Дополняем пробелами справа
        module_name = module_name.ljust(self.module_width)
        colored_module = f"{self.CYAN}{module_name}{self.RESET}"
        
        # Форматируем уровень в квадратных скобках (цветной, жирный)
        colored_level = f"{level_color}{self.BOLD}[{short_level}]{self.RESET}"
        
        # Получаем сообщение
        message = record.getMessage()
        
        # Особая обработка для специальных символов/паттернов в сообщениях
        if message.startswith('✓'):
            # Успешные сообщения - зеленый + жирный
            message = f"{self.COLORS['INFO']}{self.BOLD}{message}{self.RESET}"
        elif message.startswith('✗'):
            # Ошибки - красный + жирный
            message = f"{self.COLORS['ERRO']}{self.BOLD}{message}{self.RESET}"
        elif message.startswith('⊘'):
            # Отключено - dim
            message = f"{self.DIM}{message}{self.RESET}"
        elif '===' in message or '---' in message or message.startswith('====='):
            # Разделители делаем dim
            message = f"{self.DIM}{message}{self.RESET}"
        
        # Собираем итоговое сообщение
        result = f"{colored_timestamp} {emoji_str}{colored_module} {colored_level} {message}"
        
        # Обработка исключений
        if record.exc_info:
            result += '\n' + self.formatException(record.exc_info)
        
        return result


def setup_logging(logging_config: "LoggingConfig") -> None:
    """
    Настраивает систему логирования для RAG System.
    
    Args:
        logging_config: Объект LoggingConfig с настройками логирования
    """
    # Получаем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, logging_config.level.upper()))
    
    # Удаляем существующие обработчики
    root_logger.handlers.clear()
    
    # === Console Handler ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, logging_config.level.upper()))
    
    if logging_config.use_colors and sys.stdout.isatty():
        # Цветной форматтер для консоли
        console_formatter = ColoredFormatter(
            use_emoji=logging_config.use_emoji,
            module_width=logging_config.module_width
        )
    else:
        # Обычный форматтер если цвета не поддерживаются
        console_formatter = logging.Formatter(
            '%(asctime)s %(name)-25s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # === File Handler (опционально) ===
    if logging_config.log_file:
        from pathlib import Path
        
        # Создаем директорию для логов
        Path(logging_config.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(logging_config.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # В файл пишем все
        
        # Обычный форматтер для файла (без цветов)
        file_formatter = logging.Formatter(
            '%(asctime)s %(name)-25s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Отключаем излишний вывод от сторонних библиотек
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    # Логируем информацию о настройке
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info(f"Logging initialized: level={logging_config.level}, "
                f"colors={logging_config.use_colors}, emoji={logging_config.use_emoji}")
    if logging_config.log_file:
        logger.info(f"Log file: {logging_config.log_file}")
    logger.info("=" * 70)


def get_logger(name: str) -> logging.Logger:
    """
    Получает логгер для модуля.
    
    Args:
        name: Имя модуля (обычно __name__)
    
    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)
