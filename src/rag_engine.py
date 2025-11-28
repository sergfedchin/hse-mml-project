"""Мультимодальный RAG Engine v3.0 с гибридным поиском для изображений"""

import asyncio
import base64
import gc
import json
import logging
import re
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import httpx
import json_repair
import ollama
import pypandoc
from chromadb.config import Settings
from docutils import nodes

# Импорт docutils для парсинга RST таблиц
from docutils.core import publish_doctree

# from docutils.parsers.rst import directives, roles, tableparser
from langchain_text_splitters import MarkdownTextSplitter
from PIL import Image as PILImage
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from src.config import Config

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Schema для Structured Output
# ============================================================================


class ImageAnalysisResult(BaseModel):
    """Структура ответа VLM для анализа изображения."""

    ocr_text: str = Field(
        default='',
        description="Весь видимый текст на изображении, построчно. Если текста нет, пустая строка."
    )
    ui_description: str = Field(
        description="Техническое описание интерфейса на русском языке: элементы UI, их расположение, структура, функции."
    )


# ============================================================================
# Промпт (объединенный)
# ============================================================================

UNIFIED_VLM_PROMPT = """Проанализируй этот скриншот и верни JSON с двумя полями:

1. "ocr_text": Извлеки ВЕСЬ видимый текст на изображении точно как он написан, построчно. Если текста нет, оставь пустую строку.

2. "ui_description": Опиши пользовательский интерфейс с техническими подробностями на русском:
   - Основные элементы (кнопки, поля, выпадающие списки, вкладки, меню)
   - Структура макета и визуальная иерархия
   - Все видимые текстовые надписи и их расположение
   - Интерактивные элементы и их функции
   - Таблицы данных или структурированная информация
   - Значки, индикаторы, элементы статуса

Сосредоточься на функциональных аспектах. Отвечай строго на русском языке БЕЗ markdown форматирования.

Верни строго валидный JSON без дополнительных пояснений.
"""

# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class MultimodalChunk:
    """Унифицированный класс для результатов поиска."""

    chunk_id: str
    type: str
    score: float
    document_url: str
    file_id: str
    content: Optional[str] = None
    section_header: Optional[str] = None
    chunk_index: Optional[int] = None
    table_metadata: Optional[Dict[str, Any]] = None
    # table_structure удалена - не используется
    table_content: Optional[List[Dict[str, Any]]] = None
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    alt_text: Optional[str] = None
    ocr_text: Optional[str] = None
    vlm_description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ============================================================================
# RST Table Parser Helper
# ============================================================================


class RSTTableParser:
    """Парсер RST таблиц с упрощением сложных структур для LLM."""

    @staticmethod
    def remove_rst_tables_by_pattern(rst_content: str) -> str:
        """Удаляет RST таблицы по паттернам."""
        # Grid table pattern
        grid_table_pattern = r"\n\+[-=+]+\+\n(?:\|[^\n]*\|\n)+\+[-=+]+\+(?:\n(?:\|[^\n]*\|\n)+\+[-=+]+\+)*"
        modified_rst = re.sub(
            grid_table_pattern, "\n\n[TABLE_PLACEHOLDER]\n\n", rst_content
        )
        return modified_rst

    @staticmethod
    def parse_rst_tables(rst_content: str) -> List[Dict[str, Any]]:
        """Парсит RST таблицы используя docutils и упрощает их структуру."""
        tables = []
        try:
            # Настройки для подавления вывода
            settings_overrides = {
                "report_level": 5,  # Только критичные ошибки
                "halt_level": 5,  # Не останавливаться
                "warning_stream": StringIO(),  # Перенаправляем предупреждения
            }

            # Подавляем stderr во время парсинга
            with redirect_stderr(StringIO()):
                doctree = publish_doctree(
                    rst_content, settings_overrides=settings_overrides
                )

            # Извлекаем таблицы
            for table_node in doctree.traverse(nodes.table):
                try:
                    parsed_table = RSTTableParser._parse_single_table(table_node)
                    if parsed_table and (
                        parsed_table["headers"] or parsed_table["rows"]
                    ):
                        tables.append(parsed_table)
                except Exception as e:
                    logger.warning(f"Failed to parse single table: {e}")
                    continue

        except Exception as e:
            logger.error(f"RST table parsing error: {e}")

        return tables

    @staticmethod
    def _parse_single_table(table_node: nodes.table) -> Optional[Dict[str, Any]]:
        """Парсит одну таблицу из doctree node."""
        result = {"caption": "", "headers": [], "rows": []}

        # Извлекаем caption
        parent = table_node.parent
        if parent:
            for sibling in parent.children:
                if isinstance(sibling, nodes.caption):
                    result["caption"] = RSTTableParser._get_clean_text(sibling)
                    break

        # Ищем tgroup
        tgroup = None
        for child in table_node.children:
            if isinstance(child, nodes.tgroup):
                tgroup = child
                break

        if not tgroup:
            return None

        # Количество колонок
        num_cols = int(tgroup.get('cols', 0))
        if num_cols == 0:
            colspecs = [c for c in tgroup.children if isinstance(c, nodes.colspec)]
            num_cols = len(colspecs)

        if num_cols == 0:
            return None

        # Парсим thead и tbody
        thead = None
        tbody = None
        for child in tgroup.children:
            if isinstance(child, nodes.thead):
                thead = child
            elif isinstance(child, nodes.tbody):
                tbody = child

        # Заголовки
        if thead:
            for row in thead.children:
                if isinstance(row, nodes.row):
                    # Для заголовков используем простой парсинг (обычно там нет rowspan)
                    header_row = RSTTableParser._parse_row_simple(row, num_cols)
                    if header_row:
                        result["headers"] = header_row
                        break

        # Если заголовков нет, проверяем первую строку tbody
        if not result["headers"] and tbody:
            row_nodes = [r for r in tbody.children if isinstance(r, nodes.row)]
            if row_nodes:
                # Пытаемся использовать первую строку как заголовок
                first_row = RSTTableParser._parse_row_simple(row_nodes[0], num_cols)
                if first_row:
                    result["headers"] = first_row
                    # Удаляем первую строку из tbody для дальнейшей обработки
                    tbody.children = tbody.children[1:]
                    logger.info("Первая строка таблицы использована как заголовок")

        if not result["headers"]:
            result["headers"] = [f"col_{i}" for i in range(num_cols)]

        # Тело таблицы - используем новый метод с поддержкой morerows
        if tbody:
            result["rows"] = RSTTableParser._parse_tbody_with_rowspan(tbody, num_cols)

        return result

    @staticmethod
    def _parse_row_simple(row_node: nodes.row, expected_cols: int) -> Optional[List[str]]:
        """Простой парсинг строки без учета rowspan (для заголовков)."""
        cells = []
        col_idx = 0

        for entry in row_node.children:
            if not isinstance(entry, nodes.entry):
                continue

            cell_text = RSTTableParser._get_clean_text(entry)
            morecols = int(entry.get('morecols', 0))

            # Дублируем значение для merged колонок
            for i in range(morecols + 1):
                if col_idx < expected_cols:
                    cells.append(cell_text)
                    col_idx += 1

        while len(cells) < expected_cols:
            cells.append("")

        cells = cells[:expected_cols]
        return cells if cells else None

    @staticmethod
    def _parse_tbody_with_rowspan(tbody: nodes.tbody, num_cols: int) -> List[List[str]]:
        """
        Парсит tbody с учетом rowspan (morerows) и colspan (morecols).
        Дублирует значения ячеек с rowspan в последующие строки.
        """
        rows = []
        # Матрица для отслеживания занятых ячеек
        # rowspan_matrix[row_idx][col_idx] = (text, remaining_rows)
        rowspan_matrix = {}

        row_nodes = [r for r in tbody.children if isinstance(r, nodes.row)]

        for row_idx, row_node in enumerate(row_nodes):
            row_cells = [''] * num_cols
            col_idx = 0

            # Сначала заполняем ячейки из предыдущих rowspan
            if row_idx in rowspan_matrix:
                for col, (text, remaining) in rowspan_matrix[row_idx].items():
                    if col < num_cols:
                        row_cells[col] = text

            # Теперь обрабатываем ячейки текущей строки
            for entry in row_node.children:
                if not isinstance(entry, nodes.entry):
                    continue

                # Пропускаем уже занятые колонки (из rowspan)
                while col_idx < num_cols and row_cells[col_idx] != '':
                    col_idx += 1

                if col_idx >= num_cols:
                    break

                cell_text = RSTTableParser._get_clean_text(entry)
                morecols = int(entry.get('morecols', 0))
                morerows = int(entry.get('morerows', 0))

                # Заполняем текущую ячейку и colspan
                for i in range(morecols + 1):
                    if col_idx < num_cols:
                        row_cells[col_idx] = cell_text

                        # Если есть rowspan, записываем в матрицу для следующих строк
                        if morerows > 0:
                            for future_row in range(row_idx + 1, row_idx + morerows + 1):
                                if future_row not in rowspan_matrix:
                                    rowspan_matrix[future_row] = {}
                                rowspan_matrix[future_row][col_idx] = (cell_text, morerows)

                        col_idx += 1

            rows.append(row_cells)

        return rows

    @staticmethod
    def _get_clean_text(node: nodes.Node) -> str:
        """
        Извлекает текст из node и очищает от ошибок docutils.
        """
        text_parts = []

        def collect_text(n):
            """Рекурсивно собирает текст, пропуская проблемные узлы."""
            # Пропускаем system_message и problematic узлы
            if isinstance(n, (nodes.system_message, nodes.problematic)):
                return

            # Если это текстовый узел, добавляем его
            if isinstance(n, nodes.Text):
                text_parts.append(str(n))

            # Рекурсивно обходим дочерние узлы
            if hasattr(n, 'children'):
                for child in n.children:
                    collect_text(child)

        collect_text(node)

        # Собираем текст
        result = " ".join(text_parts).strip()

        # Очищаем от множественных пробелов
        result = re.sub(r'\s+', ' ', result)

        # ПОСТОБРАБОТКА: удаляем сообщения об ошибках docutils
        # Удаляем "No role entry for ..."
        result = re.sub(r'No role entry for "[^"]*" in module "[^"]*"\.\s*', '', result)

        # Удаляем "Trying ... as canonical role name."
        result = re.sub(r'Trying "[^"]*" as canonical role name\.\s*', '', result)

        # Удаляем "Unknown interpreted text role ..."
        result = re.sub(r'Unknown interpreted text role "[^"]*"\.\s*', '', result)

        # Удаляем общий паттерн ошибок docutils
        result = re.sub(r':\d+: \([A-Z]+/\d+\) [^\n]+\n?', '', result)

        # Очищаем повторные пробелы после удаления
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    @staticmethod
    def table_to_markdown(table_data: Dict[str, Any]) -> str:
        """Конвертирует таблицу в Markdown."""
        md_lines = []

        if table_data.get("caption"):
            md_lines.append(f"**{table_data['caption']}**\n")

        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        if not rows and not headers:
            return ""

        num_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
        if num_cols == 0:
            return ""

        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        for row in rows:
            normalized_row = (row + [""] * num_cols)[:num_cols]
            escaped_row = [cell.replace("|", "\\|").replace("\n", " ") for cell in normalized_row]
            md_lines.append("| " + " | ".join(escaped_row) + " |")

        return "\n".join(md_lines)


# ============================================================================
# RAG Engine
# ============================================================================


class RAGEngine:
    def __init__(self, config: Config):
        self.config = config

        # ChromaDB
        self.chroma_client = None
        self.text_collection = None
        self.image_collection = None
        self.table_collection = None

        # Ollama HTTP client
        self.ollama_client = None

        # BM25 индексы
        self._bm25_text = None
        self._bm25_text_corpus = []
        self._bm25_text_ids = []

        self._bm25_tables = None
        self._bm25_tables_corpus = []
        self._bm25_tables_ids = []

        self._bm25_images = None
        self._bm25_images_corpus = []
        self._bm25_images_ids = []

        self.documents_metadata = {}
        self.hybrid_config = config.hybrid_search

        logger.info("Multimodal RAG Engine v3.0 initialized")

    async def initialize(self):
        """Инициализация всех компонентов."""
        try:
            Path(self.config.images_storage_path).mkdir(parents=True, exist_ok=True)
            Path(self.config.vector_db_path).mkdir(parents=True, exist_ok=True)

            # ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.vector_db_path,
                settings=Settings(anonymized_telemetry=False),
            )

            self.text_collection = self.chroma_client.get_or_create_collection(
                name="text_chunks",
                metadata={"description": "Text content with embeddings"},
            )

            self.image_collection = self.chroma_client.get_or_create_collection(
                name="images", metadata={"description": "Images with visual embeddings"}
            )

            self.table_collection = self.chroma_client.get_or_create_collection(
                name="tables", metadata={"description": "Tables with row-level chunks"}
            )

            logger.info("Vector DB initialized with 3 collections")

            # Ollama client
            self.ollama_client = ollama.Client(
                self.config.ollama_base_url,
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            )

            # Проверка подключения
            try:
                _ = self.ollama_client.list()
                logger.info("Ollama connection verified")
            except Exception as e:
                logger.error(f"Cannot connect to Ollama: {e}")

            try:
                self.ollama_client.show(self.config.vlm_model)
                logger.info(
                    f"Model {self.config.vlm_model} successfully found in Ollama!"
                )
            except Exception as e:
                logger.error(
                    f"Model {self.config.vlm_model} not found in Ollama. Available models: {self.ollama_client.list().models}. Error: {str(e)}"
                )
                raise

            # Инициализация BM25
            await self._initialize_bm25_indices()

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            raise

    async def _initialize_bm25_indices(self):
        """Инициализация BM25 индексов из существующих данных."""
        try:
            # Текстовые чанки
            text_results = self.text_collection.get()
            if text_results["ids"]:
                self._bm25_text_ids = text_results["ids"]
                self._bm25_text_corpus = [
                    doc.lower().split() for doc in text_results["documents"]
                ]
                self._bm25_text = BM25Okapi(self._bm25_text_corpus)
                logger.info(
                    f"BM25 text index initialized with {len(self._bm25_text_ids)} documents"
                )

            # Таблицы
            table_results = self.table_collection.get()
            if table_results["ids"]:
                self._bm25_tables_ids = table_results["ids"]
                self._bm25_tables_corpus = []
                for doc in table_results["documents"]:
                    # Используем документ (embedding_text) напрямую
                    self._bm25_tables_corpus.append(doc.lower().split())
                self._bm25_tables = BM25Okapi(self._bm25_tables_corpus)
                logger.info(
                    f"BM25 table index initialized with {len(self._bm25_tables_ids)} table chunks"
                )

            # Изображения
            image_results = self.image_collection.get()
            if image_results["ids"]:
                self._bm25_images_ids = image_results["ids"]
                self._bm25_images_corpus = []
                for meta in image_results["metadatas"]:
                    text = (
                        f"{meta.get('alt_text', '')} {meta.get('ocr_text', '')}".lower()
                    )
                    self._bm25_images_corpus.append(text.split())
                self._bm25_images = BM25Okapi(self._bm25_images_corpus)
                logger.info(
                    f"BM25 image index initialized with {len(self._bm25_images_ids)} images"
                )

        except Exception as e:
            logger.warning(f"Could not initialize BM25 indices: {e}")

    async def cleanup(self):
        """Очистка ресурсов."""
        self._bm25_text = None
        self._bm25_tables = None
        self._bm25_images = None
        gc.collect()
        logger.info("RAG Engine cleaned up")

    def is_ready(self) -> bool:
        """Проверка готовности системы."""
        return all(
            [
                self.chroma_client,
                self.text_collection,
                self.image_collection,
                self.table_collection,
                self.ollama_client,
            ]
        )

    # ========================================================================
    # НОРМАЛИЗАЦИЯ СКОРОВ
    # ========================================================================

    def _normalize_scores_minmax(
        self, chunks: List[MultimodalChunk]
    ) -> List[MultimodalChunk]:
        """Нормализует скоры в диапазон [0, 1] используя min-max нормализацию."""
        if not chunks:
            return chunks

        scores = [c.score for c in chunks]
        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score < 1e-9:
            for chunk in chunks:
                chunk.score = 1.0 if chunk.score > 0 else 0.001
            return chunks

        for chunk in chunks:
            chunk.score = (chunk.score - min_score) / (max_score - min_score)

        return chunks

    # ========================================================================
    # ДОБАВЛЕНИЕ ДОКУМЕНТА
    # ========================================================================

    async def add_document(
        self, rst_content: str, document_url: str, images_base_path: str
    ) -> str:
        """Добавление документа с полной мультимодальной обработкой."""
        file_id = str(uuid.uuid4())
        logger.info(f"Processing document {file_id}: {document_url}")

        try:
            # 1. Извлекаем таблицы из RST
            parsed_tables = RSTTableParser.parse_rst_tables(rst_content)
            logger.info(f"Extracted {len(parsed_tables)} tables from RST")

            # 2. Удаляем таблицы из RST текста
            rst_without_tables = RSTTableParser.remove_rst_tables_by_pattern(
                rst_content
            )
            logger.debug(f"RST after table removal: {len(rst_without_tables)} chars")

            # 3. Конвертация очищенного RST → Markdown
            markdown_content = await self._rst_to_markdown(rst_without_tables)
            logger.debug(f"Converted to Markdown ({len(markdown_content)} chars)")

            # 4. Обработка таблиц в структурированный формат с чанкингом
            tables_chunks = await self._process_parsed_tables(parsed_tables, file_id)
            logger.info(f"Processed {len(tables_chunks)} table chunks")

            # 5. Извлечение и обработка изображений
            images_info = await self._extract_and_process_images(
                markdown_content, images_base_path, file_id
            )
            logger.info(f"Processed {len(images_info)} images")

            # 6. Разбиение текста на чанки (таблиц уже нет в тексте)
            text_chunks = await self._split_text_with_context(
                markdown_content, file_id, document_url
            )
            logger.info(f"Created {len(text_chunks)} text chunks")

            # 7. Индексирование
            await self._embed_and_store_text_chunks(text_chunks)
            await self._embed_and_store_images(images_info, file_id, document_url)
            await self._embed_and_store_tables(tables_chunks, file_id, document_url)

            # 8. Обновление BM25
            await self._update_bm25_indices(text_chunks, tables_chunks, images_info)

            # 9. Сохранение метаданных
            self.documents_metadata[file_id] = {
                "document_url": document_url,
                "chunks_count": len(text_chunks),
                "images_count": len(images_info),
                "tables_chunks_count": len(tables_chunks),
                "indexed_at": asyncio.get_event_loop().time(),
            }

            logger.info(f"Document {file_id} indexed successfully")
            return file_id

        except Exception as e:
            logger.error(f"Error adding document {file_id}: {e}", exc_info=True)
            await self._rollback_document(file_id)
            raise

    # ========================================================================
    # УДАЛЕНИЕ ДОКУМЕНТА
    # ========================================================================

    async def delete_document(self, file_id: str) -> bool:
        """Полное удаление документа из системы."""
        if file_id not in self.documents_metadata:
            logger.warning(f"Document {file_id} not found in metadata")
            return False

        logger.info(f"Starting deletion of document {file_id}")

        try:
            await self._delete_text_chunks(file_id)
            await self._delete_tables(file_id)
            await self._delete_images(file_id)

            del self.documents_metadata[file_id]

            logger.info(f"Document {file_id} successfully deleted")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {file_id}: {e}", exc_info=True)
            raise

    async def _delete_text_chunks(self, file_id: str):
        """Удаление текстовых чанков."""
        try:
            results = self.text_collection.get(where={"file_id": file_id})
            if not results["ids"]:
                return

            chunk_ids = results["ids"]
            self.text_collection.delete(ids=chunk_ids)

            if self._bm25_text:
                new_ids = []
                new_corpus = []
                for idx, doc_id in enumerate(self._bm25_text_ids):
                    if doc_id not in chunk_ids:
                        new_ids.append(doc_id)
                        new_corpus.append(self._bm25_text_corpus[idx])

                self._bm25_text_ids = new_ids
                self._bm25_text_corpus = new_corpus
                self._bm25_text = BM25Okapi(new_corpus) if new_corpus else None

        except Exception as e:
            logger.error(f"Error deleting text chunks: {e}")
            raise

    async def _delete_tables(self, file_id: str):
        """Удаление табличных чанков."""
        try:
            results = self.table_collection.get(where={"file_id": file_id})
            if not results["ids"]:
                return

            chunk_ids = results["ids"]
            self.table_collection.delete(ids=chunk_ids)

            if self._bm25_tables:
                new_ids = []
                new_corpus = []
                for idx, chunk_id in enumerate(self._bm25_tables_ids):
                    if chunk_id not in chunk_ids:
                        new_ids.append(chunk_id)
                        new_corpus.append(self._bm25_tables_corpus[idx])

                self._bm25_tables_ids = new_ids
                self._bm25_tables_corpus = new_corpus
                self._bm25_tables = BM25Okapi(new_corpus) if new_corpus else None

        except Exception as e:
            logger.error(f"Error deleting table chunks: {e}")
            raise

    async def _delete_images(self, file_id: str):
        """Удаление изображений."""
        try:
            results = self.image_collection.get(where={"file_id": file_id})
            if not results["ids"]:
                return

            image_ids = results["ids"]
            metadatas = results["metadatas"]

            # Удаление файлов
            for metadata in metadatas:
                image_path = metadata.get("image_path")
                if image_path:
                    try:
                        Path(image_path).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Failed to delete {image_path}: {e}")

            self.image_collection.delete(ids=image_ids)

            if self._bm25_images:
                new_ids = []
                new_corpus = []
                for idx, image_id in enumerate(self._bm25_images_ids):
                    if image_id not in image_ids:
                        new_ids.append(image_id)
                        new_corpus.append(self._bm25_images_corpus[idx])

                self._bm25_images_ids = new_ids
                self._bm25_images_corpus = new_corpus
                self._bm25_images = BM25Okapi(new_corpus) if new_corpus else None

        except Exception as e:
            logger.error(f"Error deleting images: {e}")
            raise

    # ========================================================================
    # КОНВЕРТАЦИЯ RST
    # ========================================================================

    async def _rst_to_markdown(self, rst_content: str) -> str:
        """Конвертация RST в Markdown."""
        try:
            markdown = pypandoc.convert_text(
                rst_content, "gfm", format="rst", extra_args=["--wrap=none"]
            )
            return markdown
        except Exception as e:
            logger.error(f"RST conversion error: {e}")
            return rst_content

    # ========================================================================
    # ОБРАБОТКА ТАБЛИЦ (БЕЗ TAG)
    # ========================================================================

    async def _process_parsed_tables(
        self, parsed_tables: List[Dict[str, Any]], file_id: str
    ) -> List[Dict[str, Any]]:
        """
        Обработка распарсенных таблиц с разбиением на row-level чанки.
        Поддержка сложных таблиц с мультистроками, мультистолбцами и большим текстом в ячейках.
        """
        all_table_chunks = []

        for table_idx, table_data in enumerate(parsed_tables):
            try:
                headers = table_data.get("headers", [])
                rows = table_data.get("rows", [])
                caption = table_data.get("caption", "")

                if not headers or not rows:
                    logger.warning(f"Table {table_idx} has no headers or rows, skipping")
                    continue

                # Рассчитываем оптимальный размер чанка
                chunk_size = self._calculate_optimal_chunk_size(headers, rows)
                overlap = max(2, chunk_size // 5)  # 20% overlap

                logger.debug(
                    f"Table {table_idx}: {len(rows)} rows, chunk_size={chunk_size}, overlap={overlap}"
                )

                # Создаем чанки по строкам
                for start_idx in range(0, len(rows), chunk_size - overlap):
                    end_idx = min(start_idx + chunk_size, len(rows))
                    chunk_rows = rows[start_idx:end_idx]

                    # Генерируем ID чанка
                    chunk_id = f"{file_id}_table_{table_idx}_rows_{start_idx}_{end_idx}"

                    # Создаем embedding text с полным контекстом
                    embedding_text = self._format_table_chunk_for_embedding(
                        caption, headers, chunk_rows, start_idx, end_idx
                    )

                    # Создаем Markdown для отображения
                    chunk_markdown = self._create_chunk_markdown(
                        caption, headers, chunk_rows, start_idx, end_idx, len(rows)
                    )

                    chunk_info = {
                        "chunk_id": chunk_id,
                        "table_id": f"{file_id}_table_{table_idx}",
                        "embedding_text": embedding_text,
                        "markdown": chunk_markdown,
                        "caption": caption,
                        "headers": headers,
                        "row_range": f"{start_idx}-{end_idx}",
                        "total_rows": len(rows),
                        "num_cols": len(headers),
                    }

                    all_table_chunks.append(chunk_info)

                    # Выходим, если достигли конца таблицы
                    if end_idx >= len(rows):
                        break

                logger.debug(
                    f"Table {table_idx} split into {len([c for c in all_table_chunks if c['table_id'] == f'{file_id}_table_{table_idx}'])} chunks"
                )

            except Exception as e:
                logger.warning(f"Failed to process table {table_idx}: {e}", exc_info=True)
                continue

        return all_table_chunks

    def _calculate_optimal_chunk_size(
        self, headers: List[str], sample_rows: List[List[str]]
    ) -> int:
        """
        Рассчитывает оптимальный размер чанка на основе ширины таблицы и размера ячеек.
        """
        num_cols = len(headers)

        # Оцениваем средний размер ячейки
        if sample_rows:
            total_cell_length = 0
            cell_count = 0
            for row in sample_rows[:10]:  # Анализируем первые 10 строк
                for cell in row:
                    total_cell_length += len(str(cell))
                    cell_count += 1

            avg_cell_length = total_cell_length / cell_count if cell_count > 0 else 20
        else:
            avg_cell_length = 20

        # Оцениваем токены на строку (примерно 1 токен на 4 символа)
        tokens_per_row = (avg_cell_length * num_cols) / 4

        # Целевой размер чанка: 400-512 токенов
        TARGET_TOKENS = 450

        # Рассчитываем количество строк
        rows_per_chunk = max(5, min(20, int(TARGET_TOKENS / tokens_per_row)))

        # Корректировки на основе ширины таблицы
        if num_cols <= 3:
            # Узкие таблицы - можно больше строк
            rows_per_chunk = min(20, rows_per_chunk * 2)
        elif num_cols > 10:
            # Широкие таблицы - меньше строк
            rows_per_chunk = max(3, rows_per_chunk // 2)

        # Корректировки на основе размера ячеек
        if avg_cell_length > 100:
            # Большой текст в ячейках (описания, примеры кода)
            rows_per_chunk = max(3, rows_per_chunk // 2)
            logger.debug(
                f"Large cell content detected (avg={avg_cell_length:.0f}), reducing chunk size to {rows_per_chunk}"
            )

        logger.debug(
            f"Calculated chunk size: {rows_per_chunk} rows (cols={num_cols}, avg_cell_len={avg_cell_length:.0f})"
        )

        return rows_per_chunk

    def _format_table_chunk_for_embedding(
        self,
        caption: str,
        headers: List[str],
        rows: List[List[str]],
        start_idx: int,
        end_idx: int,
    ) -> str:
        """
        Форматирует чанк таблицы для генерации embedding.
        Включает caption, заголовки и строки в структурированном виде.
        """
        parts = []

        # 1. Caption для контекста
        if caption:
            parts.append(f"Таблица: {caption}")

        # 2. Заголовки (критично для поиска по именам параметров)
        parts.append(f"Колонки: {' | '.join(headers)}")

        # 3. Строки в формате "Header: Value"
        for row_idx, row in enumerate(rows):
            row_parts = []
            for col_idx, cell_value in enumerate(row):
                if col_idx < len(headers):
                    header = headers[col_idx]
                    # Обрабатываем многострочный текст
                    clean_value = str(cell_value).replace("\n", " ").strip()
                    if clean_value:
                        row_parts.append(f"{header}: {clean_value}")

            if row_parts:
                parts.append(" | ".join(row_parts))

        # 4. Метаинформация для отладки
        parts.append(f"(строки {start_idx+1}-{end_idx})")

        return "\n".join(parts)

    def _create_chunk_markdown(
        self,
        caption: str,
        headers: List[str],
        rows: List[List[str]],
        start_idx: int,
        end_idx: int,
        total_rows: int,
    ) -> str:
        """
        Создает Markdown представление чанка таблицы для отображения в LLM.
        """
        md_lines = []

        # Caption с указанием диапазона строк
        if caption:
            md_lines.append(f"**{caption}** (строки {start_idx+1}-{end_idx} из {total_rows})\n")
        else:
            md_lines.append(f"**Таблица** (строки {start_idx+1}-{end_idx} из {total_rows})\n")

        # Заголовки
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Строки данных
        for row in rows:
            normalized_row = (row + [""] * len(headers))[: len(headers)]
            # Экранируем спецсимволы и заменяем переносы строк на пробелы
            escaped_row = [
                str(cell).replace("|", "\\|").replace("\n", " ").strip()
                for cell in normalized_row
            ]
            md_lines.append("| " + " | ".join(escaped_row) + " |")

        return "\n".join(md_lines)

    # ========================================================================
    # ОБРАБОТКА ИЗОБРАЖЕНИЙ
    # ========================================================================

    async def _extract_and_process_images(
        self, markdown_content: str, images_base_path: str, file_id: str
    ) -> List[Dict[str, Any]]:
        """Извлечение изображений из Markdown и их обработка."""
        images_info = []
        image_pattern = r"!\[([^\]]*)\]\(([^\)]+)\)"
        matches = re.findall(image_pattern, markdown_content)

        if not matches:
            return images_info

        for alt_text, image_path in matches:
            gc.collect()
            try:
                result = await self._process_single_image(
                    alt_text, image_path, images_base_path, file_id
                )
                images_info.append(result)
            except Exception as e:
                logger.warning(f"Image processing failed for {image_path}: {e}")
                continue

        return images_info

    async def _process_single_image(
        self, alt_text: str, image_path: str, images_base_path: str, file_id: str
    ) -> Dict[str, Any]:
        """Обработка одного изображения."""
        try:
            clean_path = image_path.lstrip("/")
            full_path = Path(images_base_path) / clean_path

            if not full_path.exists():
                raise FileNotFoundError(f"Image not found: {full_path}")

            image = PILImage.open(full_path).convert("RGB")

            image_id = f"{file_id}_{uuid.uuid4().hex}"
            storage_path = Path(self.config.images_storage_path) / f"{image_id}.jpg"
            image.save(storage_path, format="JPEG", quality=85)

            loop = asyncio.get_event_loop()
            analysis_result = await loop.run_in_executor(
                None, self._run_vlm_structured_sync, image
            )

            return {
                "image_id": image_id,
                "image_path": str(storage_path),
                "alt_text": alt_text,
                "ocr_text": analysis_result.ocr_text,
                "vlm_description": analysis_result.ui_description,
                "original_path": image_path,
            }

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    def _run_vlm_structured_sync(self, image: PILImage.Image) -> ImageAnalysisResult:
        """VLM запрос с structured output."""
        try:
            max_dim = 1280
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim), PILImage.Resampling.LANCZOS)

            if image.width < 32 or image.height < 32:
                scale = 32.0 / min(image.width, image.height)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, PILImage.Resampling.LANCZOS)

            img_buffer = BytesIO()
            image.save(img_buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            response: ollama.ChatResponse = self.ollama_client.chat(
                model=self.config.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": UNIFIED_VLM_PROMPT,
                        "images": [img_base64],
                    }
                ],
                stream=False,
                format=ImageAnalysisResult.model_json_schema(),
                options={
                    "temperature": 0.9,
                    "num_ctx": 4096,
                    "num_predict": 400,
                },
                keep_alive="10s",
            )

            result = ImageAnalysisResult.model_validate_json(
                json_repair.repair_json(response.message.content, return_objects=False)
            )

            return result

        except Exception as e:
            logger.warning(f"VLM processing failed: {e}")
            return ImageAnalysisResult(ocr_text="", ui_description="")

    # ========================================================================
    # РАЗБИЕНИЕ ТЕКСТА НА ЧАНКИ
    # ========================================================================

    async def _split_text_with_context(
        self, markdown_content: str, file_id: str, document_url: str
    ) -> List[Dict[str, Any]]:
        """Разбиение текста на чанки."""
        splitter = MarkdownTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        chunks = splitter.split_text(markdown_content)

        # Извлекаем заголовки секций
        section_headers = self._extract_section_headers(markdown_content)

        chunks_with_metadata = []
        for idx, chunk_text in enumerate(chunks):
            # Находим ближайший заголовок
            section_header = self._find_nearest_header(
                chunk_text, section_headers, markdown_content
            )

            chunks_with_metadata.append(
                {
                    "chunk_id": f"{file_id}_chunk_{idx}",
                    "text": chunk_text,
                    "section_header": section_header,
                    "chunk_index": idx,
                    "file_id": file_id,
                    "document_url": document_url,
                }
            )

        return chunks_with_metadata

    def _extract_section_headers(self, markdown_content: str) -> List[Tuple[str, int]]:
        """Извлечение заголовков секций с их позициями."""
        headers = []
        header_pattern = r"^(#{1,6})\s+(.+)$"

        for match in re.finditer(header_pattern, markdown_content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            position = match.start()
            headers.append((title, position, level))

        return headers

    def _find_nearest_header(
        self,
        chunk_text: str,
        section_headers: List[Tuple[str, int]],
        full_content: str,
    ) -> str:
        """Находит ближайший заголовок для чанка."""
        chunk_position = full_content.find(chunk_text)
        if chunk_position == -1:
            return ""

        nearest_header = ""
        for header, pos, level in reversed(section_headers):
            if pos < chunk_position:
                nearest_header = header
                break

        return nearest_header

    # ========================================================================
    # ИНДЕКСИРОВАНИЕ
    # ========================================================================

    async def _embed_and_store_text_chunks(self, chunks: List[Dict[str, Any]]):
        """Эмбеддинг и сохранение текстовых чанков."""
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "file_id": chunk["file_id"],
                "document_url": chunk["document_url"],
                "section_header": chunk.get("section_header", ""),
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]

        try:
            embeddings = []
            batch_size = 10
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                batch_embeddings = await self._generate_embeddings(batch)
                embeddings.extend(batch_embeddings)

            self.text_collection.add(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )

            logger.info(f"Stored {len(chunks)} text chunks in vector DB")

        except Exception as e:
            logger.error(f"Error storing text chunks: {e}")
            raise

    async def _embed_and_store_images(
        self, images_info: List[Dict[str, Any]], file_id: str, document_url: str
    ):
        """Эмбеддинг и сохранение изображений."""
        if not images_info:
            return

        ids = [img["image_id"] for img in images_info]
        documents = [
            f"{img['alt_text']} {img['ocr_text']} {img['vlm_description']}"
            for img in images_info
        ]
        metadatas = [
            {
                "file_id": file_id,
                "document_url": document_url,
                "image_path": img["image_path"],
                "alt_text": img["alt_text"],
                "ocr_text": img["ocr_text"],
                "vlm_description": img["vlm_description"],
                "original_path": img["original_path"],
            }
            for img in images_info
        ]

        try:
            embeddings = await self._generate_embeddings(documents)

            self.image_collection.add(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )

            logger.info(f"Stored {len(images_info)} images in vector DB")

        except Exception as e:
            logger.error(f"Error storing images: {e}")
            raise

    async def _embed_and_store_tables(
        self, tables_chunks: List[Dict[str, Any]], file_id: str, document_url: str
    ):
        """
        Эмбеддинг и сохранение табличных чанков.
        Каждый чанк индексируется отдельно для точного поиска по содержимому.
        """
        if not tables_chunks:
            return

        ids = [chunk["chunk_id"] for chunk in tables_chunks]
        documents = [chunk["embedding_text"] for chunk in tables_chunks]
        metadatas = [
            {
                "file_id": file_id,
                "document_url": document_url,
                "table_id": chunk["table_id"],
                "caption": chunk.get("caption", "") or "",
                "row_range": chunk["row_range"],
                "total_rows": chunk["total_rows"],
                "num_cols": chunk["num_cols"],
                "markdown": chunk["markdown"],
                "headers": json.dumps(chunk["headers"], ensure_ascii=False),
            }
            for chunk in tables_chunks
        ]

        try:
            embeddings = []
            batch_size = 10
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                batch_embeddings = await self._generate_embeddings(batch)
                embeddings.extend(batch_embeddings)

            self.table_collection.add(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )

            logger.info(f"Stored {len(tables_chunks)} table chunks in vector DB")

        except Exception as e:
            logger.error(f"Error storing tables: {e}")
            raise

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов через Ollama."""
        embeddings = []
        for text in texts:
            try:
                max_length = 8000
                truncated_text = text[:max_length] if len(text) > max_length else text

                response = self.ollama_client.embeddings(
                    model=self.config.text_embedding_model, prompt=truncated_text
                )
                embeddings.append(response["embedding"])

            except Exception as e:
                logger.error(f"Embedding generation error: {e}")
                embeddings.append([0.0] * 768)

        return embeddings

    # ========================================================================
    # ОБНОВЛЕНИЕ BM25
    # ========================================================================

    async def _update_bm25_indices(
        self,
        text_chunks: List[Dict[str, Any]],
        tables_chunks: List[Dict[str, Any]],
        images_info: List[Dict[str, Any]],
    ):
        """Обновление BM25 индексов."""
        # Текстовые чанки
        for chunk in text_chunks:
            self._bm25_text_ids.append(chunk["chunk_id"])
            self._bm25_text_corpus.append(chunk["text"].lower().split())

        if self._bm25_text_corpus:
            self._bm25_text = BM25Okapi(self._bm25_text_corpus)

        # Табличные чанки - индексируем по embedding_text
        for chunk in tables_chunks:
            self._bm25_tables_ids.append(chunk["chunk_id"])
            # Используем embedding_text для BM25 (содержит caption + headers + rows)
            self._bm25_tables_corpus.append(chunk["embedding_text"].lower().split())

        if self._bm25_tables_corpus:
            self._bm25_tables = BM25Okapi(self._bm25_tables_corpus)

        # Изображения
        for img in images_info:
            self._bm25_images_ids.append(img["image_id"])
            text = f"{img.get('alt_text', '')} {img.get('ocr_text', '')}".lower()
            self._bm25_images_corpus.append(text.split())

        if self._bm25_images_corpus:
            self._bm25_images = BM25Okapi(self._bm25_images_corpus)

        logger.debug("BM25 indices updated")

    # ========================================================================
    # ОТКАТ ПРИ ОШИБКЕ
    # ========================================================================

    async def _rollback_document(self, file_id: str):
        """Откат изменений при ошибке добавления документа."""
        logger.warning(f"Rolling back document {file_id}")
        try:
            try:
                self.text_collection.delete(where={"file_id": file_id})
            except:
                pass

            try:
                self.table_collection.delete(where={"file_id": file_id})
            except:
                pass

            try:
                self.image_collection.delete(where={"file_id": file_id})
            except:
                pass

        except Exception as e:
            logger.error(f"Rollback failed for {file_id}: {e}")

    # ========================================================================
    # ПОИСК
    # ========================================================================

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        search_images: bool = True,
        search_tables: bool = True,
    ) -> List[MultimodalChunk]:
        """Гибридный поиск с нормализацией скоров."""
        all_results = []

        # Текстовый поиск
        text_results = await self._hybrid_search_text(query, top_k)
        all_results.extend(text_results)

        # Поиск по таблицам
        if search_tables:
            table_results = await self._hybrid_search_tables(query, top_k)
            all_results.extend(table_results)

        # Поиск по изображениям
        if search_images:
            image_results = await self._hybrid_search_images(query, top_k)
            all_results.extend(image_results)

        # Нормализация скоров
        all_results = self._normalize_scores_minmax(all_results)

        # Сортировка и ограничение
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:top_k]

    async def _hybrid_search_text(
        self, query: str, top_k: int
    ) -> List[MultimodalChunk]:
        """Гибридный поиск по текстовым чанкам."""
        results = []

        # Vector search
        try:
            query_embedding = await self._generate_embeddings([query])
            vector_results = self.text_collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
            )

            for i in range(len(vector_results["ids"][0])):
                chunk = MultimodalChunk(
                    chunk_id=vector_results["ids"][0][i],
                    type="text",
                    score=1.0 - vector_results["distances"][0][i],
                    document_url=vector_results["metadatas"][0][i]["document_url"],
                    file_id=vector_results["metadatas"][0][i]["file_id"],
                    content=vector_results["documents"][0][i],
                    section_header=vector_results["metadatas"][0][i].get(
                        "section_header"
                    ),
                    chunk_index=vector_results["metadatas"][0][i].get("chunk_index"),
                )
                results.append(chunk)

        except Exception as e:
            logger.error(f"Vector search error: {e}")

        # BM25 search
        if self._bm25_text:
            try:
                tokenized_query = query.lower().split()
                bm25_scores = self._bm25_text.get_scores(tokenized_query)
                top_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[:top_k]

                for idx in top_indices:
                    if bm25_scores[idx] > 0:
                        chunk_id = self._bm25_text_ids[idx]
                        chunk_data = self.text_collection.get(ids=[chunk_id])

                        if chunk_data["ids"]:
                            chunk = MultimodalChunk(
                                chunk_id=chunk_id,
                                type="text",
                                score=float(bm25_scores[idx]),
                                document_url=chunk_data["metadatas"][0]["document_url"],
                                file_id=chunk_data["metadatas"][0]["file_id"],
                                content=chunk_data["documents"][0],
                                section_header=chunk_data["metadatas"][0].get(
                                    "section_header"
                                ),
                                chunk_index=chunk_data["metadatas"][0].get(
                                    "chunk_index"
                                ),
                            )
                            results.append(chunk)

            except Exception as e:
                logger.error(f"BM25 search error: {e}")

        return results

    async def _hybrid_search_tables(
        self, query: str, top_k: int
    ) -> List[MultimodalChunk]:
        """Гибридный поиск по табличным чанкам."""
        results = []

        # Vector search
        try:
            query_embedding = await self._generate_embeddings([query])
            vector_results = self.table_collection.query(
                query_embeddings=query_embedding,
                n_results=top_k * 2,  # Берем больше, т.к. могут быть чанки одной таблицы
            )

            for i in range(len(vector_results["ids"][0])):
                metadata = vector_results["metadatas"][0][i]

                chunk = MultimodalChunk(
                    chunk_id=vector_results["ids"][0][i],
                    type="table",
                    score=1.0 - vector_results["distances"][0][i],
                    document_url=metadata["document_url"],
                    file_id=metadata["file_id"],
                    table_metadata={
                        "table_id": metadata.get("table_id"),
                        "caption": metadata.get("caption", ""),
                        "row_range": metadata.get("row_range"),
                        "total_rows": metadata.get("total_rows", 0),
                        "num_cols": metadata.get("num_cols", 0),
                    },
                    content=metadata.get("markdown", ""),
                )
                results.append(chunk)

        except Exception as e:
            logger.error(f"Table vector search error: {e}")

        # BM25 search
        if self._bm25_tables:
            try:
                tokenized_query = query.lower().split()
                bm25_scores = self._bm25_tables.get_scores(tokenized_query)
                top_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[:top_k * 2]

                for idx in top_indices:
                    if bm25_scores[idx] > 0:
                        chunk_id = self._bm25_tables_ids[idx]
                        table_data = self.table_collection.get(ids=[chunk_id])

                        if table_data["ids"]:
                            metadata = table_data["metadatas"][0]

                            chunk = MultimodalChunk(
                                chunk_id=chunk_id,
                                type="table",
                                score=float(bm25_scores[idx]),
                                document_url=metadata["document_url"],
                                file_id=metadata["file_id"],
                                table_metadata={
                                    "table_id": metadata.get("table_id"),
                                    "caption": metadata.get("caption", ""),
                                    "row_range": metadata.get("row_range"),
                                    "total_rows": metadata.get("total_rows", 0),
                                    "num_cols": metadata.get("num_cols", 0),
                                },
                                content=metadata.get("markdown", ""),
                            )
                            results.append(chunk)

            except Exception as e:
                logger.error(f"Table BM25 search error: {e}")

        return results

    async def _hybrid_search_images(
        self, query: str, top_k: int
    ) -> List[MultimodalChunk]:
        """Гибридный поиск по изображениям."""
        results = []

        # Vector search
        try:
            query_embedding = await self._generate_embeddings([query])
            vector_results = self.image_collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
            )

            for i in range(len(vector_results["ids"][0])):
                metadata = vector_results["metadatas"][0][i]

                chunk = MultimodalChunk(
                    chunk_id=vector_results["ids"][0][i],
                    type="image",
                    score=1.0 - vector_results["distances"][0][i],
                    document_url=metadata["document_url"],
                    file_id=metadata["file_id"],
                    image_path=metadata.get("image_path"),
                    alt_text=metadata.get("alt_text", ""),
                    ocr_text=metadata.get("ocr_text", ""),
                    vlm_description=metadata.get("vlm_description", ""),
                )
                results.append(chunk)

        except Exception as e:
            logger.error(f"Image vector search error: {e}")

        # BM25 search
        if self._bm25_images:
            try:
                tokenized_query = query.lower().split()
                bm25_scores = self._bm25_images.get_scores(tokenized_query)
                top_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[:top_k]

                for idx in top_indices:
                    if bm25_scores[idx] > 0:
                        image_id = self._bm25_images_ids[idx]
                        image_data = self.image_collection.get(ids=[image_id])

                        if image_data["ids"]:
                            metadata = image_data["metadatas"][0]

                            chunk = MultimodalChunk(
                                chunk_id=image_id,
                                type="image",
                                score=float(bm25_scores[idx]),
                                document_url=metadata["document_url"],
                                file_id=metadata["file_id"],
                                image_path=metadata.get("image_path"),
                                alt_text=metadata.get("alt_text", ""),
                                ocr_text=metadata.get("ocr_text", ""),
                                vlm_description=metadata.get("vlm_description", ""),
                            )
                            results.append(chunk)

            except Exception as e:
                logger.error(f"Image BM25 search error: {e}")

        return results

    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================================

    def get_document_stats(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Получение статистики по документу."""
        return self.documents_metadata.get(file_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Список всех проиндексированных документов."""
        return [
            {"file_id": file_id, **metadata}
            for file_id, metadata in self.documents_metadata.items()
        ]
