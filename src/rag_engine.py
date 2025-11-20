""" Мультимодальный RAG Engine v3.0 с поддержкой удаления документов """

import asyncio
import base64
import gc
import hashlib
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import httpx
import json_repair
import ollama
import pypandoc
from bs4 import BeautifulSoup
from chromadb.config import Settings
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
    table_structure: Optional[Dict[str, Any]] = None
    table_content: Optional[List[Dict[str, Any]]] = None
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    alt_text: Optional[str] = None
    ocr_text: Optional[str] = None
    vlm_description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class HybridSearchConfig:
    """Конфигурация гибридного поиска."""

    weight_semantic: float = 0.7
    weight_lexical: float = 0.3
    top_k_retrieve: int = 100
    top_k_rerank: int = 50
    final_k: int = 10
    rrf_k: int = 60


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

        self.documents_metadata = {}
        self.hybrid_config = HybridSearchConfig()

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
                name="tables", metadata={"description": "Tables with TAG structure"}
            )

            logger.info("Vector DB initialized with 3 collections")

            # Ollama client
            self.ollama_client = ollama.Client(
                self.config.ollama_base_url,
                timeout=httpx.Timeout(
                    connect=10.0, read=300.0, write=10.0, pool=10.0
                ),
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
                for meta in table_results["metadatas"]:
                    text = (
                        f"{meta.get('summary', '')} {meta.get('context', '')}".lower()
                    )
                    self._bm25_tables_corpus.append(text.split())
                self._bm25_tables = BM25Okapi(self._bm25_tables_corpus)
                logger.info(
                    f"BM25 table index initialized with {len(self._bm25_tables_ids)} tables"
                )

        except Exception as e:
            logger.warning(f"Could not initialize BM25 indices: {e}")

    async def cleanup(self):
        """Очистка ресурсов."""
        self._bm25_text = None
        self._bm25_tables = None
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
    # ДОБАВЛЕНИЕ ДОКУМЕНТА
    # ========================================================================

    async def add_document(
        self, rst_content: str, document_url: str, images_base_path: str
    ) -> str:
        """Добавление документа с полной мультимодальной обработкой."""
        file_id = str(uuid.uuid4())
        logger.info(f"Processing document {file_id}: {document_url}")

        try:
            # 1. Конвертация RST → Markdown
            markdown_content = await self._rst_to_markdown(rst_content)
            logger.debug(f"Converted to Markdown ({len(markdown_content)} chars)")

            # 2. Извлечение и обработка изображений (OCR + VLM одним запросом)
            images_info = await self._extract_and_process_images(
                markdown_content, images_base_path, file_id
            )
            logger.info(f"Processed {len(images_info)} images")

            # 3. Извлечение и обработка таблиц
            (
                markdown_without_tables,
                tables_info,
            ) = await self._extract_and_process_tables(markdown_content, file_id)
            logger.info(f"Extracted {len(tables_info)} tables")

            # 4. Разбиение текста на чанки
            text_chunks = await self._split_text_with_context(
                markdown_without_tables, file_id, document_url
            )
            logger.info(f"Created {len(text_chunks)} text chunks")

            # 5. Индексирование
            await self._embed_and_store_text_chunks(text_chunks)
            await self._embed_and_store_images(images_info, file_id, document_url)
            await self._embed_and_store_tables(tables_info, file_id, document_url)

            # 6. Обновление BM25
            await self._update_bm25_indices(text_chunks, tables_info)

            # 7. Сохранение метаданных
            self.documents_metadata[file_id] = {
                "document_url": document_url,
                "chunks_count": len(text_chunks),
                "images_count": len(images_info),
                "tables_count": len(tables_info),
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
        """
        Полное удаление документа из системы.
        
        Удаляет:
        - Все текстовые чанки из векторной БД и BM25
        - Все таблицы из векторной БД и BM25
        - Все изображения из векторной БД
        - Все файлы изображений из storage
        - Метаданные документа
        
        Args:
            file_id: Идентификатор документа для удаления
            
        Returns:
            bool: True если документ успешно удален, False если документ не найден
        """
        if file_id not in self.documents_metadata:
            logger.warning(f"Document {file_id} not found in metadata")
            return False

        logger.info(f"Starting deletion of document {file_id}")

        try:
            # 1. Удаление текстовых чанков из ChromaDB и BM25
            await self._delete_text_chunks(file_id)

            # 2. Удаление таблиц из ChromaDB и BM25
            await self._delete_tables(file_id)

            # 3. Удаление изображений (файлы + ChromaDB)
            await self._delete_images(file_id)

            # 4. Удаление метаданных документа
            del self.documents_metadata[file_id]

            logger.info(f"Document {file_id} successfully deleted")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {file_id}: {e}", exc_info=True)
            raise

    async def _delete_text_chunks(self, file_id: str):
        """Удаление текстовых чанков документа из ChromaDB и BM25."""
        try:
            # Получить все ID чанков документа
            results = self.text_collection.get(where={"file_id": file_id})

            if not results["ids"]:
                logger.debug(f"No text chunks found for document {file_id}")
                return

            chunk_ids = results["ids"]
            logger.info(f"Deleting {len(chunk_ids)} text chunks for document {file_id}")

            # Удалить из ChromaDB
            self.text_collection.delete(ids=chunk_ids)

            # Удалить из BM25 индекса
            if self._bm25_text:
                # Создаем новые списки без удаляемых чанков
                new_ids = []
                new_corpus = []

                for idx, doc_id in enumerate(self._bm25_text_ids):
                    if doc_id not in chunk_ids:
                        new_ids.append(doc_id)
                        new_corpus.append(self._bm25_text_corpus[idx])

                self._bm25_text_ids = new_ids
                self._bm25_text_corpus = new_corpus

                # Пересоздаем BM25 индекс
                if self._bm25_text_corpus:
                    self._bm25_text = BM25Okapi(self._bm25_text_corpus)
                else:
                    self._bm25_text = None

                logger.debug(f"Removed {len(chunk_ids)} chunks from BM25 text index")

        except Exception as e:
            logger.error(f"Error deleting text chunks for {file_id}: {e}")
            raise

    async def _delete_tables(self, file_id: str):
        """Удаление таблиц документа из ChromaDB и BM25."""
        try:
            # Получить все ID таблиц документа
            results = self.table_collection.get(where={"file_id": file_id})

            if not results["ids"]:
                logger.debug(f"No tables found for document {file_id}")
                return

            table_ids = results["ids"]
            logger.info(f"Deleting {len(table_ids)} tables for document {file_id}")

            # Удалить из ChromaDB
            self.table_collection.delete(ids=table_ids)

            # Удалить из BM25 индекса
            if self._bm25_tables:
                # Создаем новые списки без удаляемых таблиц
                new_ids = []
                new_corpus = []

                for idx, table_id in enumerate(self._bm25_tables_ids):
                    if table_id not in table_ids:
                        new_ids.append(table_id)
                        new_corpus.append(self._bm25_tables_corpus[idx])

                self._bm25_tables_ids = new_ids
                self._bm25_tables_corpus = new_corpus

                # Пересоздаем BM25 индекс
                if self._bm25_tables_corpus:
                    self._bm25_tables = BM25Okapi(self._bm25_tables_corpus)
                else:
                    self._bm25_tables = None

                logger.debug(f"Removed {len(table_ids)} tables from BM25 table index")

        except Exception as e:
            logger.error(f"Error deleting tables for {file_id}: {e}")
            raise

    async def _delete_images(self, file_id: str):
        """Удаление изображений документа: файлы из storage и записи из ChromaDB."""
        try:
            # Получить все изображения документа
            results = self.image_collection.get(where={"file_id": file_id})

            if not results["ids"]:
                logger.debug(f"No images found for document {file_id}")
                return

            image_ids = results["ids"]
            metadatas = results["metadatas"]

            logger.info(f"Deleting {len(image_ids)} images for document {file_id}")

            # Удалить файлы изображений из storage
            deleted_files = 0
            for metadata in metadatas:
                image_path = metadata.get("image_path")
                if image_path:
                    try:
                        path = Path(image_path)
                        if path.exists():
                            path.unlink()
                            deleted_files += 1
                            logger.debug(f"Deleted image file: {image_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete image file {image_path}: {e}")

            logger.info(f"Deleted {deleted_files} image files from storage")

            # Удалить из ChromaDB
            self.image_collection.delete(ids=image_ids)

            logger.debug(f"Removed {len(image_ids)} images from ChromaDB")

        except Exception as e:
            logger.error(f"Error deleting images for {file_id}: {e}")
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
    # ОБРАБОТКА ИЗОБРАЖЕНИЙ (VLM + Structured Output)
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

        # Последовательная обработка для экономии памяти
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
        """Обработка одного изображения: единый запрос VLM с structured output."""
        try:
            # Корректировка пути
            clean_path = image_path.lstrip("/")
            full_path = Path(images_base_path) / clean_path

            if not full_path.exists():
                raise FileNotFoundError(f"Image not found: {full_path}")

            image = PILImage.open(full_path).convert("RGB")
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            image_id = f"{file_id}_{image_hash}"

            storage_path = Path(self.config.images_storage_path) / f"{image_id}.jpg"
            image.save(storage_path, format="JPEG", quality=85)

            # Единый запрос к VLM с structured output
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
        """Единый запрос к VLM с structured output через JSON schema."""
        try:
            # Resize для защиты от OOM
            max_dim = 1280
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim), PILImage.Resampling.LANCZOS)

            # Гарантировать минимальный размер 32x32 для vision-моделей
            if image.width < 32 or image.height < 32:
                # Увеличиваем до 32x32 с сохранением пропорций
                scale = 32.0 / min(image.width, image.height)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, PILImage.Resampling.LANCZOS)
                logger.warning(
                    f"Image upscaled to {new_size} due to vision model minimum size requirement"
                )

            # Конвертация в base64
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

        except httpx.HTTPStatusError as e:
            logger.error(f"VLM HTTP error {e.response.status_code}: {e}")
            return ImageAnalysisResult(ocr_text="", ui_description="")
        except json.JSONDecodeError as e:
            logger.error(f"VLM returned invalid JSON: {e}")
            return ImageAnalysisResult(ocr_text="", ui_description="")
        except Exception as e:
            logger.warning(f"VLM processing failed: {e}")
            return ImageAnalysisResult(ocr_text="", ui_description="")

    # ========================================================================
    # ОБРАБОТКА ТАБЛИЦ (TAG)
    # ========================================================================

    async def _extract_and_process_tables(
        self, markdown_content: str, file_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Извлечение таблиц с TAG структурой."""
        tables_info = []
        content_without_tables = markdown_content

        # Паттерны для таблиц
        grid_table_pattern = r"(\n\+[-+]+\+\n(?:\|[^\n]*\|\n)+\+[-+]+\+)"
        pipe_table_pattern = (
            r"(\n\|[^\n]+\|\n\|[-\s|:]+\|\n(?:\|[^\n]+\|\n)+)"
        )
        html_table_pattern = r"(<table[^>]*>.*?</table>)"

        all_patterns = [
            ("grid", grid_table_pattern),
            ("pipe", pipe_table_pattern),
            ("html", html_table_pattern),
        ]

        for table_type, pattern in all_patterns:
            matches = re.finditer(
                pattern, markdown_content, re.DOTALL | re.IGNORECASE
            )

            for idx, match in enumerate(matches):
                table_text = match.group(0)
                table_id = f"{file_id}_table_{len(tables_info)}"

                try:
                    if table_type == "html":
                        table_data = self._parse_html_table(table_text)
                    else:
                        table_data = self._parse_markdown_table(table_text)

                    if table_data:
                        tables_info.append(
                            {
                                "table_id": table_id,
                                "table_type": table_type,
                                "content": table_data,
                                "raw_text": table_text,
                            }
                        )

                        # Заменяем таблицу на placeholder
                        placeholder = f"[TABLE_{len(tables_info) - 1}]"
                        content_without_tables = content_without_tables.replace(
                            table_text, placeholder, 1
                        )

                except Exception as e:
                    logger.warning(f"Failed to parse table: {e}")

        return content_without_tables, tables_info

    def _parse_markdown_table(self, table_text: str) -> Optional[List[Dict[str, Any]]]:
        """Парсинг Markdown таблицы."""
        try:
            lines = [line.strip() for line in table_text.split("\n") if line.strip()]

            header_line = None
            data_lines = []

            for line in lines:
                if "|" in line and not line.strip().startswith("+"):
                    if "---" in line or ":-" in line:
                        continue
                    if header_line is None:
                        header_line = line
                    else:
                        data_lines.append(line)

            if not header_line:
                return None

            headers = [h.strip() for h in header_line.split("|") if h.strip()]
            rows = []

            for line in data_lines:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) == len(headers):
                    row = {headers[i]: cells[i] for i in range(len(headers))}
                    rows.append(row)

            return rows if rows else None

        except Exception as e:
            logger.debug(f"Markdown table parsing error: {e}")
            return None

    def _parse_html_table(self, table_html: str) -> Optional[List[Dict[str, Any]]]:
        """Парсинг HTML таблицы."""
        try:
            soup = BeautifulSoup(table_html, "html.parser")
            table = soup.find("table")

            if not table:
                return None

            # Извлечение заголовков
            headers = []
            thead = table.find("thead")
            if thead:
                header_row = thead.find("tr")
                if header_row:
                    headers = [
                        th.get_text(strip=True) for th in header_row.find_all(["th", "td"])
                    ]

            if not headers:
                first_row = table.find("tr")
                if first_row:
                    headers = [
                        cell.get_text(strip=True)
                        for cell in first_row.find_all(["th", "td"])
                    ]

            # Извлечение данных
            rows = []
            tbody = table.find("tbody") or table

            for row in tbody.find_all("tr")[1 if not thead else 0:]:
                cells = [
                    cell.get_text(strip=True) for cell in row.find_all(["td", "th"])
                ]
                if len(cells) == len(headers):
                    row_dict = {headers[i]: cells[i] for i in range(len(headers))}
                    rows.append(row_dict)

            return rows if rows else None

        except Exception as e:
            logger.debug(f"HTML table parsing error: {e}")
            return None

    # ========================================================================
    # РАЗБИЕНИЕ ТЕКСТА
    # ========================================================================

    async def _split_text_with_context(
        self, markdown_content: str, file_id: str, document_url: str
    ) -> List[Dict[str, Any]]:
        """Разбиение текста на чанки с контекстом."""
        try:
            splitter = MarkdownTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            chunks = splitter.split_text(markdown_content)

            result_chunks = []
            current_header = ""

            for idx, chunk in enumerate(chunks):
                # Поиск заголовка в чанке
                header_match = re.search(r"^#{1,6}\s+(.+)", chunk, re.MULTILINE)
                if header_match:
                    current_header = header_match.group(1).strip()

                chunk_id = f"{file_id}_chunk_{idx}"
                result_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "content": chunk,
                        "section_header": current_header,
                        "chunk_index": idx,
                        "file_id": file_id,
                        "document_url": document_url,
                    }
                )

            return result_chunks

        except Exception as e:
            logger.error(f"Text splitting error: {e}")
            return []

    # ========================================================================
    # ИНДЕКСИРОВАНИЕ
    # ========================================================================

    async def _embed_and_store_text_chunks(self, chunks: List[Dict[str, Any]]):
        """Сохранение текстовых чанков в ChromaDB."""
        if not chunks:
            return

        try:
            chunk_ids = [c["chunk_id"] for c in chunks]
            documents = [c["content"] for c in chunks]
            metadatas = [
                {
                    "file_id": c["file_id"],
                    "document_url": c["document_url"],
                    "section_header": c["section_header"],
                    "chunk_index": c["chunk_index"],
                }
                for c in chunks
            ]

            # Получение эмбеддингов
            embeddings = await self._get_text_embeddings(documents)

            # Сохранение в ChromaDB
            self.text_collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        except Exception as e:
            logger.error(f"Error storing text chunks: {e}")
            raise

    async def _embed_and_store_images(
        self, images_info: List[Dict[str, Any]], file_id: str, document_url: str
    ):
        """Сохранение изображений в ChromaDB."""
        if not images_info:
            return

        try:
            for img_info in images_info:
                # Создание текстового представления для эмбеддинга
                text_repr = (
                    f"{img_info['alt_text']} {img_info['ocr_text']} {img_info['vlm_description']}"
                )

                # Получение эмбеддинга
                embedding = await self._get_text_embeddings([text_repr])

                # Сохранение в ChromaDB
                self.image_collection.add(
                    ids=[img_info["image_id"]],
                    embeddings=embedding,
                    documents=[text_repr],
                    metadatas=[
                        {
                            "file_id": file_id,
                            "document_url": document_url,
                            "image_path": img_info["image_path"],
                            "alt_text": img_info["alt_text"],
                            "ocr_text": img_info["ocr_text"],
                            "vlm_description": img_info["vlm_description"],
                        }
                    ],
                )

        except Exception as e:
            logger.error(f"Error storing images: {e}")
            raise

    async def _embed_and_store_tables(
        self, tables_info: List[Dict[str, Any]], file_id: str, document_url: str
    ):
        """Сохранение таблиц в ChromaDB."""
        if not tables_info:
            return

        try:
            for table_info in tables_info:
                # Создание текстового представления (TAG)
                table_text = json.dumps(table_info["content"], ensure_ascii=False)

                # Получение эмбеддинга
                embedding = await self._get_text_embeddings([table_text])

                # Сохранение в ChromaDB
                self.table_collection.add(
                    ids=[table_info["table_id"]],
                    embeddings=embedding,
                    documents=[table_text],
                    metadatas=[
                        {
                            "file_id": file_id,
                            "document_url": document_url,
                            "table_type": table_info["table_type"],
                            "raw_text": table_info["raw_text"],
                        }
                    ],
                )

        except Exception as e:
            logger.error(f"Error storing tables: {e}")
            raise

    # ========================================================================
    # ОБНОВЛЕНИЕ BM25
    # ========================================================================

    async def _update_bm25_indices(
        self, text_chunks: List[Dict[str, Any]], tables_info: List[Dict[str, Any]]
    ):
        """Обновление BM25 индексов."""
        try:
            # Текстовые чанки
            if text_chunks:
                for chunk in text_chunks:
                    self._bm25_text_ids.append(chunk["chunk_id"])
                    self._bm25_text_corpus.append(chunk["content"].lower().split())

                self._bm25_text = BM25Okapi(self._bm25_text_corpus)

            # Таблицы
            if tables_info:
                for table in tables_info:
                    self._bm25_tables_ids.append(table["table_id"])
                    table_text = json.dumps(
                        table["content"], ensure_ascii=False
                    ).lower()
                    self._bm25_tables_corpus.append(table_text.split())

                self._bm25_tables = BM25Okapi(self._bm25_tables_corpus)

        except Exception as e:
            logger.warning(f"BM25 update error: {e}")

    # ========================================================================
    # ОТКАТ (ROLLBACK)
    # ========================================================================

    async def _rollback_document(self, file_id: str):
        """Откат при ошибке добавления документа."""
        try:
            logger.info(f"Rolling back document {file_id}")
            
            # Удаляем из всех коллекций
            for collection in [
                self.text_collection,
                self.image_collection,
                self.table_collection,
            ]:
                try:
                    results = collection.get(where={"file_id": file_id})
                    if results["ids"]:
                        collection.delete(ids=results["ids"])
                except Exception:
                    pass

            # Удаляем файлы изображений
            try:
                results = self.image_collection.get(where={"file_id": file_id})
                for metadata in results.get("metadatas", []):
                    image_path = metadata.get("image_path")
                    if image_path:
                        try:
                            Path(image_path).unlink(missing_ok=True)
                        except Exception:
                            pass
            except Exception:
                pass

            # Удаляем метаданные
            if file_id in self.documents_metadata:
                del self.documents_metadata[file_id]

            logger.info(f"Rolled back document {file_id}")

        except Exception as e:
            logger.error(f"Rollback error: {e}")

    # ========================================================================
    # ЭМБЕДДИНГИ
    # ========================================================================

    async def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов через Ollama."""
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self._get_embeddings_sync, texts
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _get_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Синхронное получение эмбеддингов."""
        try:
            response: ollama.EmbedResponse = self.ollama_client.embed(
                model=self.config.text_embedding_model,
                input=texts,
                keep_alive="10s",
            )

            # Fallback на пустые эмбеддинги
            if not response.embeddings or len(response.embeddings) != len(texts):
                return [[0.0] * 768 for _ in texts]

            return response.embeddings

        except Exception as e:
            logger.warning(f"Embedding failed for text: {e}")
            return [[0.0] * 768 for _ in texts]

    # ========================================================================
    # МУЛЬТИМОДАЛЬНЫЙ ПОИСК
    # ========================================================================

    async def hybrid_search(self, query: str, top_k: int = 10) -> List[MultimodalChunk]:
        """
        Мультимодальный гибридный поиск по ВСЕМ модальностям.
        
        Выполняет параллельный поиск по:
        - Текстовым чанкам (BM25 + векторный поиск)
        - Изображениям (векторный поиск по OCR + VLM описаниям)
        - Таблицам (BM25 + векторный поиск по TAG структуре)
        
        Args:
            query: Поисковый запрос
            top_k: Количество лучших результатов для возврата
            
        Returns:
            List[MultimodalChunk]: Отсортированный список результатов из всех модальностей
        """
        try:
            # Параллельный поиск по всем коллекциям
            text_results_task = self._search_text_chunks(query, top_k * 2)
            image_results_task = self._search_images(query, top_k * 2)
            table_results_task = self._search_tables(query, top_k * 2)

            text_results, image_results, table_results = await asyncio.gather(
                text_results_task, image_results_task, table_results_task
            )

            # Объединение и сортировка результатов по score
            all_results = text_results + image_results + table_results
            all_results.sort(key=lambda x: x.score, reverse=True)

            logger.info(
                f"Multimodal search completed: {len(text_results)} text, "
                f"{len(image_results)} images, {len(table_results)} tables"
            )

            return all_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid search error: {e}", exc_info=True)
            return []

    async def _search_text_chunks(
        self, query: str, top_k: int
    ) -> List[MultimodalChunk]:
        """Поиск по текстовым чанкам (гибридный BM25 + векторный)."""
        try:
            # BM25 поиск
            bm25_scores = []
            if self._bm25_text:
                query_tokens = query.lower().split()
                bm25_scores = self._bm25_text.get_scores(query_tokens)

            # Векторный поиск
            query_embedding = await self._get_text_embeddings([query])

            vector_results = self.text_collection.query(
                query_embeddings=query_embedding,
                n_results=min(
                    top_k * 2,
                    max(1, len(self._bm25_text_ids)) if self._bm25_text_ids else top_k,
                ),
            )

            # Объединение результатов (RRF)
            results = []

            for idx, (doc_id, doc, metadata, distance) in enumerate(
                zip(
                    vector_results["ids"][0],
                    vector_results["documents"][0],
                    vector_results["metadatas"][0],
                    vector_results["distances"][0],
                )
            ):
                # Векторный скор (косинусное сходство)
                vector_score = 1 / (1 + distance)

                # BM25 скор
                bm25_score = 0.0
                if self._bm25_text and doc_id in self._bm25_text_ids:
                    bm25_idx = self._bm25_text_ids.index(doc_id)
                    if bm25_idx < len(bm25_scores):
                        bm25_score = bm25_scores[bm25_idx]

                # Гибридная оценка
                hybrid_score = (
                    self.hybrid_config.weight_semantic * vector_score
                    + self.hybrid_config.weight_lexical * (1 + bm25_score)
                )

                chunk = MultimodalChunk(
                    chunk_id=doc_id,
                    type="text",
                    score=hybrid_score,
                    document_url=metadata["document_url"],
                    file_id=metadata["file_id"],
                    content=doc,
                    section_header=metadata.get("section_header"),
                    chunk_index=metadata.get("chunk_index"),
                )

                results.append(chunk)

            # Сортировка по score
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.debug(f"Text collection is empty or error: {e}")
            return []

    async def _search_images(self, query: str, top_k: int) -> List[MultimodalChunk]:
        """Поиск по изображениям."""
        try:
            query_embedding = await self._get_text_embeddings([query])

            results = self.image_collection.query(
                query_embeddings=query_embedding, n_results=top_k
            )

            chunks = []
            for doc_id, doc, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1 / (1 + distance)

                chunk = MultimodalChunk(
                    chunk_id=doc_id,
                    type="image",
                    score=score,
                    document_url=metadata["document_url"],
                    file_id=metadata["file_id"],
                    image_path=metadata.get("image_path"),
                    alt_text=metadata.get("alt_text"),
                    ocr_text=metadata.get("ocr_text"),
                    vlm_description=metadata.get("vlm_description"),
                )

                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.debug(f"Image collection is empty or error: {e}")
            return []

    async def _search_tables(self, query: str, top_k: int) -> List[MultimodalChunk]:
        """Поиск по таблицам (TAG)."""
        try:
            # BM25 поиск
            bm25_scores = []
            if self._bm25_tables:
                query_tokens = query.lower().split()
                bm25_scores = self._bm25_tables.get_scores(query_tokens)

            # Векторный поиск
            query_embedding = await self._get_text_embeddings([query])

            vector_results = self.table_collection.query(
                query_embeddings=query_embedding,
                n_results=min(
                    top_k * 2,
                    max(1, len(self._bm25_tables_ids))
                    if self._bm25_tables_ids
                    else top_k,
                ),
            )

            results = []

            for idx, (doc_id, doc, metadata, distance) in enumerate(
                zip(
                    vector_results["ids"][0],
                    vector_results["documents"][0],
                    vector_results["metadatas"][0],
                    vector_results["distances"][0],
                )
            ):
                vector_score = 1 / (1 + distance)

                # BM25 скор
                bm25_score = 0.0
                if self._bm25_tables and doc_id in self._bm25_tables_ids:
                    bm25_idx = self._bm25_tables_ids.index(doc_id)
                    if bm25_idx < len(bm25_scores):
                        bm25_score = bm25_scores[bm25_idx]

                # Гибридная оценка
                hybrid_score = (
                    self.hybrid_config.weight_semantic * vector_score
                    + self.hybrid_config.weight_lexical * (1 + bm25_score)
                )

                chunk = MultimodalChunk(
                    chunk_id=doc_id,
                    type="table",
                    score=hybrid_score,
                    document_url=metadata["document_url"],
                    file_id=metadata["file_id"],
                    content=doc,
                    table_metadata=metadata,
                )

                results.append(chunk)

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.debug(f"Table collection is empty or error: {e}")
            return []
