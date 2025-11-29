"""
Обработка и подготовка документов для индексации
"""

import asyncio
import base64
import re
import uuid
from contextlib import redirect_stderr
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json_repair
import ollama
import pypandoc
from docutils import nodes
from docutils.core import publish_doctree
from langchain_text_splitters import MarkdownTextSplitter
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from src.rag.schema import DocumentChunk, ImageAnalysis
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Pydantic Schema для Structured Output
# ============================================================================


class ImageAnalysisResult(BaseModel):
    """Структура ответа VLM для анализа изображения."""

    ocr_text: str = Field(
        default="", description="Весь видимый текст на изображении, построчно."
    )
    ui_description: str = Field(
        description="Техническое описание интерфейса на русском языке."
    )


UNIFIED_VLM_PROMPT = """Проанализируй этот скриншот и верни JSON с двумя полями:

1. "ocr_text": Извлеки ВЕСЬ видимый текст на изображении точно как он написан, построчно.

2. "ui_description": Опиши пользовательский интерфейс с техническими подробностями на русском:
   - Основные элементы (кнопки, поля, выпадающие списки, вкладки, меню)
   - Структура макета и визуальная иерархия
   - Все видимые текстовые надписи и их расположение
   - Интерактивные элементы и их функции
   - Таблицы данных или структурированная информация

Отвечай строго на русском языке БЕЗ markdown форматирования.
Верни строго валидный JSON без дополнительных пояснений.
"""


# ============================================================================
# RST Table Parser
# ============================================================================


class RSTTableParser:
    """Парсер RST таблиц с упрощением сложных структур."""

    @staticmethod
    def remove_rst_tables_by_pattern(rst_content: str) -> str:
        """Удаляет RST таблицы по паттернам."""
        grid_table_pattern = r"\n\+[-=+]+\+\n(?:\|[^\n]*\|\n)+\+[-=+]+\+(?:\n(?:\|[^\n]*\|\n)+\+[-=+]+\+)*"
        modified_rst = re.sub(
            grid_table_pattern, "\n\n[TABLE_PLACEHOLDER]\n\n", rst_content
        )
        return modified_rst

    @staticmethod
    def parse_rst_tables(rst_content: str) -> List[Dict[str, Any]]:
        """Парсит RST таблицы используя docutils."""
        tables = []
        try:
            settings_overrides = {
                "report_level": 5,
                "halt_level": 5,
                "warning_stream": StringIO(),
            }

            with redirect_stderr(StringIO()):
                doctree = publish_doctree(
                    rst_content, settings_overrides=settings_overrides
                )

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
        num_cols = int(tgroup.get("cols", 0))
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
                    header_row = RSTTableParser._parse_row_simple(row, num_cols)
                    if header_row:
                        result["headers"] = header_row
                        break

        # Если заголовков нет, используем первую строку tbody
        if not result["headers"] and tbody:
            row_nodes = [r for r in tbody.children if isinstance(r, nodes.row)]
            if row_nodes:
                first_row = RSTTableParser._parse_row_simple(row_nodes[0], num_cols)
                if first_row:
                    result["headers"] = first_row
                    tbody.children = tbody.children[1:]

        if not result["headers"]:
            result["headers"] = [f"col_{i}" for i in range(num_cols)]

        # Тело таблицы
        if tbody:
            result["rows"] = RSTTableParser._parse_tbody_with_rowspan(tbody, num_cols)

        return result

    @staticmethod
    def _parse_row_simple(
        row_node: nodes.row, expected_cols: int
    ) -> Optional[List[str]]:
        """Простой парсинг строки без учета rowspan."""
        cells = []
        col_idx = 0

        for entry in row_node.children:
            if not isinstance(entry, nodes.entry):
                continue

            cell_text = RSTTableParser._get_clean_text(entry)
            morecols = int(entry.get("morecols", 0))

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
        """Парсит tbody с учетом rowspan (morerows) и colspan (morecols)."""
        rows = []
        rowspan_matrix = {}
        row_nodes = [r for r in tbody.children if isinstance(r, nodes.row)]

        for row_idx, row_node in enumerate(row_nodes):
            row_cells = [""] * num_cols
            col_idx = 0

            # Заполняем ячейки из предыдущих rowspan
            if row_idx in rowspan_matrix:
                for col, (text, remaining) in rowspan_matrix[row_idx].items():
                    if col < num_cols:
                        row_cells[col] = text

            # Обрабатываем ячейки текущей строки
            for entry in row_node.children:
                if not isinstance(entry, nodes.entry):
                    continue

                # Пропускаем занятые колонки
                while col_idx < num_cols and row_cells[col_idx] != "":
                    col_idx += 1

                if col_idx >= num_cols:
                    break

                cell_text = RSTTableParser._get_clean_text(entry)
                morecols = int(entry.get("morecols", 0))
                morerows = int(entry.get("morerows", 0))

                # Заполняем текущую ячейку и colspan
                for i in range(morecols + 1):
                    if col_idx < num_cols:
                        row_cells[col_idx] = cell_text

                        # Записываем rowspan для будущих строк
                        if morerows > 0:
                            for future_row in range(
                                row_idx + 1, row_idx + morerows + 1
                            ):
                                if future_row not in rowspan_matrix:
                                    rowspan_matrix[future_row] = {}
                                rowspan_matrix[future_row][col_idx] = (
                                    cell_text,
                                    morerows,
                                )

                        col_idx += 1

            rows.append(row_cells)

        return rows

    @staticmethod
    def _get_clean_text(node: nodes.Node) -> str:
        """Извлекает текст из node и очищает от ошибок docutils."""
        text_parts = []

        def collect_text(n):
            if isinstance(n, (nodes.system_message, nodes.problematic)):
                return
            if isinstance(n, nodes.Text):
                text_parts.append(str(n))
            if hasattr(n, "children"):
                for child in n.children:
                    collect_text(child)

        collect_text(node)
        result = " ".join(text_parts).strip()
        result = re.sub(r"\s+", " ", result)

        # Постобработка: удаляем сообщения об ошибках docutils
        result = re.sub(r'No role entry for "[^"]*" in module "[^"]*"\.\s*', "", result)
        result = re.sub(r'Trying "[^"]*" as canonical role name\.\s*', "", result)
        result = re.sub(r'Unknown interpreted text role "[^"]*"\.\s*', "", result)
        result = re.sub(r":\d+: \([A-Z]+/\d+\) [^\n]+\n?", "", result)
        result = re.sub(r"\s+", " ", result).strip()

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
            escaped_row = [
                cell.replace("|", "\\|").replace("\n", " ") for cell in normalized_row
            ]
            md_lines.append("| " + " | ".join(escaped_row) + " |")

        return "\n".join(md_lines)


# ============================================================================
# Document Processor
# ============================================================================


class DocumentProcessor:
    """Обработчик документов: RST → chunks."""

    def __init__(
        self,
        ollama_client: ollama.Client,
        vlm_model: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        images_storage_path: str = "./data/images",
    ):
        self.ollama_client = ollama_client
        self.vlm_model = vlm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.images_storage_path = Path(images_storage_path)
        self.images_storage_path.mkdir(parents=True, exist_ok=True)

    async def process_document(
        self,
        rst_content: str,
        document_url: str,
        images_base_path: str,
        file_id: Optional[str] = None,
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Обрабатывает RST документ и возвращает список чанков.

        Returns:
            (chunks, metadata) где chunks = List[DocumentChunk], metadata = Dict
        """
        if file_id is None:
            file_id = str(uuid.uuid4())

        logger.info(f"Processing document {file_id}: {document_url}")

        # 1. Парсинг таблиц из RST
        parsed_tables = RSTTableParser.parse_rst_tables(rst_content)
        logger.info(f"Extracted {len(parsed_tables)} tables")

        # 2. Удаление таблиц из RST
        rst_without_tables = RSTTableParser.remove_rst_tables_by_pattern(rst_content)

        # 3. Конвертация RST → Markdown
        markdown_content = await self._rst_to_markdown(rst_without_tables)

        # 4. Извлечение структуры заголовков для breadcrumbs
        header_structure = self._extract_header_structure(markdown_content)

        # 5. Разбиение текста на чанки с breadcrumbs
        text_chunks = await self._split_text_to_chunks(
            markdown_content, file_id, document_url, header_structure
        )
        logger.info(f"Created {len(text_chunks)} text chunks")

        # 6. Обработка таблиц → чанки
        table_chunks = await self._process_tables(parsed_tables, file_id, document_url)
        logger.info(f"Created {len(table_chunks)} table chunks")

        # 7. Обработка изображений → текстовые чанки
        image_chunks = await self._process_images(
            markdown_content, images_base_path, file_id, document_url
        )
        logger.info(f"Created {len(image_chunks)} image-content chunks")

        # 8. Объединение всех чанков
        all_chunks = text_chunks + table_chunks + image_chunks

        metadata = {
            "file_id": file_id,
            "document_url": document_url,
            "text_chunks": len(text_chunks),
            "table_chunks": len(table_chunks),
            "image_chunks": len(image_chunks),
            "total_chunks": len(all_chunks),
        }

        return all_chunks, metadata

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

    def _extract_header_structure(
        self, markdown_content: str
    ) -> List[Tuple[str, int, int]]:
        """
        Извлекает заголовки с их позициями и уровнями.

        Returns:
            List[(header_text, position, level)]
        """
        headers = []
        header_pattern = r"^(#{1,6})\s+(.+)$"

        for match in re.finditer(header_pattern, markdown_content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            position = match.start()
            headers.append((title, position, level))

        return headers

    def _build_breadcrumbs(
        self, chunk_position: int, header_structure: List[Tuple[str, int, int]]
    ) -> str:
        """
        Строит путь breadcrumbs для чанка.

        Пример: "Глава 1 > Раздел 2 > Подраздел 3"
        """
        breadcrumbs = []
        current_level = 0

        for header_text, pos, level in header_structure:
            if pos >= chunk_position:
                break

            # Если уровень меньше или равен текущему, обновляем стек
            if level <= current_level:
                # Удаляем заголовки с уровнем >= текущего
                breadcrumbs = [b for b in breadcrumbs if b[1] < level]

            breadcrumbs.append((header_text, level))
            current_level = level

        return " > ".join([text for text, _ in breadcrumbs])

    async def _split_text_to_chunks(
        self,
        markdown_content: str,
        file_id: str,
        document_url: str,
        header_structure: List[Tuple[str, int, int]],
    ) -> List[DocumentChunk]:
        """Разбивает текст на чанки с breadcrumbs."""
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        text_splits = splitter.split_text(markdown_content)
        chunks = []

        for idx, text in enumerate(text_splits):
            # Находим позицию чанка в оригинальном тексте
            chunk_position = markdown_content.find(text)
            breadcrumbs = self._build_breadcrumbs(chunk_position, header_structure)

            chunk = DocumentChunk(
                id=f"{file_id}_text_{idx}",
                content=text,
                type="text",
                breadcrumbs=breadcrumbs,
                metadata={
                    "file_id": file_id,
                    "document_url": document_url,
                    "chunk_index": idx,
                    "breadcrumbs": breadcrumbs,
                },
            )
            chunks.append(chunk)

        return chunks

    async def _process_tables(
        self, parsed_tables: List[Dict[str, Any]], file_id: str, document_url: str
    ) -> List[DocumentChunk]:
        """Обрабатывает таблицы в чанки."""
        table_chunks = []

        for table_idx, table_data in enumerate(parsed_tables):
            try:
                headers = table_data.get("headers", [])
                rows = table_data.get("rows", [])
                caption = table_data.get("caption", "")

                if not headers or not rows:
                    continue

                # Определяем оптимальный размер чанка
                chunk_size = self._calculate_table_chunk_size(headers, rows)
                overlap = max(2, chunk_size // 5)

                # Разбиваем таблицу на row-level чанки
                for start_idx in range(0, len(rows), chunk_size - overlap):
                    end_idx = min(start_idx + chunk_size, len(rows))
                    chunk_rows = rows[start_idx:end_idx]

                    # Создаем Markdown представление чанка
                    chunk_markdown = self._create_table_chunk_markdown(
                        caption, headers, chunk_rows, start_idx, end_idx, len(rows)
                    )

                    # Создаем текст для embedding (структурированный)
                    embedding_text = self._format_table_for_embedding(
                        caption, headers, chunk_rows
                    )

                    chunk_id = f"{file_id}_table_{table_idx}_rows_{start_idx}_{end_idx}"

                    chunk = DocumentChunk(
                        id=chunk_id,
                        content=chunk_markdown,  # Markdown для отображения
                        type="table",
                        metadata={
                            "file_id": file_id,
                            "document_url": document_url,
                            "table_id": f"{file_id}_table_{table_idx}",
                            "caption": caption,
                            "headers": headers,
                            "row_range": f"{start_idx}-{end_idx}",
                            "total_rows": len(rows),
                            "num_cols": len(headers),
                            "embedding_text": embedding_text,  # Для векторизации
                        },
                    )

                    table_chunks.append(chunk)

                    if end_idx >= len(rows):
                        break

            except Exception as e:
                logger.warning(f"Failed to process table {table_idx}: {e}")
                continue

        return table_chunks

    def _calculate_table_chunk_size(
        self, headers: List[str], sample_rows: List[List[str]]
    ) -> int:
        """Рассчитывает оптимальный размер чанка таблицы."""
        num_cols = len(headers)

        # Оценка среднего размера ячейки
        if sample_rows:
            total_cell_length = sum(
                len(str(cell)) for row in sample_rows[:10] for cell in row
            )
            cell_count = sum(len(row) for row in sample_rows[:10])
            avg_cell_length = total_cell_length / cell_count if cell_count > 0 else 20
        else:
            avg_cell_length = 20

        # Токены на строку (1 токен ≈ 4 символа)
        tokens_per_row = (avg_cell_length * num_cols) / 4

        TARGET_TOKENS = 450
        rows_per_chunk = max(5, min(20, int(TARGET_TOKENS / tokens_per_row)))

        # Корректировки
        if num_cols <= 3:
            rows_per_chunk = min(20, rows_per_chunk * 2)
        elif num_cols > 10:
            rows_per_chunk = max(3, rows_per_chunk // 2)

        if avg_cell_length > 100:
            rows_per_chunk = max(3, rows_per_chunk // 2)

        return rows_per_chunk

    def _create_table_chunk_markdown(
        self,
        caption: str,
        headers: List[str],
        rows: List[List[str]],
        start_idx: int,
        end_idx: int,
        total_rows: int,
    ) -> str:
        """Создает Markdown представление чанка таблицы."""
        md_lines = []

        if caption:
            md_lines.append(
                f"**{caption}** (строки {start_idx + 1}-{end_idx} из {total_rows})\n"
            )
        else:
            md_lines.append(
                f"**Таблица** (строки {start_idx + 1}-{end_idx} из {total_rows})\n"
            )

        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows:
            normalized_row = (row + [""] * len(headers))[: len(headers)]
            escaped_row = [
                str(cell).replace("|", "\\|").replace("\n", " ").strip()
                for cell in normalized_row
            ]
            md_lines.append("| " + " | ".join(escaped_row) + " |")

        return "\n".join(md_lines)

    def _format_table_for_embedding(
        self, caption: str, headers: List[str], rows: List[List[str]]
    ) -> str:
        """Форматирует таблицу для генерации embedding."""
        parts = []

        if caption:
            parts.append(f"Таблица: {caption}")

        parts.append(f"Колонки: {' | '.join(headers)}")

        for row in rows:
            row_parts = []
            for col_idx, cell_value in enumerate(row):
                if col_idx < len(headers):
                    header = headers[col_idx]
                    clean_value = str(cell_value).replace("\n", " ").strip()
                    if clean_value:
                        row_parts.append(f"{header}: {clean_value}")

            if row_parts:
                parts.append(" | ".join(row_parts))

        return "\n".join(parts)

    async def _process_images(
        self,
        markdown_content: str,
        images_base_path: str,
        file_id: str,
        document_url: str,
    ) -> List[DocumentChunk]:
        """
        Обрабатывает изображения и создает текстовые чанки с их описанием.
        Изображения НЕ сохраняются в векторную БД отдельно.
        """
        image_chunks = []
        image_pattern = r"!\[([^\]]*)\]\(([^\)]+)\)"
        matches = re.findall(image_pattern, markdown_content)

        if not matches:
            return image_chunks

        for idx, (alt_text, image_path) in enumerate(matches):
            try:
                # Анализ изображения через VLM
                analysis = await self._analyze_image(image_path, images_base_path)

                # Сохраняем физический файл изображения на диск
                clean_path = image_path.lstrip("/")
                full_path = Path(images_base_path) / clean_path

                if not full_path.exists():
                    logger.warning(f"Image not found: {full_path}")
                    continue

                image_id = f"{file_id}_img_{idx}"
                storage_path = self.images_storage_path / f"{image_id}.jpg"

                # Копируем изображение
                image = PILImage.open(full_path).convert("RGB")
                image.save(storage_path, format="JPEG", quality=85)

                # Создаем текстовый чанк с описанием изображения
                image_text_content = analysis.to_text(alt_text)

                chunk = DocumentChunk(
                    id=image_id,
                    content=image_text_content,
                    type="image_content",
                    metadata={
                        "file_id": file_id,
                        "document_url": document_url,
                        "image_path": str(storage_path),
                        "alt_text": alt_text,
                        "ocr_text": analysis.ocr_text,
                        "ui_description": analysis.ui_description,
                        "original_path": image_path,
                    },
                )

                image_chunks.append(chunk)

            except Exception as e:
                logger.warning(f"Failed to process image {image_path}: {e}")
                continue

        return image_chunks

    async def _analyze_image(
        self, image_path: str, images_base_path: str
    ) -> ImageAnalysis:
        """Анализирует изображение через VLM."""
        try:
            clean_path = image_path.lstrip("/")
            full_path = Path(images_base_path) / clean_path

            if not full_path.exists():
                raise FileNotFoundError(f"Image not found: {full_path}")

            image = PILImage.open(full_path).convert("RGB")

            # Оптимизация размера
            max_dim = 1280
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim), PILImage.Resampling.LANCZOS)

            if image.width < 32 or image.height < 32:
                scale = 32.0 / min(image.width, image.height)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, PILImage.Resampling.LANCZOS)

            # Конвертация в base64
            img_buffer = BytesIO()
            image.save(img_buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            # Вызов VLM
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._run_vlm_sync, img_base64)

            return ImageAnalysis(
                ocr_text=result.ocr_text, ui_description=result.ui_description
            )

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return ImageAnalysis(ocr_text="", ui_description="")

    def _run_vlm_sync(self, img_base64: str) -> ImageAnalysisResult:
        """Синхронный вызов VLM с structured output."""
        try:
            response = self.ollama_client.chat(
                model=self.vlm_model,
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
