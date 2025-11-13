"""
RAG Engine - основная логика работы с документами и векторной базой.
"""

import asyncio
import base64
import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any
import io

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import MarkdownTextSplitter
import pypandoc
from PIL import Image
import pytesseract
import httpx

from config import Config

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Класс для управления RAG-системой: индексация документов и поиск.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.chroma_client = None
        self.collection = None
        self.ollama_client = None
        
        # Модели загружаются лениво (только при использовании)
        self._text_embeddings_model = None
        self._vision_embeddings_model = None
        
        # Хранилище метаданных документов в памяти
        self.documents_metadata = {}
        
        logger.info("RAG Engine initialized")
    
    async def initialize(self):
        """Инициализация компонентов системы"""
        try:
            # Создание директории для изображений
            Path(self.config.images_storage_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Images storage: {self.config.images_storage_path}")
            
            # Инициализация ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.vector_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Создание или получение коллекции
            self.collection = self.chroma_client.get_or_create_collection(
                name="documentation",
                metadata={"description": "Technical documentation with multimodal support"}
            )
            
            logger.info(f"Vector DB initialized at {self.config.vector_db_path}")
            
            # Инициализация HTTP клиента для Ollama
            self.ollama_client = httpx.AsyncClient(
                base_url=self.config.ollama_base_url,
                timeout=120.0
            )
            
            logger.info(f"Ollama client initialized at {self.config.ollama_base_url}")
            
            # Проверка доступности Ollama
            try:
                response = await self.ollama_client.get("/api/tags")
                response.raise_for_status()
                logger.info("Ollama connection verified")
            except Exception as e:
                logger.warning(f"Cannot connect to Ollama: {e}")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            raise
    
    async def cleanup(self):
        """Очистка ресурсов"""
        if self.ollama_client:
            await self.ollama_client.aclose()
        logger.info("RAG Engine cleaned up")
    
    def is_ready(self) -> bool:
        """Проверка готовности системы"""
        return (
            self.chroma_client is not None and
            self.collection is not None and
            self.ollama_client is not None
        )
    
    async def check_vector_db(self) -> bool:
        """Проверка соединения с векторной БД"""
        try:
            self.collection.count()
            return True
        except:
            return False
    
    async def add_document(
        self,
        rst_content: str,
        document_url: str,
        images_base_path: str
    ) -> str:
        """
        Добавить новый документ в систему.
        
        Args:
            rst_content: Содержимое RST файла
            document_url: URL страницы документации
            images_base_path: Путь к директории с изображениями
            
        Returns:
            UUID документа
        """
        file_id = str(uuid.uuid4())
        logger.info(f"Processing document {file_id}")
        
        try:
            # 1. Конвертация RST в Markdown
            markdown_content = await self._rst_to_markdown(rst_content)
            logger.debug(f"Converted RST to Markdown ({len(markdown_content)} chars)")
            
            # 2. Извлечение изображений из содержимого
            images_info = await self._extract_images(
                markdown_content,
                images_base_path,
                file_id
            )
            logger.info(f"Extracted {len(images_info)} images")
            
            # 3. Обработка таблиц
            markdown_content, tables_info = await self._process_tables(markdown_content)
            logger.info(f"Processed {len(tables_info)} tables")
            
            # 4. Разбиение на чанки
            chunks = await self._split_into_chunks(markdown_content)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # 5. Генерация эмбеддингов для текстовых чанков
            text_embeddings = await self._generate_text_embeddings(
                [chunk['text'] for chunk in chunks]
            )
            logger.debug(f"Generated {len(text_embeddings)} text embeddings")
            
            # 6. Генерация эмбеддингов для изображений
            image_embeddings = []
            if images_info:
                image_embeddings = await self._generate_image_embeddings(
                    [img['path'] for img in images_info]
                )
                logger.debug(f"Generated {len(image_embeddings)} image embeddings")
            
            # 7. Добавление в векторную БД
            await self._add_to_vector_db(
                file_id=file_id,
                document_url=document_url,
                chunks=chunks,
                text_embeddings=text_embeddings,
                images_info=images_info,
                image_embeddings=image_embeddings,
                tables_info=tables_info
            )
            
            # 8. Сохранение метаданных документа
            self.documents_metadata[file_id] = {
                'document_url': document_url,
                'chunks_count': len(chunks),
                'images_count': len(images_info),
                'tables_count': len(tables_info)
            }
            
            logger.info(f"Document {file_id} added successfully")
            return file_id
            
        except Exception as e:
            logger.error(f"Error adding document {file_id}: {e}", exc_info=True)
            # Откат изменений
            await self._rollback_document(file_id)
            raise
    
    async def delete_document(self, file_id: str):
        """
        Удалить документ и все связанные данные.
        
        Args:
            file_id: UUID документа
        """
        logger.info(f"Deleting document {file_id}")
        
        try:
            # 1. Удаление из векторной БД
            self.collection.delete(
                where={"file_id": file_id}
            )
            
            # 2. Удаление изображений с диска
            images_dir = Path(self.config.images_storage_path) / file_id
            if images_dir.exists():
                for img_file in images_dir.glob("*"):
                    img_file.unlink()
                images_dir.rmdir()
                logger.debug(f"Deleted images directory: {images_dir}")
            
            # 3. Удаление метаданных
            if file_id in self.documents_metadata:
                del self.documents_metadata[file_id]
            
            logger.info(f"Document {file_id} deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting document {file_id}: {e}", exc_info=True)
            raise
    
    async def document_exists(self, file_id: str) -> bool:
        """Проверить существование документа"""
        try:
            results = self.collection.get(
                where={"file_id": file_id},
                limit=1
            )
            return len(results['ids']) > 0
        except:
            return False
    
    async def search(
        self,
        query_text: str,
        query_images: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Выполнить поиск по базе документов.
        
        Args:
            query_text: Текстовый запрос
            query_images: Список изображений в base64 (опционально)
            
        Returns:
            Список результатов поиска
        """
        logger.info(f"Searching for: {query_text[:100]}")
        
        try:
            results = []
            
            # 1. Текстовый поиск
            text_results = await self._search_by_text(query_text)
            results.extend(text_results)
            
            # 2. Визуальный поиск (если есть изображения)
            if query_images:
                visual_results = await self._search_by_images(query_images)
                results.extend(visual_results)
            
            # 3. Объединение и ранжирование результатов
            merged_results = self._merge_and_rank_results(results)
            
            # 4. Ограничение количества результатов
            top_results = merged_results[:self.config.max_search_results]
            
            # 5. Обогащение результатов изображениями
            enriched_results = await self._enrich_results_with_images(top_results)
            
            logger.info(f"Returning {len(enriched_results)} results")
            return enriched_results
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику по системе"""
        try:
            total_chunks = self.collection.count()
            
            total_documents = len(self.documents_metadata)
            
            total_images = sum(
                meta['images_count']
                for meta in self.documents_metadata.values()
            )
            
            # Размер векторной БД
            db_path = Path(self.config.vector_db_path)
            db_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
            db_size_str = self._format_size(db_size)
            
            return {
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'total_images': total_images,
                'vector_db_size': db_size_str
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'total_images': 0,
                'vector_db_size': '0 B'
            }
    
    # ========== Приватные методы ==========
    
    async def _rst_to_markdown(self, rst_content: str) -> str:
        """Конвертировать RST в Markdown"""
        try:
            # Используем pypandoc для конвертации
            markdown = pypandoc.convert_text(
                rst_content,
                'gfm',
                format='rst',
                extra_args=['--wrap=none']
            )
            return markdown
        except Exception as e:
            logger.error(f"RST conversion error: {e}")
            # Fallback: возвращаем исходный текст
            return rst_content
    
    async def _extract_images(
        self,
        content: str,
        images_base_path: str,
        file_id: str
    ) -> List[Dict[str, str]]:
        """Извлечь изображения из содержимого и сохранить на диск"""
        images_info = []
        
        # Поиск ссылок на изображения в Markdown
        image_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        matches = re.finditer(image_pattern, content)
        
        images_dir = Path(self.config.images_storage_path) / file_id
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for match in matches:
            alt_text = match.group(1)
            image_path = match.group(2)
            
            # Полный путь к исходному изображению
            source_path = Path(images_base_path) / image_path
            
            if source_path.exists():
                # Генерация UUID для изображения
                img_id = str(uuid.uuid4())
                img_ext = source_path.suffix
                
                # Целевой путь
                target_path = images_dir / f"{img_id}{img_ext}"
                
                # Копирование изображения
                target_path.write_bytes(source_path.read_bytes())
                
                images_info.append({
                    'id': img_id,
                    'path': str(target_path),
                    'alt_text': alt_text,
                    'original_path': image_path
                })
                
                logger.debug(f"Saved image: {img_id}")
        
        return images_info
    
    async def _process_tables(self, content: str) -> tuple:
        """Обработать таблицы в содержимом"""
        tables_info = []
        
        # Простая эвристика: поиск HTML таблиц
        table_pattern = r'<table[^>]*>.*?</table>'
        tables = re.findall(table_pattern, content, re.DOTALL)
        
        for idx, table_html in enumerate(tables):
            table_id = f"table_{idx}"
            tables_info.append({
                'id': table_id,
                'html': table_html,
                'summary': f"Table {idx + 1}"  # Можно улучшить через LLM
            })
        
        return content, tables_info
    
    async def _split_into_chunks(self, content: str) -> List[Dict[str, str]]:
        """Разбить содержимое на чанки"""
        splitter = MarkdownTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        texts = splitter.split_text(content)
        
        chunks = []
        for idx, text in enumerate(texts):
            chunks.append({
                'id': str(uuid.uuid4()),
                'text': text,
                'index': idx
            })
        
        return chunks
    
    async def _generate_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Генерация текстовых эмбеддингов через Ollama"""
        embeddings = []
        
        for text in texts:
            try:
                response = await self.ollama_client.post(
                    "/api/embeddings",
                    json={
                        "model": self.config.text_embedding_model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                embedding = response.json()['embedding']
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error generating text embedding: {e}")
                # Fallback: нулевой вектор
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    async def _generate_image_embeddings(self, image_paths: List[str]) -> List[List[float]]:
        """Генерация визуальных эмбеддингов через Ollama"""
        embeddings = []
        
        for img_path in image_paths:
            try:
                # Чтение изображения и конвертация в base64
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                img_b64 = base64.b64encode(img_data).decode('utf-8')
                
                # Запрос к Ollama для визуального эмбеддинга
                response = await self.ollama_client.post(
                    "/api/embeddings",
                    json={
                        "model": self.config.vision_embedding_model,
                        "prompt": "image embedding",
                        "images": [img_b64]
                    }
                )
                response.raise_for_status()
                embedding = response.json()['embedding']
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error generating image embedding for {img_path}: {e}")
                # Fallback: нулевой вектор
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    async def _add_to_vector_db(
        self,
        file_id: str,
        document_url: str,
        chunks: List[Dict],
        text_embeddings: List[List[float]],
        images_info: List[Dict],
        image_embeddings: List[List[float]],
        tables_info: List[Dict]
    ):
        """Добавить данные в векторную БД"""
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        # Добавление текстовых чанков
        for chunk, embedding in zip(chunks, text_embeddings):
            ids.append(chunk['id'])
            embeddings.append(embedding)
            documents.append(chunk['text'])
            metadatas.append({
                'file_id': file_id,
                'document_url': document_url,
                'type': 'text',
                'chunk_index': chunk['index']
            })
        
        # Добавление изображений
        for img, embedding in zip(images_info, image_embeddings):
            ids.append(img['id'])
            embeddings.append(embedding)
            documents.append(img['alt_text'])
            metadatas.append({
                'file_id': file_id,
                'document_url': document_url,
                'type': 'image',
                'image_path': img['path']
            })
        
        # Добавление в ChromaDB
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
    
    async def _search_by_text(self, query: str) -> List[Dict]:
        """Поиск по текстовому запросу"""
        # Генерация эмбеддинга запроса
        query_embedding = await self._generate_text_embeddings([query])
        
        # Поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.config.max_search_results * 2,  # Берём с запасом
            where={"type": {"$in": ["text", "image"]}}
        )
        
        # Форматирование результатов
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1.0 - results['distances'][0][i],
                'source': 'text_search'
            })
        
        return formatted_results
    
    async def _search_by_images(self, images_b64: List[str]) -> List[Dict]:
        """Поиск по изображениям"""
        all_results = []
        
        for img_b64 in images_b64:
            try:
                # Генерация эмбеддинга для query-изображения
                response = await self.ollama_client.post(
                    "/api/embeddings",
                    json={
                        "model": self.config.vision_embedding_model,
                        "prompt": "query image",
                        "images": [img_b64]
                    }
                )
                response.raise_for_status()
                query_embedding = response.json()['embedding']
                
                # Поиск в ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=self.config.max_search_results,
                    where={"type": "image"}
                )
                
                # Форматирование результатов
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1.0 - results['distances'][0][i],
                        'source': 'visual_search'
                    })
                
            except Exception as e:
                logger.error(f"Error in visual search: {e}")
        
        return all_results
    
    def _merge_and_rank_results(self, results: List[Dict]) -> List[Dict]:
        """Объединить и ранжировать результаты"""
        # Группировка по document_url
        grouped = {}
        for result in results:
            url = result['metadata']['document_url']
            if url not in grouped:
                grouped[url] = []
            grouped[url].append(result)
        
        # Ранжирование: берём лучший результат для каждого документа
        ranked = []
        for url, group in grouped.items():
            best = max(group, key=lambda x: x['similarity'])
            ranked.append(best)
        
        # Сортировка по similarity
        ranked.sort(key=lambda x: x['similarity'], reverse=True)
        
        return ranked
    
    async def _enrich_results_with_images(self, results: List[Dict]) -> List[Dict]:
        """Обогатить результаты изображениями"""
        enriched = []
        
        for result in results:
            file_id = result['metadata']['file_id']
            document_url = result['metadata']['document_url']
            
            # Получение изображений для этого документа
            images = self._get_document_images(file_id)
            
            enriched.append({
                'document_url': document_url,
                'text': result['text'],
                'images': images,
                'similarity': result['similarity']
            })
        
        return enriched
    
    def _get_document_images(self, file_id: str) -> List[str]:
        """Получить изображения документа в base64"""
        images_b64 = []
        
        images_dir = Path(self.config.images_storage_path) / file_id
        if images_dir.exists():
            for img_file in images_dir.glob("*"):
                try:
                    img_data = img_file.read_bytes()
                    img_b64 = base64.b64encode(img_data).decode('utf-8')
                    images_b64.append(img_b64)
                except Exception as e:
                    logger.error(f"Error reading image {img_file}: {e}")
        
        return images_b64
    
    async def _rollback_document(self, file_id: str):
        """Откат изменений при ошибке"""
        try:
            await self.delete_document(file_id)
        except:
            pass
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Форматирование размера в человекочитаемый вид"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
