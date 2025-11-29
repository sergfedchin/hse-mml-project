"""
Главный оркестратор RAG системы v4.0
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import ollama

from src.config import Config
from src.rag.ingestion import DocumentProcessor
from src.rag.query_processing import QueryProcessor
from src.rag.ranking import CrossEncoderRanker
from src.rag.schema import SearchResult
from src.rag.search import HybridRetriever, KeywordSearchEngine
from src.rag.storage import VectorStorage
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGSystem:
    """Главный класс RAG системы v4.0."""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Объект конфигурации из src.config.Config
        """
        self.config = config
        
        # Путь для сохранения метаданных
        self.metadata_file = Path(self.config.vector_db_path) / "documents_metadata.json"
        
        # Компоненты (инициализируются в initialize())
        self.ollama_client: Optional[ollama.Client] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.vector_storage: Optional[VectorStorage] = None
        self.keyword_engine: Optional[KeywordSearchEngine] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.query_processor: Optional[QueryProcessor] = None
        self.reranker: Optional[CrossEncoderRanker] = None
        
        # Метаданные документов (загружаются из файла при старте)
        self.documents_metadata: Dict[str, Dict] = {}
        
        logger.info("RAG System v4.0 created")
    
    def _save_metadata(self):
        """Сохраняет метаданные документов на диск."""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents_metadata, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Metadata saved: {len(self.documents_metadata)} documents")
        
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _load_metadata(self):
        """Загружает метаданные документов с диска."""
        try:
            if not self.metadata_file.exists():
                logger.info("No metadata file found, starting fresh")
                return
            
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.documents_metadata = json.load(f)
            
            logger.info(f"✓ Metadata loaded: {len(self.documents_metadata)} documents")
        
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            self.documents_metadata = {}
    
    def _reconstruct_metadata_from_db(self):
        """
        Восстанавливает метаданные из ChromaDB (если metadata файл потерян).
        Это fallback на случай если documents_metadata.json был удален.
        """
        try:
            logger.info("Reconstructing metadata from ChromaDB...")
            
            # Получаем все документы из ChromaDB
            all_docs = self.vector_storage.get_all_documents()
            
            # Группируем по file_id
            file_ids = set()
            for doc in all_docs:
                file_id = doc["metadata"].get("file_id")
                if file_id:
                    file_ids.add(file_id)
            
            # Восстанавливаем базовую информацию о каждом документе
            for file_id in file_ids:
                # Получаем все чанки этого документа
                chunks = [d for d in all_docs if d["metadata"].get("file_id") == file_id]
                
                # Подсчитываем типы чанков
                text_chunks = sum(1 for c in chunks if c["metadata"].get("type") == "text")
                table_chunks = sum(1 for c in chunks if c["metadata"].get("type") == "table")
                image_chunks = sum(1 for c in chunks if c["metadata"].get("type") == "image_content")
                
                # Получаем document_url (из первого чанка)
                document_url = chunks[0]["metadata"].get("document_url", "") if chunks else ""
                
                self.documents_metadata[file_id] = {
                    "file_id": file_id,
                    "document_url": document_url,
                    "text_chunks": text_chunks,
                    "table_chunks": table_chunks,
                    "image_chunks": image_chunks,
                    "total_chunks": len(chunks),
                }
            
            # Сохраняем восстановленные метаданные
            self._save_metadata()
            
            logger.info(f"✓ Metadata reconstructed for {len(self.documents_metadata)} documents")
        
        except Exception as e:
            logger.error(f"Failed to reconstruct metadata: {e}")
    
    async def initialize(self):
        """Инициализация всех компонентов."""
        logger.info("Initializing RAG System v4.0...")
        
        try:
            # Создание директорий
            Path(self.config.images_storage_path).mkdir(parents=True, exist_ok=True)
            Path(self.config.vector_db_path).mkdir(parents=True, exist_ok=True)
            
            # 1. Ollama Client
            self.ollama_client = ollama.Client(
                self.config.ollama_base_url,
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            )
            
            # Проверка подключения
            try:
                _ = self.ollama_client.list()
                logger.info("✓ Ollama connection verified")
            except Exception as e:
                logger.error(f"Cannot connect to Ollama: {e}")
                raise
            
            # Проверка VLM модели
            try:
                self.ollama_client.show(self.config.vlm_model)
                logger.info(f"✓ VLM model {self.config.vlm_model} found")
            except Exception as e:
                logger.error(f"VLM model {self.config.vlm_model} not found: {e}")
                raise
            
            # 2. Document Processor
            self.document_processor = DocumentProcessor(
                ollama_client=self.ollama_client,
                vlm_model=self.config.vlm_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                images_storage_path=self.config.images_storage_path
            )
            logger.info("✓ Document Processor initialized")
            
            # 3. Vector Storage
            self.vector_storage = VectorStorage(
                db_path=self.config.vector_db_path,
                ollama_client=self.ollama_client,
                embedding_model=self.config.text_embedding_model
            )
            logger.info("✓ Vector Storage initialized")
            
            # 4. Загрузка метаданных документов
            self._load_metadata()
            
            # Если метаданные не загрузились, но в БД есть данные - восстанавливаем
            if not self.documents_metadata:
                all_docs = self.vector_storage.get_all_documents()
                if all_docs:
                    logger.warning("Metadata file missing but database has data, reconstructing...")
                    self._reconstruct_metadata_from_db()
            
            # 5. Keyword Search Engine (BM25) с кэшированием
            cache_dir = None
            if self.config.bm25_cache_enabled:
                cache_dir = str(Path(self.config.vector_db_path) / "bm25_cache")
            
            self.keyword_engine = KeywordSearchEngine(
                cache_dir=cache_dir,
                auto_unload=self.config.bm25_auto_unload
            )
            
            # Загружаем существующие документы для BM25
            documents = self.vector_storage.get_all_documents()
            self.keyword_engine.initialize(documents)
            
            if self.config.bm25_cache_enabled:
                logger.info("✓ Keyword Search Engine (BM25) initialized with cache")
            else:
                logger.info("✓ Keyword Search Engine (BM25) initialized without cache")
            
            # 6. Hybrid Retriever
            self.hybrid_retriever = HybridRetriever(
                vector_storage=self.vector_storage,
                keyword_engine=self.keyword_engine
            )
            logger.info("✓ Hybrid Retriever initialized")
            
            # 7. Query Processor (если включен в конфиге)
            if self.config.query_expansion_enabled:
                self.query_processor = QueryProcessor(
                    ollama_client=self.ollama_client,
                    llm_model=self.config.text_generation_model,
                    max_queries=self.config.query_expansion_max_queries
                )
                logger.info("✓ Query Processor initialized")
            else:
                self.query_processor = None
                logger.info("⊘ Query Expansion disabled")
            
            # 8. Reranker (если включен в конфиге)
            if self.config.reranker_enabled:
                self.reranker = CrossEncoderRanker(
                    model_name=self.config.reranker_model_name,
                    auto_unload=self.config.reranker_auto_unload
                )
                logger.info("✓ Cross-Encoder Reranker initialized (lazy loading)")
            else:
                self.reranker = None
                logger.info("⊘ Reranker disabled")
            
            logger.info("=" * 60)
            logger.info("RAG System v4.0 initialized successfully!")
            logger.info(f"Indexed documents: {len(self.documents_metadata)}")
            logger.info("=" * 60)
        
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise
    
    def is_ready(self) -> bool:
        """Проверка готовности системы."""
        required_components = [
            self.ollama_client,
            self.document_processor,
            self.vector_storage,
            self.keyword_engine,
            self.hybrid_retriever
        ]
        return all(required_components)
    
    # ========================================================================
    # ДОБАВЛЕНИЕ ДОКУМЕНТА
    # ========================================================================
    
    async def add_document(
        self,
        rst_content: str,
        document_url: str,
        images_base_path: str,
        file_id: Optional[str] = None
    ) -> str:
        """
        Индексирует документ в систему.
        
        Args:
            rst_content: RST контент документа
            document_url: URL документа
            images_base_path: Путь к директории с изображениями
            file_id: Опционально, ID документа (генерируется автоматически)
        
        Returns:
            file_id документа
        """
        logger.info(f"Adding document: {document_url}")
        
        try:
            # 1. Обработка документа → чанки
            chunks, metadata = await self.document_processor.process_document(
                rst_content=rst_content,
                document_url=document_url,
                images_base_path=images_base_path,
                file_id=file_id
            )
            
            file_id = metadata["file_id"]
            
            # 2. Сохранение в векторную БД
            await self.vector_storage.add_chunks(chunks)
            
            # 3. Обновление BM25 индекса
            await self._update_bm25_index()
            
            # 4. Сохранение метаданных (в память И на диск)
            self.documents_metadata[file_id] = metadata
            self._save_metadata()
            
            logger.info(f"✓ Document {file_id} indexed: {metadata['total_chunks']} chunks")
            
            return file_id
        
        except Exception as e:
            logger.error(f"Error adding document: {e}", exc_info=True)
            # Откат изменений
            if file_id:
                await self._rollback_document(file_id)
            raise
    
    async def _update_bm25_index(self):
        """Обновляет BM25 индекс (переинициализация после изменений)."""
        # Инвалидируем кэш
        self.keyword_engine.invalidate_cache()
        
        # Переинициализируем индекс из ChromaDB
        documents = self.vector_storage.get_all_documents()
        self.keyword_engine.initialize(documents)
        
        logger.info("BM25 index updated")
    
    async def _rollback_document(self, file_id: str):
        """Откат изменений при ошибке добавления документа."""
        logger.warning(f"Rolling back document {file_id}")
        try:
            # Получаем metadata для удаления физических файлов
            metadatas = self.vector_storage.get_file_metadata(file_id)
            
            # Удаляем физические файлы изображений
            self._delete_physical_files(metadatas)
            
            # Удаляем из ChromaDB
            self.vector_storage.delete_by_file_id(file_id)
            
            # Удаляем из metadata (если было добавлено)
            if file_id in self.documents_metadata:
                del self.documents_metadata[file_id]
                self._save_metadata()
            
            logger.info(f"✓ Rollback completed for {file_id}")
        except Exception as e:
            logger.error(f"Rollback failed for {file_id}: {e}")
    
    # ========================================================================
    # УДАЛЕНИЕ ДОКУМЕНТА
    # ========================================================================
    
    async def delete_document(self, file_id: str) -> bool:
        """
        ПОЛНОЕ удаление документа из системы.
        
        Удаляет:
        1. Все эмбеддинги из ChromaDB
        2. Физические файлы изображений с диска
        3. Записи из BM25 индекса
        4. Метаданные документа (из памяти и файла)
        
        Args:
            file_id: ID документа для удаления
        
        Returns:
            True если удаление успешно, False если документ не найден
        """
        if file_id not in self.documents_metadata:
            logger.warning(f"Document {file_id} not found in metadata")
            return False
        
        logger.info(f"Starting COMPLETE deletion of document {file_id}")
        
        try:
            # ================================================================
            # ШАГ 1: Получаем метаданные ПЕРЕД удалением из ChromaDB
            # ================================================================
            metadatas = self.vector_storage.get_file_metadata(file_id)
            
            if not metadatas:
                logger.warning(f"No chunks found for file_id={file_id} in ChromaDB")
            else:
                logger.info(f"Found {len(metadatas)} chunks to delete")
            
            # ================================================================
            # ШАГ 2: Удаляем физические файлы изображений
            # ================================================================
            deleted_files = self._delete_physical_files(metadatas)
            logger.info(f"Deleted {deleted_files} physical image files")
            
            # ================================================================
            # ШАГ 3: Удаляем из ChromaDB
            # ================================================================
            self.vector_storage.delete_by_file_id(file_id)
            
            # ================================================================
            # ШАГ 4: Переинициализируем BM25 индекс (без удаленного документа)
            # ================================================================
            self.keyword_engine.invalidate_cache()
            documents = self.vector_storage.get_all_documents()
            self.keyword_engine.initialize(documents)
            logger.info(f"BM25 index reinitialized without {file_id}")
            
            # ================================================================
            # ШАГ 5: Удаляем метаданные (из памяти И файла)
            # ================================================================
            doc_metadata = self.documents_metadata.pop(file_id)
            self._save_metadata()
            
            # ================================================================
            # ФИНАЛЬНАЯ ПРОВЕРКА
            # ================================================================
            logger.info("=" * 60)
            logger.info(f"✓ Document {file_id} COMPLETELY deleted:")
            logger.info(f"  - ChromaDB chunks: {len(metadatas)} removed")
            logger.info(f"  - Physical files: {deleted_files} removed")
            logger.info(f"  - Text chunks: {doc_metadata.get('text_chunks', 0)}")
            logger.info(f"  - Table chunks: {doc_metadata.get('table_chunks', 0)}")
            logger.info(f"  - Image chunks: {doc_metadata.get('image_chunks', 0)}")
            logger.info("=" * 60)
            
            return True
        
        except Exception as e:
            logger.error(f"Error during COMPLETE deletion of {file_id}: {e}", exc_info=True)
            raise
    
    def _delete_physical_files(self, metadatas: List[Dict]) -> int:
        """
        Удаляет физические файлы изображений с диска.
        
        Args:
            metadatas: Список метаданных чанков
        
        Returns:
            Количество удаленных файлов
        """
        deleted_count = 0
        
        for metadata in metadatas:
            # Удаляем только для чанков типа "image_content"
            if metadata.get("type") == "image_content":
                image_path = metadata.get("image_path")
                
                if image_path:
                    try:
                        file_path = Path(image_path)
                        
                        if file_path.exists():
                            file_path.unlink()
                            deleted_count += 1
                            logger.debug(f"Deleted image file: {image_path}")
                        else:
                            logger.warning(f"Image file not found: {image_path}")
                    
                    except Exception as e:
                        logger.error(f"Failed to delete image file {image_path}: {e}")
        
        return deleted_count
    
    # ========================================================================
    # ПОИСК
    # ========================================================================
    
    async def search(
        self,
        user_query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Полный поисковый пайплайн v4.0.
        
        Workflow:
        1. Query Expansion → 2-3 поисковых фразы (если включено)
        2. Hybrid Retrieval (Vector + BM25) → N кандидатов
        3. Cross-Encoder Reranking → топ-K результатов (если включено)
        
        Args:
            user_query: Вопрос пользователя
            top_k: Количество результатов (если None, берется из config)
        
        Returns:
            Список SearchResult отсортированных по релевантности
        """
        # Используем параметры из конфига
        if top_k is None:
            top_k = self.config.max_search_results
        
        retrieval_top_k = self.config.retrieval_top_k
        
        logger.info(f"Search query: '{user_query}'")
        
        try:
            # Step 1: Query Expansion (если включено в конфиге)
            if self.config.query_expansion_enabled and self.query_processor:
                search_queries = await self.query_processor.expand_query(user_query)
            else:
                search_queries = [user_query]
            
            logger.info(f"Search queries: {search_queries}")
            
            # Step 2: Hybrid Retrieval
            candidates = await self.hybrid_retriever.search(
                queries=search_queries,
                top_k=retrieval_top_k  # Из конфига
            )
            
            logger.info(f"Retrieved {len(candidates)} candidates")
            
            if not candidates:
                logger.warning("No candidates found")
                return []
            
            # Step 3: Reranking (если включено в конфиге)
            if self.config.reranker_enabled and self.reranker:
                final_results = self.reranker.rank(
                    user_query=user_query,  # Используем оригинальный запрос
                    candidates=candidates,
                    top_k=top_k
                )
            else:
                # Без реранкинга просто берем первые top_k
                final_results = candidates[:top_k]
            
            logger.info(f"Returning {len(final_results)} results")
            
            return final_results
        
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    # ========================================================================
    # УТИЛИТЫ
    # ========================================================================
    
    def get_document_stats(self, file_id: str) -> Optional[Dict]:
        """Получить статистику по документу."""
        return self.documents_metadata.get(file_id)
    
    def list_documents(self) -> List[Dict]:
        """Список всех документов."""
        return [
            {"file_id": file_id, **metadata}
            for file_id, metadata in self.documents_metadata.items()
        ]
    
    async def cleanup(self):
        """Очистка ресурсов."""
        logger.info("Cleaning up RAG System...")
        # Сохраняем метаданные перед выходом
        self._save_metadata()
