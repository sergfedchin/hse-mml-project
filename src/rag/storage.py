"""
Обертка над ChromaDB для хранения векторов
"""

from typing import Any, Dict, List, Optional

import chromadb
import ollama
from chromadb.config import Settings

from src.rag.schema import DocumentChunk, SearchResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorStorage:
    """Хранилище векторов (ChromaDB)."""

    def __init__(
        self,
        db_path: str,
        ollama_client: ollama.Client,
        embedding_model: str,
        collection_name: str = "rag_chunks_v4",
    ):
        self.db_path = db_path
        self.ollama_client = ollama_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"description": "Unified RAG chunks v4.0"}
        )

        logger.info(f"VectorStorage initialized: {collection_name}")

    async def add_chunks(self, chunks: List[DocumentChunk]):
        """Добавляет чанки в хранилище."""
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]

        # Генерируем эмбеддинги
        embeddings = []
        for chunk in chunks:
            if chunk.embedding:
                embeddings.append(chunk.embedding)
            else:
                # Для таблиц используем embedding_text
                text = chunk.metadata.get("embedding_text", chunk.content)
                emb = await self._generate_embedding(text)
                embeddings.append(emb)

        # Подготовка документов и метаданных
        documents = []
        metadatas = []

        for chunk in chunks:
            # Для таблиц сохраняем embedding_text как document
            if chunk.type == "table":
                documents.append(chunk.metadata.get("embedding_text", chunk.content))
            else:
                documents.append(chunk.content)

            metadatas.append(
                {
                    "type": chunk.type,
                    "file_id": chunk.metadata.get("file_id", ""),
                    "document_url": chunk.metadata.get("document_url", ""),
                    "breadcrumbs": chunk.breadcrumbs or "",
                    **{
                        k: v
                        for k, v in chunk.metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                }
            )

        try:
            self.collection.add(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )
            logger.info(f"Added {len(chunks)} chunks to VectorStorage")

        except Exception as e:
            logger.error(f"Error adding chunks to VectorStorage: {e}")
            raise

    async def search(
        self, query: str, top_k: int = 20, filter_dict: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Векторный поиск."""
        try:
            query_embedding = await self._generate_embedding(query)

            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k, where=filter_dict
            )

            search_results = []

            for i in range(len(results["ids"][0])):
                # Для таблиц берем Markdown из metadata, для текста - document
                chunk_type = results["metadatas"][0][i].get("type", "text")

                if chunk_type == "table":
                    content = results["metadatas"][0][i].get(
                        "markdown", results["documents"][0][i]
                    )
                else:
                    content = results["documents"][0][i]

                result = SearchResult(
                    chunk_id=results["ids"][0][i],
                    score=1.0
                    - results["distances"][0][i],  # Косинусное расстояние → сходство
                    content=content,
                    metadata=results["metadatas"][0][i],
                    source_engine="vector",
                    chunk_type=chunk_type,
                )
                search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Возвращает все документы для инициализации BM25."""
        try:
            results = self.collection.get()

            documents = []
            for i in range(len(results["ids"])):
                documents.append(
                    {
                        "id": results["ids"][i],
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )

            return documents

        except Exception as e:
            logger.error(f"Error fetching all documents: {e}")
            return []

    def get_file_metadata(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Получает метаданные всех чанков документа ПЕРЕД удалением.
        Критично для получения путей к файлам изображений.

        Args:
            file_id: ID документа

        Returns:
            Список метаданных чанков
        """
        try:
            results = self.collection.get(where={"file_id": file_id})

            if not results["ids"]:
                return []

            metadatas = []
            for i in range(len(results["ids"])):
                metadatas.append(
                    {
                        "chunk_id": results["ids"][i],
                        "type": results["metadatas"][i].get("type"),
                        **results["metadatas"][i],
                    }
                )

            logger.debug(
                f"Retrieved metadata for {len(metadatas)} chunks of file_id={file_id}"
            )
            return metadatas

        except Exception as e:
            logger.error(f"Error fetching metadata for file_id={file_id}: {e}")
            return []

    def delete_by_file_id(self, file_id: str):
        """Удаляет все чанки документа по file_id из ChromaDB."""
        try:
            self.collection.delete(where={"file_id": file_id})
            logger.info(f"Deleted chunks for file_id={file_id} from ChromaDB")
        except Exception as e:
            logger.error(f"Error deleting file_id={file_id} from ChromaDB: {e}")
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Генерирует эмбеддинг через Ollama."""
        try:
            max_length = 8000
            truncated_text = text[:max_length] if len(text) > max_length else text

            response = self.ollama_client.embeddings(
                model=self.embedding_model, prompt=truncated_text
            )

            return response["embedding"]

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            # Возвращаем нулевой вектор при ошибке
            return [0.0] * 768
