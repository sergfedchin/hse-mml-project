"""
FastAPI HTTP Server для RAG System v4.0

Запуск:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Swagger UI:
    http://localhost:8000/docs

ReDoc:
    http://localhost:8000/redoc
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.config import load_config
from src.rag.main import RAGSystem
from src.utils.logging_config import get_logger, setup_logging

# Глобальная переменная для RAG системы
rag: Optional[RAGSystem] = None
logger = get_logger(__name__)


# ============================================================================
# Pydantic Schemas (запросы и ответы)
# ============================================================================


class HealthResponse(BaseModel):
    """Ответ health check."""

    status: str = Field(..., description="Статус системы")
    version: str = Field(..., description="Версия RAG System")
    documents_count: int = Field(
        ..., description="Количество проиндексированных документов"
    )
    components: Dict[str, bool] = Field(..., description="Статус компонентов")


class SystemInfoResponse(BaseModel):
    """Информация о системе."""

    version: str = Field(..., description="Версия RAG System")
    documents_indexed: int = Field(..., description="Количество документов")
    components: Dict[str, Any] = Field(..., description="Конфигурация компонентов")


class AddDocumentRequest(BaseModel):
    """Запрос на добавление документа."""

    rst_content: str = Field(..., description="RST контент документа", min_length=1)
    document_url: str = Field(..., description="URL документа", min_length=1)
    images_base_path: str = Field(..., description="Путь к директории с изображениями")
    file_id: Optional[str] = Field(
        None, description="ID документа (опционально, генерируется автоматически)"
    )

    @field_validator("rst_content")
    @classmethod
    def validate_rst_content(cls, v: str) -> str:
        if len(v.strip()) == 0:
            raise ValueError("RST content не может быть пустым")
        return v


class AddDocumentResponse(BaseModel):
    """Ответ на добавление документа."""

    success: bool = Field(..., description="Успешность операции")
    file_id: str = Field(..., description="ID добавленного документа")
    message: str = Field(..., description="Сообщение о результате")
    metadata: Dict[str, Any] = Field(..., description="Метаданные документа")


class DeleteDocumentResponse(BaseModel):
    """Ответ на удаление документа."""

    success: bool = Field(..., description="Успешность операции")
    file_id: str = Field(..., description="ID удаленного документа")
    message: str = Field(..., description="Сообщение о результате")


class DocumentInfo(BaseModel):
    """Информация о документе."""

    file_id: str = Field(..., description="ID документа")
    document_url: str = Field(..., description="URL документа")
    text_chunks: int = Field(..., description="Количество текстовых чанков")
    table_chunks: int = Field(..., description="Количество табличных чанков")
    image_chunks: int = Field(..., description="Количество чанков с изображениями")
    total_chunks: int = Field(..., description="Общее количество чанков")


class ListDocumentsResponse(BaseModel):
    """Список документов."""

    success: bool = Field(..., description="Успешность операции")
    count: int = Field(..., description="Количество документов")
    documents: List[DocumentInfo] = Field(..., description="Список документов")


class SearchRequest(BaseModel):
    """Запрос поиска."""

    query: str = Field(..., description="Поисковый запрос", min_length=1)
    top_k: Optional[int] = Field(5, description="Количество результатов", ge=1, le=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if len(v.strip()) == 0:
            raise ValueError("Query не может быть пустым")
        return v.strip()


class SearchResultItem(BaseModel):
    """Один результат поиска."""

    chunk_id: str = Field(..., description="ID чанка")
    score: float = Field(..., description="Релевантность (0-1)")
    content: str = Field(..., description="Контент чанка")
    chunk_type: str = Field(..., description="Тип чанка (text/table/image_content)")
    source_engine: str = Field(..., description="Источник (vector/keyword/hybrid)")
    metadata: Dict[str, Any] = Field(..., description="Метаданные чанка")


class SearchResponse(BaseModel):
    """Ответ на поиск."""

    success: bool = Field(..., description="Успешность операции")
    query: str = Field(..., description="Исходный запрос")
    results_count: int = Field(..., description="Количество результатов")
    results: List[SearchResultItem] = Field(..., description="Результаты поиска")


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""

    success: bool = Field(False, description="Успешность операции")
    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Описание ошибки")
    details: Optional[Dict[str, Any]] = Field(None, description="Дополнительные детали")


# ============================================================================
# Lifecycle (startup/shutdown)
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global rag

    logger.info("=" * 70)
    logger.info("Starting RAG System HTTP Server...")
    logger.info("=" * 70)

    try:
        # Загрузка конфигурации
        config = load_config("config.toml")

        # Настройка логирования
        setup_logging(config.logging)

        # Инициализация RAG системы
        rag = RAGSystem(config)
        await rag.initialize()

        logger.info("=" * 70)
        logger.info("✓ RAG System HTTP Server started successfully!")
        logger.info("✓ API Docs: http://localhost:8000/docs")
        logger.info("=" * 70)

        yield  # Сервер работает

    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise

    finally:
        # Cleanup при остановке
        logger.info("Shutting down RAG System HTTP Server...")
        if rag:
            await rag.cleanup()
        logger.info("✓ Server stopped gracefully")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="RAG System API",
    description="Мультимодальная RAG-система v4.0 с поддержкой текста, таблиц и изображений",
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware (опционально)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Обработчик HTTP исключений."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": "HTTPException", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Обработчик общих исключений."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "InternalServerError",
            "message": "Internal server error occurred",
            "details": {"type": type(exc).__name__},
        },
    )


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", response_model=SystemInfoResponse, tags=["System"])
async def root():
    """
    Получить информацию о системе.
    """
    if not rag or not rag.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System is not ready",
        )

    return SystemInfoResponse(
        version="4.0.0",
        documents_indexed=len(rag.documents_metadata),
        components={
            "query_expansion": rag.config.query_expansion_enabled,
            "reranker": rag.config.reranker_enabled,
            "bm25_cache": rag.config.bm25_cache_enabled,
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Проверяет готовность всех компонентов системы.
    """
    if not rag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System is not initialized",
        )

    components_status = {
        "ollama": rag.ollama_client is not None,
        "vector_storage": rag.vector_storage is not None,
        "keyword_engine": rag.keyword_engine is not None,
        "hybrid_retriever": rag.hybrid_retriever is not None,
        "query_processor": rag.query_processor is not None
        if rag.config.query_expansion_enabled
        else True,
        "reranker": rag.reranker is not None if rag.config.reranker_enabled else True,
    }

    all_ready = all(components_status.values())

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        version="4.0.0",
        documents_count=len(rag.documents_metadata),
        components=components_status,
    )


@app.post(
    "/documents",
    response_model=AddDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def add_document(request: AddDocumentRequest):
    """
    Добавить (проиндексировать) новый документ.

    Обрабатывает RST документ, извлекает текст, таблицы и изображения,
    создает эмбеддинги и добавляет в систему.
    """
    if not rag or not rag.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System is not ready",
        )

    logger.info(f"API: Adding document {request.document_url}")

    try:
        file_id = await rag.add_document(
            rst_content=request.rst_content,
            document_url=request.document_url,
            images_base_path=request.images_base_path,
            file_id=request.file_id,
        )

        metadata = rag.get_document_stats(file_id)

        logger.info(f"✓ API: Document {file_id} added successfully")

        return AddDocumentResponse(
            success=True,
            file_id=file_id,
            message=f"Document indexed successfully with {metadata['total_chunks']} chunks",
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"✗ API: Failed to add document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add document: {str(e)}",
        )


@app.delete(
    "/documents/{file_id}", response_model=DeleteDocumentResponse, tags=["Documents"]
)
async def delete_document(file_id: str):
    """
    Удалить документ из системы.

    Полностью удаляет документ: эмбеддинги, физические файлы, метаданные.
    """
    if not rag or not rag.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System is not ready",
        )

    logger.info(f"API: Deleting document {file_id}")

    try:
        success = await rag.delete_document(file_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found",
            )

        logger.info(f"✓ API: Document {file_id} deleted successfully")

        return DeleteDocumentResponse(
            success=True, file_id=file_id, message="Document deleted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ API: Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )


@app.get("/documents", response_model=ListDocumentsResponse, tags=["Documents"])
async def list_documents():
    """
    Получить список всех проиндексированных документов.
    """
    if not rag or not rag.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System is not ready",
        )

    try:
        documents = rag.list_documents()

        documents_info = [DocumentInfo(**doc) for doc in documents]

        return ListDocumentsResponse(
            success=True, count=len(documents_info), documents=documents_info
        )

    except Exception as e:
        logger.error(f"✗ API: Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        )


@app.get("/documents/{file_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_document_info(file_id: str):
    """
    Получить информацию о конкретном документе.
    """
    if not rag or not rag.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System is not ready",
        )

    try:
        metadata = rag.get_document_stats(file_id)

        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found",
            )

        return DocumentInfo(**metadata)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ API: Failed to get document info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document info: {str(e)}",
        )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Выполнить поиск по проиндексированным документам.

    Использует гибридный поиск (Vector + BM25) с опциональным
    Query Expansion и Reranking.
    """
    if not rag or not rag.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System is not ready",
        )

    logger.info(f"API: Search query: '{request.query}' (top_k={request.top_k})")

    try:
        results = await rag.search(user_query=request.query, top_k=request.top_k)

        search_results = [
            SearchResultItem(
                chunk_id=r.chunk_id,
                score=r.score,
                content=r.content,
                chunk_type=r.chunk_type,
                source_engine=r.source_engine,
                metadata=r.metadata,
            )
            for r in results
        ]

        logger.info(f"✓ API: Search completed, found {len(search_results)} results")

        return SearchResponse(
            success=True,
            query=request.query,
            results_count=len(search_results),
            results=search_results,
        )

    except Exception as e:
        logger.error(f"✗ API: Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


# ============================================================================
# Main (для запуска без uvicorn)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # В production установите False
        log_level="info",
    )
