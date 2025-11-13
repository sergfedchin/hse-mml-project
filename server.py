"""
FastAPI сервер для RAG-системы с поддержкой мультимодального поиска.
"""

import argparse
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.rag_engine import RAGEngine
from src.config import load_config, Config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальные переменные
rag_engine: Optional[RAGEngine] = None
config: Optional[Config] = None


# Pydantic модели для API
class FileAddRequest(BaseModel):
    """Модель для добавления файла через JSON"""
    content: str = Field(..., description="Содержимое .rst файла")
    url: str = Field(..., description="Ссылка на страницу документации")
    images_base_path: str = Field(..., description="Базовый путь до директории с изображениями")


class FileAddResponse(BaseModel):
    """Ответ на добавление файла"""
    file_id: str
    status: str = "success"
    message: str = "File added successfully"


class FileDeleteResponse(BaseModel):
    """Ответ на удаление файла"""
    status: str = "success"
    message: str = "File and associated data deleted successfully"


class QueryRequest(BaseModel):
    """Запрос поиска"""
    text: str = Field(..., description="Текстовый запрос")
    images: Optional[List[str]] = Field(None, description="Массив изображений в base64")


class QueryResult(BaseModel):
    """Результат поиска"""
    document_url: str
    text: str
    images: List[str]
    similarity: float


class QueryResponse(BaseModel):
    """Ответ на запрос поиска"""
    results: List[QueryResult]


class HealthResponse(BaseModel):
    """Состояние системы"""
    status: str
    models_loaded: bool
    vector_db_connected: bool


class StatsResponse(BaseModel):
    """Статистика системы"""
    total_documents: int
    total_chunks: int
    total_images: int
    vector_db_size: str


class ErrorResponse(BaseModel):
    """Стандартизированная ошибка"""
    error: str
    message: str
    details: dict = {}


# Lifespan manager для инициализации/очистки ресурсов
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global rag_engine, config
    
    logger.info("Starting RAG server...")
    
    try:
        # Инициализация RAG engine
        rag_engine = RAGEngine(config)
        await rag_engine.initialize()
        logger.info("RAG engine initialized successfully")
        
        yield
        
    finally:
        # Очистка ресурсов
        logger.info("Shutting down RAG server...")
        if rag_engine:
            await rag_engine.cleanup()


# Создание FastAPI приложения
app = FastAPI(
    title="RAG Documentation System",
    description="Multimodal RAG system for technical documentation",
    version="1.0.0",
    lifespan=lifespan
)


# Middleware для request ID и логирования
@app.middleware("http")
async def add_request_id(request, call_next):
    """Добавляет уникальный ID к каждому запросу"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# API эндпоинты
@app.post("/api/files", response_model=FileAddResponse)
async def add_file(
    content: Optional[str] = Form(None),
    url: str = Form(...),
    images_base_path: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Добавить новый файл документации в систему.
    
    Поддерживает два способа загрузки:
    1. Через form-data с содержимым в поле 'content'
    2. Через multipart upload с файлом
    """
    try:
        # Получение содержимого файла
        if file:
            rst_content = (await file.read()).decode('utf-8')
        elif content:
            rst_content = content
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'content' or 'file' must be provided"
            )
        
        logger.info(f"Adding file from URL: {url}")
        
        # Обработка через RAG engine
        file_id = await rag_engine.add_document(
            rst_content=rst_content,
            document_url=url,
            images_base_path=images_base_path
        )
        
        logger.info(f"File added successfully with ID: {file_id}")
        
        return FileAddResponse(file_id=file_id)
        
    except Exception as e:
        logger.error(f"Error adding file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="file_addition_error",
                message=f"Failed to add file: {str(e)}",
                details={"url": url}
            ).dict()
        )


@app.delete("/api/files/{file_id}", response_model=FileDeleteResponse)
async def delete_file(file_id: str):
    """
    Удалить файл и все связанные данные из системы.
    """
    try:
        logger.info(f"Deleting file: {file_id}")
        
        # Проверка существования файла
        exists = await rag_engine.document_exists(file_id)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error="file_not_found",
                    message=f"File with ID {file_id} not found",
                    details={"file_id": file_id}
                ).dict()
            )
        
        # Удаление через RAG engine
        await rag_engine.delete_document(file_id)
        
        logger.info(f"File deleted successfully: {file_id}")
        
        return FileDeleteResponse()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="file_deletion_error",
                message=f"Failed to delete file: {str(e)}",
                details={"file_id": file_id}
            ).dict()
        )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Выполнить поиск по базе документов.
    
    Поддерживает текстовый поиск и поиск по изображениям.
    """
    try:
        logger.info(f"Processing query: {request.text[:100]}...")
        
        if request.images:
            logger.info(f"Query includes {len(request.images)} images")
        
        # Выполнение поиска через RAG engine
        results = await rag_engine.search(
            query_text=request.text,
            query_images=request.images
        )
        
        logger.info(f"Found {len(results)} results")
        
        # Конвертация результатов в формат API
        query_results = [
            QueryResult(
                document_url=r['document_url'],
                text=r['text'],
                images=r['images'],
                similarity=r['similarity']
            )
            for r in results
        ]
        
        return QueryResponse(results=query_results)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="query_error",
                message=f"Failed to process query: {str(e)}",
                details={"query": request.text[:100]}
            ).dict()
        )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Проверить состояние системы.
    """
    try:
        models_loaded = rag_engine is not None and rag_engine.is_ready()
        vector_db_connected = rag_engine is not None and await rag_engine.check_vector_db()
        
        status = "healthy" if (models_loaded and vector_db_connected) else "degraded"
        
        return HealthResponse(
            status=status,
            models_loaded=models_loaded,
            vector_db_connected=vector_db_connected
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            vector_db_connected=False
        )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Получить статистику по системе.
    """
    try:
        stats = await rag_engine.get_statistics()
        
        return StatsResponse(
            total_documents=stats['total_documents'],
            total_chunks=stats['total_chunks'],
            total_images=stats['total_images'],
            vector_db_size=stats['vector_db_size']
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="stats_error",
                message=f"Failed to get statistics: {str(e)}"
            ).dict()
        )


def main():
    """Точка входа для запуска сервера"""
    global config
    
    parser = argparse.ArgumentParser(description="RAG Documentation Server")
    parser.add_argument(
        '--config',
        type=str,
        default='config.toml',
        help='Path to configuration file (default: config.toml)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Настройка CORS если включено
    if config.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled")
    
    # Запуск сервера
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    main()
