# Мультимодальная RAG-система

Система поиска по технической документации с поддержкой текста, таблиц и изображений на основе гибридного подхода (vector + keyword search).

## Оглавление

- [Архитектура системы](#архитектура-системы)
- [Обработка документов](#обработка-документов)
- [Процесс поиска](#процесс-поиска)
- [API эндпоинты](#api-эндпоинты)
- [Тестирование](#тестирование)
- [Установка и запуск](#установка-и-запуск)

## Архитектура системы

Система построена на модульной архитектуре:

### Компоненты

- **Ollama** — локальный inference сервер для работы с моделями
  - `nomic-embed-text:v1.5` — текстовые эмбеддинги (768 dim)
  - `qwen3-vl:4b-instruct` — VLM для анализа изображений (OCR + описание UI)
  - `qwen3:4b-instruct` — LLM для Query Expansion
- **ChromaDB** — векторная база данных для хранения эмбеддингов
- **BM25** — keyword-based поиск с кэшированием индекса
- **Cross-Encoder** — реранкер BAAI/bge-reranker-v2-m3 для финальной сортировки

### Типы чанков

Система создает три типа чанков:
- `text` — текстовые фрагменты с breadcrumbs навигацией
- `table` — таблицы с умным разбиением по строкам
- `image_content` — текстовые описания изображений (OCR + UI analysis)

## Обработка документов

При индексации RST документа выполняется:

1. **Парсинг таблиц** — извлечение grid tables из RST через docutils
2. **Конвертация** — RST → Markdown через pandoc
3. **Chunking текста** — разбиение с учетом структуры заголовков и breadcrumbs
4. **Обработка таблиц** — умное разбиение больших таблиц по строкам с контекстом заголовков
5. **Анализ изображений** — VLM генерирует OCR текст и техническое описание UI
6. **Генерация эмбеддингов** — для всех типов чанков через Ollama
7. **Индексация** — сохранение в ChromaDB + построение BM25 индекса

Физические файлы изображений сохраняются локально, в векторную БД попадают только текстовые описания.

## Процесс поиска

Полный search pipeline:

1. **Query Expansion** (опционально) — LLM генерирует 2-3 поисковые фразы из запроса
2. **Hybrid Retrieval** — параллельный поиск:
   - Vector search (косинусное сходство эмбеддингов)
   - BM25 keyword search (статистический поиск по токенам)
   - Картинки ищутся по описанием, таблицы — по содержимому
   - Объединение результатов с весами
3. **Reranking** (опционально) — Cross-Encoder пересортировывает кандидатов по точной релевантности
4. **Возврат Top-K** — финальные результаты с метаданными

Все этапы настраиваются через `config.toml`.

## API эндпоинты

### `GET /health`
Проверка статуса системы и компонентов.

**Ответ:**
```json
{
  "status": "healthy",
  "version": "4.0.0",
  "documents_count": 10,
  "components": {
    "ollama": true,
    "vector_storage": true,
    "keyword_engine": true,
    "hybrid_retriever": true,
    "query_processor": true,
    "reranker": true
  }
}
```

### `POST /documents`
Добавление документа в систему.

**Параметры:**
```json
{
  "rst_content": "string",
  "document_url": "string",
  "images_base_path": "string",
  "file_id": "string (optional)"
}
```

**Ответ:**
```json
{
  "success": true,
  "file_id": "abc123",
  "message": "Document indexed successfully with 42 chunks",
  "metadata": {
    "text_chunks": 25,
    "table_chunks": 12,
    "image_chunks": 5,
    "total_chunks": 42
  }
}
```

### `GET /documents`
Список всех проиндексированных документов.

### `GET /documents/{file_id}`
Информация о конкретном документе.

### `DELETE /documents/{file_id}`
Полное удаление документа (эмбеддинги + физические файлы + метаданные).

### `POST /search`
Поиск по проиндексированным документам.

**Параметры:**
```json
{
  "query": "string",
  "top_k": 5
}
```

**Ответ:**
```json
{
  "success": true,
  "query": "как настроить интеграцию",
  "results_count": 5,
  "results": [
    {
      "chunk_id": "doc_123_text_5",
      "score": 0.89,
      "content": "текст чанка...",
      "chunk_type": "text",
      "source_engine": "hybrid",
      "metadata": {
        "file_id": "doc_123",
        "document_url": "https://...",
        "breadcrumbs": "Глава 1 > Настройка > Интеграция"
      }
    }
  ]
}
```

## Тестирование

Система тестирования включает сравнение с baseline подходом:

### Baseline RAG
- Простой chunking (1000 токенов)
- Только vector search
- Без VLM, без реранкинга, без query expansion

### Генерация тестовых запросов

LLM автоматически генерирует вопросы на основе содержимого документов:
- Текстовые запросы (~758 из 100 документов)
- Запросы по изображениям (~128 из VLM-описаний)

### Метрики оценки

- **Recall@K** — доля запросов, где правильный документ в топ-K
- **Precision@K** — точность в топ-K результатах
- **MRR** (Mean Reciprocal Rank) — средняя позиция первого правильного ответа
- **NDCG@K** — normalized discounted cumulative gain

### Результаты

Тестирование на 100 документах показало улучшения Full RAG над Baseline:

| Метрика | Baseline | Full RAG | Улучшение |
|---------|----------|----------|-----------|
| Recall@5 | 0.48 | 0.87 | +84% |
| Precision@5 | 0.13 | 0.42 | +213% |
| MRR | 0.38 | 0.79 | +108% |
| NDCG@5 | 0.49 | 1.42 | +189% |

## Установка и запуск

### Требования

- Python 3.11+
- Ollama с установленными моделями
- 8GB+ RAM

### Установка зависимостей

```bash
# Клонировать репозиторий
git clone <repo_url>
cd hse-mml-project

# Установить зависимости
uv sync --frozen
```

### Установка Ollama моделей

```bash
ollama pull nomic-embed-text:v1.5
ollama pull qwen3-vl:4b-instruct
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

### Конфигурация

Отредактируйте `config.toml`:

```toml
[ollama]
base_url = "http://localhost:11434"

[search]
max_results = 5
retrieval_top_k = 50

[query_expansion]
enabled = true

[reranker]
enabled = true
model_name = "BAAI/bge-reranker-v2-m3"
```

### Запуск сервера

```bash
uvicorn app:app --host 0.0.0.0 --port 9876
```

Документация API: http://localhost:9876/docs

### Структура проекта

```
hse-mml-project/
├── app.py                    # FastAPI сервер
├── config.toml               # Конфигурация
├── src/
│   ├── config.py             # Загрузка конфигурации
│   ├── rag/
│   │   ├── main.py           # Главный оркестратор
│   │   ├── ingestion.py      # Обработка документов
│   │   ├── storage.py        # ChromaDB интерфейс
│   │   ├── search.py         # Hybrid retrieval
│   │   ├── query_processing.py  # Query expansion
│   │   ├── ranking.py        # Cross-Encoder reranker
│   │   └── schema.py         # Pydantic модели
│   └── utils/
│       └── logging_config.py
└── testing/
    └── test_rag.py           # Сравнение RAG с бэйзлайном
    └── test_rag_python.py    # Скрипт проверки работы Python SDK
    └── test_rag_requests.py  # Скрипт проверки работы REST API
```
