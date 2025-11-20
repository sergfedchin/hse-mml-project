import logging
import asyncio
from pathlib import Path
from pprint import pprint
from src.config import load_config
from src.rag_engine import RAGEngine

# Настройка логирования ДО инициализации RAG Engine
logging.basicConfig(
    level=logging.INFO,  # Уровень логирования (DEBUG для более детальных логов)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Вывод в консоль
    ]
)

# Опционально: установить уровень для конкретных модулей
logging.getLogger('src.rag_engine').setLevel(logging.DEBUG)  # Детальные логи RAG
logging.getLogger('httpx').setLevel(logging.WARNING)  # Меньше шума от HTTP-клиента
logging.getLogger('chromadb').setLevel(logging.WARNING)  # Меньше шума от ChromaDB

async def main():
    config = load_config("config.toml")
    rag_engine = RAGEngine(config)
    await rag_engine.initialize()

    if rag_engine.is_ready():
        print("✓ RAG Engine успешно инициализирован!")

    base_url = "https://doc.ru.universe-data.ru/6.13.0-EE"
    docs_base_path = Path("data/sphinx-service-docs-6.13.0-RU/docs/")
    document_paths = [
        "content/getting_started/datamodel/dqconfigstart.rst",
        "content/guides/install/distrib.rst",
        "content/guides/install/hunspell.rst",
        "content/guides/system_admin/accounts/propsacc.rst",
    ]
    doc_ids = []
    for doc_path in [Path(d) for d in document_paths]:
        url = f"{base_url}/{doc_path.with_suffix('.html')}"
        doc_id = await rag_engine.add_document(
            rst_content=(docs_base_path / doc_path).read_text(),
            document_url=url,
            images_base_path=docs_base_path.absolute(),
        )
        print(f"✓ Добавлен документ '{docs_base_path / doc_path}': {doc_id}\n")
        doc_ids.append(doc_id)

    # stats = await rag_engine.get_statistics()
    # print('\nТекущая статистика RAG-системы:')
    # pprint(stats, width=120)

    query = "создать правило качества в расширенном режиме"
    print(f'\n\nВыполняем запрос: "{query}"\n')
    results = await rag_engine.hybrid_search(
        query=query,
        top_k=5,
    )
    print(f"✓ Найдено результатов: {len(results)}\n")
    print(("\n\n" + "-" * 80 + "\n\n").join([str(r) for r in results]))

    print('\n\nУдаляем документы\n')
    for doc_id in doc_ids:
        res = await rag_engine.delete_document(doc_id)
        print(f"✓ Документ {doc_id} удален: {res}")

    await rag_engine.cleanup()
    print("✓ Ресурсы очищены")

if __name__ == "__main__":
    asyncio.run(main())
