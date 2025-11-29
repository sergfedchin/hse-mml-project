import asyncio
from pathlib import Path
# from pprint import pprint
from src.config import load_config
from src.rag import RAGSystem
from src.utils.logging_config import setup_logging



async def search_query(rag_system: RAGSystem, query: str, top_k: int = 5):
    print(f'\n\nВыполняем запрос: "{query}"\n')
    results = await rag_system.search(
        user_query=query,
        top_k=top_k,
    )
    print(f"✓ Найдено результатов: {len(results)}\n")
    delimeter = "\n\n" + "-" * 120 + "\n\n"
    print(delimeter + delimeter.join([str(r) for r in results]) + delimeter)

async def main():
    config = load_config("config.toml")
    setup_logging(config.logging)
    rag_system = RAGSystem(config)
    await rag_system.initialize()

    if rag_system.is_ready():
        print("✓ RAG Engine успешно инициализирован!")

    base_url = "https://doc.ru.universe-data.ru/6.13.0-EE"
    docs_base_path = Path("data/sphinx-service-docs-6.13.0-RU/docs/")
    document_paths = [
        # "content/getting_started/datamodel/dqconfigstart.rst",
        # "content/guides/install/distrib.rst",
        # "content/guides/install/hunspell.rst",
        # "content/guides/sdk/backend/high_level_be.rst",
        "content/guides/data_admin/datamodel/attribs/attrtypes.rst",
        # "content/guides/data_admin/classifiers/class_imp_exp.rst",
        # "content/guides/system_admin/accounts/propsacc.rst",
    ]
    doc_ids = []
    for doc_path in [Path(d) for d in document_paths]:
        url = f"{base_url}/{doc_path.with_suffix('.html')}"
        doc_id = await rag_system.add_document(
            rst_content=(docs_base_path / doc_path).read_text(),
            document_url=url,
            images_base_path=docs_base_path.absolute(),
        )
        print(f"✓ Добавлен документ '{docs_base_path / doc_path}': {doc_id}\n")
        doc_ids.append(doc_id)

    # stats = await rag_system.get_statistics()
    # print('\nТекущая статистика RAG-системы:')
    # pprint(stats, width=120)

    queries = [
        # "создать правило качества в расширенном режиме?",
        # "способы аутентификации внешнего пользователя"б
        "Что делает атрибут, если его тип \"локальное перечисление\"?"
    ]
    for q in queries:
        await search_query(rag_system, q, 5)


    print('\n\nУдаляем документы\n')
    for doc_id in doc_ids:
        res = await rag_system.delete_document(doc_id)
        print(f"✓ Документ {doc_id} удален: {res}")

    await rag_system.cleanup()
    print("✓ Ресурсы очищены")

if __name__ == "__main__":
    asyncio.run(main())
