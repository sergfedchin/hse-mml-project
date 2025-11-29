from pathlib import Path
from pprint import pprint

import requests
# from pprint import pprint

def main():
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
    # for doc_path in [Path(d) for d in document_paths]:
    #     url = f"{base_url}/{doc_path.with_suffix('.html')}"
    #     body = {
    #         'rst_content': (docs_base_path / doc_path).read_text(),
    #         'document_url': url,
    #         'images_base_path': str(docs_base_path.absolute()),
    #     }
    #     response = requests.post(
    #         url='http://localhost:8000/documents',
    #         json=body
    #     )
    #     response.raise_for_status()
    #     doc_id = response.json()['file_id']
    #     print(f"✓ Добавлен документ '{docs_base_path / doc_path}': {doc_id}\n")
    #     doc_ids.append(doc_id)

    queries = [
        # "создать правило качества в расширенном режиме?",
        # "способы аутентификации внешнего пользователя"б
        "Что делает атрибут, если его тип \"локальное перечисление\"?"
    ]
    for q in queries:
        body = {
            'query': q,
            'top_k': 5
        }
        print(f'\nВыполняем запрос:\n{q}\n')
        response = requests.post(
            url='http://localhost:8000/search',
            json=body
        )
        response.raise_for_status()
        pprint(response.json(), sort_dicts=False, width=160)


    # print('\n\nУдаляем документы\n')
    # for doc_id in doc_ids:
    #     res = await rag_system.delete_document(doc_id)
    #     print(f"✓ Документ {doc_id} удален: {res}")

    # await rag_system.cleanup()
    # print("✓ Ресурсы очищены")

if __name__ == "__main__":
    main()
