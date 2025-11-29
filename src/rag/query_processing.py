"""
Обработка пользовательских запросов: Query Expansion через LLM
"""

from typing import List

import ollama

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


QUERY_EXPANSION_PROMPT = """Ты - эксперт по поиску информации в технической документации.

Пользователь задал вопрос или описал проблему. Твоя задача - преобразовать его запрос в 2-3 точных поисковых фразы для поиска в документации.

Правила:
1. Каждая фраза должна быть короткой (2-5 слов)
2. Фразы должны быть РАЗНЫМИ и покрывать разные аспекты вопроса
3. Используй технические термины, если они есть в запросе
4. Избегай общих слов ("как", "что", "где")
5. Отвечай ТОЛЬКО списком фраз через запятую, без нумерации и пояснений

Примеры:
Входной запрос: "У меня не открывается форма редактирования пользователя, выдает ошибку 502"
Ответ: ошибка 502, форма редактирования пользователя, bad gateway

Входной запрос: "Как настроить интеграцию с внешней системой через API?"
Ответ: настройка API, интеграция внешняя система, конфигурация подключения

Входной запрос пользователя: 
```
{user_query}
```

Твой ответ (только фразы через запятую):"""


class QueryProcessor:
    """Обработчик запросов: расширение через LLM."""

    def __init__(
        self, ollama_client: ollama.Client, llm_model: str, max_queries: int = 3
    ):
        self.ollama_client = ollama_client
        self.llm_model = llm_model
        self.max_queries = max_queries

    async def expand_query(self, user_query: str) -> List[str]:
        """
        Расширяет пользовательский запрос в набор поисковых фраз.

        Args:
            user_query: Исходный запрос пользователя

        Returns:
            Список из 2-N поисковых фраз (N = self.max_queries)
        """
        try:
            prompt = QUERY_EXPANSION_PROMPT.format(user_query=user_query)

            response = self.ollama_client.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.3,
                    "num_predict": 100,
                },
                stream=False,
            )

            answer = response.message.content.strip()

            # Парсим ответ
            phrases = [phrase.strip() for phrase in answer.split(",")]
            phrases = [p for p in phrases if p and len(p) > 2]

            # Ограничиваем max_queries фразами
            phrases = phrases[: self.max_queries]

            # Если LLM не вернул фразы, используем оригинальный запрос
            if not phrases:
                phrases = [user_query]

            logger.info(f"Query expanded: '{user_query}' -> {phrases}")

            return phrases

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            # Fallback: возвращаем оригинальный запрос
            return [user_query]
