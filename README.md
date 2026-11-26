# Local Second Brain (Hybrid ONNX + PostgreSQL RAG)

Приватная система управления знаниями с изолированными рабочими пространствами, гибридным поиском и локальной генерацией ответов без облачных API.

## Архитектура и стек
- **База и поиск:** PostgreSQL 16 (`pgvector`, `pg_trgm`, двуязычный FTS `russian` + `simple`, графовый индекс `HNSW`).
- **Векторизация (CPU ONNX):** `fastembed` (`intfloat/multilingual-e5-small`, размерность `384`, квантованная модель `model_quantized.onnx` с префиксами `query:` / `passage:` и освобождением памяти после батчей).
- **Генерация:** Ollama (`qwen3:1.7b`, режим `think: false`, `num_ctx: 3072`, `temperature: 0.3`).
- **Парсинг и UI:** `pymupdf4llm` (конвертация PDF в Markdown с сохранением таблиц и кода), `python-docx`, Streamlit, Plotly.

## Ключевые алгоритмы
1. **Code-Aware Markdown Chunking:** нарезка документов по заголовкам (`1400–1800` символов) с защитой блоков кода и дедупликацией файлов по `SHA-256`.
2. **Query Normalization (`text_utils.py`):** автоисправление сбитой раскладки (`QWERTY -> ЙЦУКЕН`) с сохранением SQL-терминов и фильтрацией слов-паразитов.
3. **FULL OUTER JOIN RRF + Hard Gate:** объединение векторного (`HNSW`) и лексического (`websearch_to_tsquery` + `word_similarity`) рейтингов по формуле Reciprocal Rank Fusion с отсечкой `vec_sim >= 0.76 OR kw_rank <= 3` до запуска LLM.
4. **Auto-Merging и порядок чтения:** подтягивание соседнего блока (`chunk_index + 1`) для лучших совпадений и сортировка фрагментов строго по порядку документа (`chunk_index ASC`).

## Быстрый запуск
```bash
git clone https://github.com/ranejeb/Second-brain.git && cd Second-brain
cp .env.example .env
docker compose up -d --build
docker exec -it sb_ollama ollama pull qwen3:1.7b
```
Интерфейс доступен на `http://localhost:8501`.
